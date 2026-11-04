from pathlib import Path

import torch

from models.feature_query import FeatureMapQuery, GlyphCNN, LiteGlyphCNN, RESIZE_ALGORITHM, load_gnu_glyph_bank
from models.vocabulary import IAM_ALPHABET, SPECIAL_ALPHABET, char_to_id


ROOT = Path(__file__).parents[1]
UNIFONT = ROOT / "files" / "unifont.pickle"


def test_fixed_vocabulary_contract():
    mapping = char_to_id()
    assert len(IAM_ALPHABET) == 80
    assert len(SPECIAL_ALPHABET) == 49
    assert list(mapping.values()) == list(range(1, 130))
    assert mapping[IAM_ALPHABET[0]] == 1
    assert mapping[SPECIAL_ALPHABET[-1]] == 129


def test_nearest_resize_and_metadata():
    glyphs, metadata = load_gnu_glyph_bank(UNIFONT)
    assert glyphs.shape == (130, 1, 32, 32)
    assert torch.count_nonzero(glyphs[0]) == 0
    assert metadata["resize_algorithm"] == RESIZE_ALGORITHM
    for y in range(0, 32, 2):
        for x in range(0, 32, 2):
            assert torch.all(glyphs[:, :, y:y + 2, x:x + 2] == glyphs[:, :, y:y + 1, x:x + 1])


def test_cnn_shape_is_64_by_8_by_8():
    cnn = GlyphCNN()
    assert isinstance(cnn, LiteGlyphCNN)
    assert cnn(torch.randn(7, 1, 32, 32)).shape == (7, 64, 8, 8)
    assert sum(parameter.numel() for parameter in cnn.parameters()) == 7_808


def test_attention_precedes_pool_and_padding_is_isolated():
    module = FeatureMapQuery(mode="stage1", unifont_path=UNIFONT).eval()
    called = []
    hook = module.spatial.attention.register_forward_hook(lambda *_: called.append(True))
    with torch.no_grad():
        compact = module(torch.tensor([[1, 2, 3]]))
        padded = module(torch.tensor([[0, 1, 0, 2, 3, 0]]))
    hook.remove()
    assert called
    torch.testing.assert_close(compact[0], padded[0, [1, 3, 4]], rtol=0, atol=0)
    assert torch.count_nonzero(padded[0, [0, 2, 5]]) == 0


def test_stage_gradients_and_materialization_equivalence(tmp_path):
    module = FeatureMapQuery(mode="stage1", unifont_path=UNIFONT).eval()
    ids = torch.tensor([[0, 1, 80, 81, 129]])
    before = module(ids)
    before.sum().backward()
    assert any(parameter.grad is not None for parameter in module.cnn.parameters())
    module.zero_grad(set_to_none=True)
    module.switch_to_stage2()
    after = module(ids)
    torch.testing.assert_close(before.detach(), after.detach(), rtol=0, atol=1e-6)
    after.sum().backward()
    assert module.feature_maps.grad is not None
    assert torch.count_nonzero(module.feature_maps.grad[0]) == 0
    assert not hasattr(module, "cnn")
    assert not hasattr(module, "glyph_bank")
    keys = module.state_dict().keys()
    assert not any(key.startswith("cnn.") or key == "glyph_bank" for key in keys)
    assert any(key.startswith("spatial.pos_cnn.") for key in keys)  # PosCNN intentionally survives stage 2.

    state_path = tmp_path / "stage2.pt"
    torch.save(module.state_dict(), state_path)
    no_gnu = FeatureMapQuery(mode="stage2")
    no_gnu.load_state_dict(torch.load(state_path, weights_only=True))
    torch.testing.assert_close(after.detach(), no_gnu(ids).detach())


def test_out_of_vocabulary_is_an_error():
    module = FeatureMapQuery(mode="stage2")
    try:
        module(torch.tensor([[130]]))
    except IndexError:
        pass
    else:
        raise AssertionError("out-of-vocabulary ID did not raise")


def test_amp_output_uses_computed_dtype_and_keeps_gradients():
    module = FeatureMapQuery(mode="stage2")
    ids = torch.tensor([[0, 1, 2, 0]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = module(ids)
        loss = output.float().sum()
    assert output.dtype == torch.bfloat16
    assert torch.count_nonzero(output[:, [0, 3]]) == 0
    loss.backward()
    assert module.feature_maps.grad is not None
