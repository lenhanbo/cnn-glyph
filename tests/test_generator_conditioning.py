import torch
from torch import nn

from models.model import Generator


def test_feature_query_residual_prevents_identical_decoder_tokens():
    generator = Generator.__new__(Generator)
    nn.Module.__init__(generator)
    generator.use_feature_query_residual = True
    generator.content_norm = nn.LayerNorm(4)

    decoder_tokens = torch.ones(1, 3, 4)
    query_tokens = torch.eye(4)[:3].unsqueeze(0)
    fused = generator._fuse_character_residual(decoder_tokens, query_tokens)

    assert not torch.equal(fused[:, 0], fused[:, 1])
    assert not torch.equal(fused[:, 1], fused[:, 2])


def test_reference_baseline_does_not_add_feature_query_residual():
    generator = Generator.__new__(Generator)
    nn.Module.__init__(generator)
    generator.use_feature_query_residual = False
    decoder_tokens = torch.randn(1, 3, 4)
    query_tokens = torch.randn(1, 3, 4)

    assert generator._fuse_character_residual(decoder_tokens, query_tokens) is decoder_tokens


def test_independent_character_render_preserves_tile_order():
    generator = Generator.__new__(Generator)
    nn.Module.__init__(generator)

    def render_one(tokens):
        # One scalar pixel per one-character input is enough to verify the
        # batch/character flattening and horizontal concatenation order.
        return tokens[:, :, :1].reshape(tokens.shape[0], 1, 1, 1)

    generator._render_character_tokens = render_one
    tokens = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])

    rendered = generator._render_character_tokens_independently(tokens)

    assert rendered.shape == (2, 1, 1, 3)
    torch.testing.assert_close(rendered[:, 0, 0], torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
