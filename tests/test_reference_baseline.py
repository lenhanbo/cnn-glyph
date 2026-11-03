import torch

from models.unifont_module import UnifontModule
from models.vocabulary import IAM_ALPHABET, SPECIAL_ALPHABET


def test_official_baseline_state_dict_layout_round_trip():
    """The reference path retains official VATr key names and tensor shapes."""
    source = UnifontModule(512, IAM_ALPHABET + SPECIAL_ALPHABET, device="cpu")
    old_style_checkpoint = source.state_dict()
    assert set(old_style_checkpoint) == {"linear.weight", "linear.bias"}
    target = UnifontModule(512, IAM_ALPHABET + SPECIAL_ALPHABET, device="cpu")
    target.load_state_dict(old_style_checkpoint, strict=True)
    ids = torch.tensor([[0, 1, 80, 81, 129]])
    torch.testing.assert_close(source(ids), target(ids))
