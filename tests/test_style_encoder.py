import pytest
import torch

from models.model import DepthwiseSeparableBlock, LiteStyleCNN


def test_lightweight_style_encoder_contract():
    encoder = LiteStyleCNN()
    output = encoder(torch.randn(2, 1, 32, 192))

    assert output.shape == (2, 512, 1, 6)
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 181_632
    assert len(encoder.blocks) == 4
    assert all(isinstance(block, DepthwiseSeparableBlock) for block in encoder.blocks)


def test_lightweight_style_encoder_pads_partial_width_patch():
    encoder = LiteStyleCNN()
    output = encoder(torch.randn(1, 1, 32, 193))

    assert output.shape == (1, 512, 1, 7)


def test_lightweight_style_encoder_rejects_wrong_height():
    encoder = LiteStyleCNN()

    with pytest.raises(ValueError, match="height 32"):
        encoder(torch.randn(1, 1, 31, 192))
