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
