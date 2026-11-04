import torch
from torch import nn
from torchvision import models

from models.model import load_pretrained_style_encoder


def test_pretrained_resnet_style_encoder_contract_and_grayscale_conversion():
    source = models.resnet18(weights=None)
    checkpoint = {"model": source.state_dict()}

    encoder = models.resnet18(weights=None)
    encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    encoder.fc = nn.Identity()
    encoder.avgpool = nn.Identity()
    load_pretrained_style_encoder(encoder, checkpoint)

    expected = source.conv1.weight.mean(dim=1, keepdim=True)
    torch.testing.assert_close(encoder.conv1.weight, expected)
    features = encoder(torch.randn(2, 1, 32, 192))
    assert features.shape == (2, 512 * 6)
    assert features.view(2, 512, 1, -1).shape == (2, 512, 1, 6)
