import random

import numpy as np
import torch
from torch import nn

from models.feature_query import FeatureMapQuery
from models.stage_manager import capture_rng_state, restore_rng_state, switch_generator_to_stage2


class TinyGenerator(nn.Module):
    def __init__(self, unifont):
        super().__init__()
        self.query_embed = FeatureMapQuery(mode="stage1", unifont_path=unifont)
        self.downstream = nn.Linear(512, 2)


def test_optimizer_survives_switch_and_rng_restores(tmp_path):
    unifont = __import__("pathlib").Path(__file__).parents[1] / "files" / "unifont.pickle"
    generator = TinyGenerator(unifont)
    optimizer = torch.optim.Adam(generator.parameters(), lr=5e-5, betas=(0.0, 0.999))
    loss = generator.downstream(generator.query_embed(torch.tensor([[1, 2]]))).sum()
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    downstream_step = optimizer.state[generator.downstream.weight]["step"].clone()
    optimizer = switch_generator_to_stage2(generator, optimizer)
    assert optimizer.state[generator.downstream.weight]["step"] == downstream_step
    assert any(generator.query_embed.feature_maps is p for g in optimizer.param_groups for p in g["params"])

    random.seed(11); np.random.seed(11); torch.manual_seed(11)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1))
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2])
