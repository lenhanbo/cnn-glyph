import inspect
from types import SimpleNamespace

import torch
from torch import nn

from models.model import VATr, ctc_loss_fp32, vatr_effective_objective


def test_ctc_loss_is_forced_to_fp32_and_backpropagates_to_half_logits():
    # The OCR network returns raw [time, batch, class] logits.
    raw = torch.randn(6, 1, 5, dtype=torch.float16, requires_grad=True)
    targets = torch.tensor([1, 2], dtype=torch.long)
    input_lengths = torch.tensor([6], dtype=torch.long)
    target_lengths = torch.tensor([2], dtype=torch.long)
    loss = ctc_loss_fp32(nn.CTCLoss(), raw, targets, input_lengths, target_lengths)
    expected = nn.CTCLoss()(
        raw.float().log_softmax(2), targets, input_lengths, target_lengths
    )
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, expected)
    assert loss.item() >= 0
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_active_generator_auxiliary_branches_backward_exactly_once():
    ocr_source = inspect.getsource(VATr.backward_G_only)
    writer_source = inspect.getsource(VATr.backward_G_WL)
    assert ocr_source.count(".backward(") == 1
    assert writer_source.count(".backward(") == 1
    assert "create_graph=True" not in ocr_source
    assert "create_graph=True" not in writer_source


def test_single_objective_matches_vatr_three_backward_accumulation():
    adv = torch.tensor(2.0, requires_grad=True)
    aux = torch.tensor(5.0, requires_grad=True)
    alpha = torch.tensor(0.25)

    objective = vatr_effective_objective(adv, aux, alpha)
    objective.backward()

    assert objective.item() == 2.0 * 3.0 + 5.0 * (1.0 + 2.0 * 0.25)
    assert adv.grad.item() == 3.0
    assert aux.grad.item() == 1.5


class _WarmupConverter:
    def encode(self, labels):
        targets = torch.tensor([[1], [2]], dtype=torch.long)
        lengths = torch.ones(len(labels), dtype=torch.long)
        return targets[:len(labels)], lengths, None


class _TinyOCR(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(6, 3))

    def forward(self, images):
        return self.logits.unsqueeze(1).expand(-1, images.shape[0], -1)


def test_ocr_real_only_warmup_updates_ocr_without_generator_forward():
    model = VATr.__new__(VATr)
    nn.Module.__init__(model)
    model.args = SimpleNamespace(device="cpu")
    model.netconverter = _WarmupConverter()
    model.netOCR = _TinyOCR()
    model.OCR_criterion = nn.CTCLoss(zero_infinity=True, reduction="none")
    model.optimizer_OCR = torch.optim.Adam(model.netOCR.parameters(), lr=1e-2)
    model._set_input({
        "img": torch.randn(2, 1, 32, 16),
        "label": [b"a", b"b"],
    })
    before = model.netOCR.logits.detach().clone()

    loss = model.optimize_OCR_real_only()

    assert torch.isfinite(loss)
    assert not torch.equal(before, model.netOCR.logits.detach())
    assert model.loss_G.item() == 0
    assert model.loss_D.item() == 0
    assert model.loss_OCR_fake.item() == 0
