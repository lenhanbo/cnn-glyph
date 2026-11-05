import torch
from torch import nn

from models.model import ctc_loss_fp32


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
