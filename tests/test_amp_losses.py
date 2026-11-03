import torch
from torch import nn

from models.model import ctc_loss_fp32


def test_ctc_loss_is_forced_to_fp32_and_backpropagates_to_half_logits():
    # [time, batch, classes], already log-normalized as expected by CTCLoss.
    raw = torch.randn(6, 1, 5, dtype=torch.float16, requires_grad=True)
    predictions = raw.float().log_softmax(2).half()
    targets = torch.tensor([1, 2], dtype=torch.long)
    input_lengths = torch.tensor([6], dtype=torch.long)
    target_lengths = torch.tensor([2], dtype=torch.long)
    loss = ctc_loss_fp32(nn.CTCLoss(), predictions, targets, input_lengths, target_lengths)
    assert loss.dtype == torch.float32
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
