import torch

from models.model import load_pretrained_ocr


def test_load_pretrained_ocr_from_full_vatr_checkpoint():
    source = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    target = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    checkpoint = {
        "model": {f"netOCR.{key}": value.clone() for key, value in source.state_dict().items()}
    }

    load_pretrained_ocr(target, checkpoint)

    for expected, actual in zip(source.parameters(), target.parameters()):
        assert torch.equal(expected, actual)


def test_load_pretrained_ocr_rejects_incompatible_checkpoint():
    target = torch.nn.Linear(3, 4)
    try:
        load_pretrained_ocr(target, {"model": {"netOCR.wrong": torch.ones(1)}})
    except RuntimeError as error:
        assert "Incompatible OCR checkpoint" in str(error)
    else:
        raise AssertionError("incompatible OCR checkpoint was accepted")


def test_load_pretrained_ocr_upgrades_legacy_ctc_classifier():
    class TinyOCR(torch.nn.Module):
        def __init__(self, outputs):
            super().__init__()
            self.linear = torch.nn.Linear(3, outputs)

    source = TinyOCR(4)
    target = TinyOCR(5)
    new_row_before = target.linear.weight[-1].detach().clone()
    new_bias_before = target.linear.bias[-1].detach().clone()
    checkpoint = {
        "model": {
            f"netOCR.{key}": value.clone()
            for key, value in source.state_dict().items()
        }
    }

    load_pretrained_ocr(target, checkpoint)

    torch.testing.assert_close(target.linear.weight[:4], source.linear.weight)
    torch.testing.assert_close(target.linear.bias[:4], source.linear.bias)
    torch.testing.assert_close(target.linear.weight[-1], new_row_before)
    torch.testing.assert_close(target.linear.bias[-1], new_bias_before)
