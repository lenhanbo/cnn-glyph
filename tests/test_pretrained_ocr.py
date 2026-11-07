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
