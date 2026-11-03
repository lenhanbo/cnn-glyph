import csv
from types import SimpleNamespace

import torch
from torch import nn

from train_feature_map import MetricsLogger, save_preview


def test_metrics_csv_appends_once_and_preserves_header(tmp_path):
    path = tmp_path / "metrics.csv"
    logger = MetricsLogger(path)
    logger.write({"epoch": 1, "batch": 1, "global_step": 1, "G": 2.5})
    logger.write({"epoch": 1, "batch": 2, "global_step": 2, "G": 1.5})
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["G"] == "2.5"
    assert rows[1]["global_step"] == "2"


class _Converter:
    def encode(self, labels):
        lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
        return torch.ones(len(labels), int(lengths.max()), dtype=torch.long), lengths, None


class _Generator(nn.Module):
    def Eval(self, styles, encoded, positions):
        return [torch.zeros(styles.shape[0], 1, 32, encoded.shape[-1] * 16, device=styles.device)]


class _PreviewModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.netconverter = _Converter()
        self.netG = _Generator()
        self.args = SimpleNamespace(resolution=16)


def test_preview_writes_style_real_and_fake_panel(tmp_path):
    data = {
        "label": [b"Hi", b"VATr"],
        "simg": torch.zeros(2, 3, 32, 20),
        "swids": torch.tensor([[12, 14, 16], [11, 13, 15]]),
        "img": torch.zeros(2, 1, 32, 40),
    }
    path = tmp_path / "preview.png"
    save_preview(_PreviewModel(), data, path, "cpu", max_samples=2)
    assert path.is_file()
