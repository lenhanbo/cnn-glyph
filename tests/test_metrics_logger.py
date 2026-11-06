import csv
import sys
from types import SimpleNamespace

import torch
from torch import nn

from train_feature_map import MetricsLogger, download_hf_checkpoint, save_preview, save_token_preview


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
    assert "OCR_fake_raw" in rows[0]
    assert "alpha_ocr" in rows[0]
    assert "w_fake_raw" in rows[0]
    assert "alpha_w" in rows[0]


class _Converter:
    def encode(self, labels):
        lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
        return torch.ones(len(labels), int(lengths.max()), dtype=torch.long), lengths, None


class _Generator(nn.Module):
    def Eval(self, styles, encoded, positions):
        return [torch.zeros(styles.shape[0], 1, 32, encoded.shape[-1] * 16, device=styles.device)]

    def inspect_character_layers(self, styles, encoded, last_layers):
        batch, characters = encoded.shape
        layers = min(last_layers, 2)
        return {
            "query_tokens": torch.arange(512, device=styles.device).float().repeat(batch, characters, 1),
            "decoder_tokens": torch.randn(layers, batch, characters, 512, device=styles.device),
            "layer_images": torch.zeros(layers, batch, 1, 32, characters * 16, device=styles.device),
            "layer_numbers": list(range(1, layers + 1)),
            "glyphs": torch.zeros(batch, characters, 1, 32, 32, device=styles.device),
        }


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


def test_token_preview_writes_per_character_layer_table(tmp_path):
    data = {
        "label": [b"Hi", b"VATr"],
        "simg": torch.zeros(2, 1, 32, 192),
        "swids": torch.tensor([[20], [24]]),
        "img": torch.zeros(2, 1, 32, 40),
    }
    path = tmp_path / "tokens.png"
    save_token_preview(_PreviewModel(), data, path, "cpu", max_samples=2, last_layers=2)
    assert path.is_file()


def test_hf_checkpoint_download_uses_dataset_repo_and_local_dir(tmp_path, monkeypatch):
    calls = {}

    def fake_download(**kwargs):
        calls.update(kwargs)
        path = tmp_path / "hub" / "checkpoints" / "latest.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fake_download))
    result = download_hf_checkpoint(
        "lenhanbo/vatrcnn", "checkpoints/latest.pth", "dataset", tmp_path / "hub"
    )
    assert result.is_file()
    assert calls == {
        "repo_id": "lenhanbo/vatrcnn",
        "filename": "checkpoints/latest.pth",
        "repo_type": "dataset",
        "local_dir": str(tmp_path / "hub"),
    }


def test_hf_checkpoint_download_auto_selects_only_pth(tmp_path, monkeypatch):
    calls = {}

    class FakeApi:
        def list_repo_files(self, **kwargs):
            assert kwargs == {"repo_id": "lenhanbo/vatrcnn", "repo_type": "dataset"}
            return [".gitattributes", "latest (4).pth"]

    def fake_download(**kwargs):
        calls.update(kwargs)
        path = tmp_path / "hub" / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, hf_hub_download=fake_download),
    )
    result = download_hf_checkpoint("lenhanbo/vatrcnn", None, "dataset", tmp_path / "hub")
    assert result.name == "latest (4).pth"
    assert calls["filename"] == "latest (4).pth"
