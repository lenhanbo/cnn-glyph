from pathlib import Path

import gdown

from util.pretrained import ensure_pretrained_style_encoder


def test_pretrained_download_is_atomic_and_cached(tmp_path, monkeypatch):
    destination = tmp_path / "resnet_18_pretrained.pth"
    calls = []

    def fake_download(*, id, output, quiet):
        calls.append((id, output, quiet))
        Path(output).write_bytes(b"checkpoint")
        return output

    monkeypatch.setattr(gdown, "download", fake_download)
    assert ensure_pretrained_style_encoder(destination) == str(destination)
    assert destination.read_bytes() == b"checkpoint"
    assert len(calls) == 1

    assert ensure_pretrained_style_encoder(destination) == str(destination)
    assert len(calls) == 1
