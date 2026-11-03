from pathlib import Path

import numpy as np
from PIL import Image

from data.raw_iam_dataset import RawIAMDataset


def _write_word(root: Path, word_id: str, value: int = 255):
    form = word_id.rsplit("-", 2)[0]
    group = form.split("-")[0]
    path = root / "words" / group / form / f"{word_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 24), value, dtype=np.uint8)).save(path)
    return path.relative_to(root).as_posix()


def test_raw_iam_form_grouping_and_shape(tmp_path):
    first = _write_word(tmp_path, "a01-000u-00-00")
    second = _write_word(tmp_path, "a01-000u-00-01", 0)
    third = _write_word(tmp_path, "a01-000x-00-00")
    text = f"{first}\tA\n{second}\tMOVE\n{third}\tto\n"
    (tmp_path / "train_gt.txt").write_text(text, encoding="utf-8")
    (tmp_path / "val_gt.txt").write_text(text, encoding="utf-8")
    dataset = RawIAMDataset(tmp_path, num_examples=3)
    assert dataset.num_writers == 2
    item = dataset[0]
    assert item["simg"].shape == (3, 32, 192)
    assert item["img"].shape[1] == 32
    assert item["simg"].min() >= -1 and item["simg"].max() <= 1


def test_raw_iam_official_writer_mapping(tmp_path):
    first = _write_word(tmp_path, "a01-000u-00-00")
    second = _write_word(tmp_path, "a01-000x-00-00")
    text = f"{first}\tA\n{second}\tto\n"
    (tmp_path / "train_gt.txt").write_text(text, encoding="utf-8")
    (tmp_path / "val_gt.txt").write_text(text, encoding="utf-8")
    forms = tmp_path / "forms.txt"
    forms.write_text("a01-000u 000\na01-000x 000\n", encoding="utf-8")
    dataset = RawIAMDataset(tmp_path, group_by="writer", forms_path=forms)
    assert dataset.num_writers == 1
