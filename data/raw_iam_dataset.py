"""Lazy IAM word loader for a raw ``iam_dataset`` directory."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from data.dataset import TextCollator
from models.vocabulary import IAM_ALPHABET


class RawIAMDataset(Dataset):
    """Read ``path<TAB>label`` ground truth without creating a large pickle.

    The supplied raw dataset has no form-to-writer metadata, so ``group_by=form``
    treats each IAM form as one style class. Every form is written by one person,
    which keeps style sampling valid, but repeated forms by the same person are
    separate writer-loss classes. Supply official forms.txt for true writer IDs.
    """

    def __init__(self, root, split="train", num_examples=1, collator_resolution=16,
                 min_virtual_size=0, group_by="form", forms_path=None):
        self.root = Path(root).resolve()
        self.num_examples = int(num_examples)
        self.min_virtual_size = int(min_virtual_size)
        self.collate_fn = TextCollator(collator_resolution)
        gt_name = "train_gt.txt" if split == "train" else "val_gt.txt"
        gt_path = self.root / gt_name
        if not gt_path.is_file() or not (self.root / "words").is_dir():
            raise FileNotFoundError(f"Expected words/ and {gt_name} below {self.root}")
        form_to_writer = self._read_forms(forms_path) if forms_path else {}
        if group_by == "writer" and not form_to_writer:
            raise ValueError("group_by=writer requires official IAM forms.txt")

        groups = {}
        skipped = 0
        for line_number, line in enumerate(gt_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                relative, label = line.split("\t", 1)
            except ValueError as error:
                raise ValueError(f"Malformed {gt_path}:{line_number}") from error
            if not label or any(char not in IAM_ALPHABET for char in label):
                skipped += 1
                continue
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Path escapes dataset root: {relative}")
            image_path = self.root / relative_path
            form_id = image_path.stem.rsplit("-", 2)[0]
            style_id = form_to_writer.get(form_id, form_id)
            groups.setdefault(style_id, []).append((image_path, label))
        self.style_ids = sorted(key for key, items in groups.items() if items)
        self.groups = {key: groups[key] for key in self.style_ids}
        self.skipped_labels = skipped
        if not self.style_ids:
            raise ValueError(f"No usable IAM examples in {gt_path}")

    @staticmethod
    def _read_forms(path):
        mapping = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) >= 2:
                mapping[fields[0]] = fields[1]
        return mapping

    @property
    def num_writers(self):
        return len(self.style_ids)

    def __len__(self):
        return max(len(self.style_ids), self.min_virtual_size)

    @staticmethod
    def _load_image(path):
        with Image.open(path) as source:
            image = source.convert("L")
        width = max(1, round(image.width * 32 / image.height))
        return TF.to_tensor(image.resize((width, 32), Image.Resampling.BICUBIC)) * 2 - 1

    def __getitem__(self, index):
        writer_index = index % len(self.style_ids)
        records = self.groups[self.style_ids[writer_index]]
        chosen = np.random.choice(len(records), self.num_examples, replace=True)
        style_images = []
        style_widths = []
        for record_index in chosen:
            image = self._load_image(records[int(record_index)][0])
            width = min(image.shape[-1], 192)
            canvas = torch.ones(1, 32, 192)
            canvas[:, :, :width] = image[:, :, :width]
            style_images.append(canvas)
            style_widths.append(width)
        real_path, real_label = random.choice(records)
        return {
            "simg": torch.cat(style_images, 0), "swids": style_widths,
            "img": self._load_image(real_path), "label": real_label.encode("utf-8"),
            "img_path": str(real_path), "idx": index, "wcl": writer_index,
        }
