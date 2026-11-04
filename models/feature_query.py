"""Two-stage GNU glyph query encoder for VATr.

Stage 1 learns a CNN over immutable 32x32 nearest-neighbour GNU glyphs.
Stage 2 replaces the glyph/CNN path with its materialized 64x8x8 feature table.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .vocabulary import IAM_ALPHABET, SPECIAL_ALPHABET, VOCAB_SIZE_WITH_PADDING

RESIZE_ALGORITHM = "nearest-repeat-2-v1"


def load_gnu_glyph_bank(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return [130,1,32,32], with row zero reserved for padding."""
    path = Path(path)
    raw = path.read_bytes()
    records = pickle.loads(raw)
    by_codepoint = {int(item["idx"][0]): np.asarray(item["mat"], dtype=np.float32) for item in records}
    rows = [np.zeros((16, 16), dtype=np.float32)]
    missing = []
    for char in IAM_ALPHABET + SPECIAL_ALPHABET:
        matrix = by_codepoint.get(ord(char))
        if matrix is None:
            missing.append(f"{char!r} (U+{ord(char):04X})")
        else:
            if matrix.shape != (16, 16):
                raise ValueError(f"GNU glyph U+{ord(char):04X} is {matrix.shape}, expected (16, 16)")
            rows.append(matrix)
    if missing:
        raise KeyError("GNU file is missing vocabulary glyphs: " + ", ".join(missing))
    glyphs16 = torch.from_numpy(np.stack(rows)).unsqueeze(1)
    glyphs32 = glyphs16.repeat_interleave(2, -2).repeat_interleave(2, -1)
    metadata = {
        "source": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "resize_algorithm": RESIZE_ALGORITHM,
        "source_shape": [VOCAB_SIZE_WITH_PADDING, 1, 16, 16],
        "shape": list(glyphs32.shape),
    }
    return glyphs32, metadata


def save_glyph_cache(unifont_path: str | Path, tensor_path: str | Path, metadata_path: str | Path) -> None:
    glyphs, metadata = load_gnu_glyph_bank(unifont_path)
    torch.save(glyphs, tensor_path)
    Path(metadata_path).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1,
                      groups=in_channels, bias=False),
            nn.GroupNorm(8, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LiteGlyphCNN(nn.Module):
    """Depthwise-separable GNU glyph encoder: [N,1,32,32] -> [N,64,8,8]."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            DepthwiseSeparableBlock(32, 64, stride=2),
            DepthwiseSeparableBlock(64, 64),
        )

    def forward(self, glyphs: torch.Tensor) -> torch.Tensor:
        result = self.network(glyphs)
        if result.shape[1:] != (64, 8, 8):
            raise RuntimeError(f"LiteGlyphCNN returned {tuple(result.shape[1:])}, expected (64, 8, 8)")
        return result


# Keep the public name used by existing stage-management code and tests.
GlyphCNN = LiteGlyphCNN


class SpatialAttentionPool(nn.Module):
    """PosCNN followed by three pre-norm spatial self-attention blocks."""
    def __init__(self, channels: int = 64, heads: int = 4, layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.pos_cnn = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=heads, dim_feedforward=channels * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        # Pre-norm is intentional. PyTorch cannot use its nested-tensor fast
        # path with norm_first=True, so disable that optional path explicitly
        # instead of emitting a warning on every construction.
        self.attention = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        maps = maps + self.pos_cnn(maps)
        tokens = maps.flatten(2).transpose(1, 2)  # [glyph, 64 spatial positions, 64 channels]
        tokens = self.attention(tokens)
        return self.norm(tokens).mean(dim=1)


class FeatureMapQuery(nn.Module):
    MODES = {"stage1", "stage2"}

    def __init__(self, out_dim: int = 512, mode: str = "stage1", unifont_path: str | Path | None = None,
                 chunk_size: int = 128, feature_maps: torch.Tensor | None = None):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown feature query mode: {mode}")
        self.mode = mode
        self.chunk_size = int(chunk_size)
        self.spatial = SpatialAttentionPool()
        self.projection = nn.Linear(64, out_dim)
        self.cache_metadata: dict[str, Any] = {}
        if mode == "stage1":
            if unifont_path is None:
                raise ValueError("stage1 requires unifont_path")
            glyphs, self.cache_metadata = load_gnu_glyph_bank(unifont_path)
            self.register_buffer("glyph_bank", glyphs, persistent=True)
            self.cnn = LiteGlyphCNN()
        else:
            if feature_maps is None:
                feature_maps = torch.zeros(VOCAB_SIZE_WITH_PADDING, 64, 8, 8)
            self._install_feature_maps(feature_maps)

    def _install_feature_maps(self, maps: torch.Tensor) -> None:
        if tuple(maps.shape) != (VOCAB_SIZE_WITH_PADDING, 64, 8, 8):
            raise ValueError(f"feature table must be [130,64,8,8], got {list(maps.shape)}")
        maps = maps.detach().float().clone()
        maps[0].zero_()
        self.feature_maps = nn.Parameter(maps)
        self.feature_maps.register_hook(self._zero_padding_gradient)

    @staticmethod
    def _zero_padding_gradient(gradient: torch.Tensor) -> torch.Tensor:
        gradient = gradient.clone()
        gradient[0].zero_()
        return gradient

    def _validate_ids(self, ids: torch.Tensor) -> None:
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("character IDs must be an integer tensor")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= VOCAB_SIZE_WITH_PADDING):
            raise IndexError("character ID is outside the fixed VATr vocabulary [0, 129]")

    def _maps_for_ids(self, ids: torch.Tensor) -> torch.Tensor:
        if self.mode == "stage1":
            return self.cnn(self.glyph_bank[ids])
        return self.feature_maps[ids]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        self._validate_ids(ids)
        flat = ids.reshape(-1)
        output = None
        valid_positions = torch.nonzero(flat != 0, as_tuple=False).flatten()
        for positions in valid_positions.split(self.chunk_size):
            maps = self._maps_for_ids(flat[positions])
            embeddings = self.projection(self.spatial(maps))
            if output is None:
                # Under AMP embeddings are FP16/BF16 while projection weights
                # remain FP32. Build padding/output in the computed dtype so
                # index_copy is valid and the query stays on the AMP path.
                output = embeddings.new_zeros(flat.numel(), self.projection.out_features)
            output = output.index_copy(0, positions, embeddings)
        if output is None:  # an all-padding query has no computed chunk
            output = self.projection.weight.new_zeros(flat.numel(), self.projection.out_features)
        return output.view(*ids.shape, -1)

    @torch.no_grad()
    def materialize(self, chunk_size: int | None = None) -> torch.Tensor:
        if self.mode != "stage1":
            return self.feature_maps.detach().float().clone()
        was_training = self.training
        self.eval()
        size = chunk_size or self.chunk_size
        device = self.glyph_bank.device
        table = torch.zeros(VOCAB_SIZE_WITH_PADDING, 64, 8, 8, device=device, dtype=torch.float32)
        cnn = self.cnn.float()
        for ids in torch.arange(1, VOCAB_SIZE_WITH_PADDING, device=device).split(size):
            table[ids] = cnn(self.glyph_bank[ids].float())
        self.train(was_training)
        return table

    @torch.no_grad()
    def switch_to_stage2(self) -> torch.Tensor:
        """Materialize in FP32 and remove every GNU/CNN dependency in-place."""
        table = self.materialize()
        del self.cnn
        del self.glyph_bank
        self._install_feature_maps(table)
        self.mode = "stage2"
        self.cache_metadata = {}
        return table

    def enforce_padding_zero(self) -> None:
        if self.mode == "stage2":
            with torch.no_grad():
                self.feature_maps[0].zero_()

    def glyph_images(self, ids: torch.Tensor) -> torch.Tensor | None:
        if self.mode != "stage1":
            return None
        return self.glyph_bank[ids]
