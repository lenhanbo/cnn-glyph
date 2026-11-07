"""Kaggle/local runner for reference, stage-1, stage-2 and automatic VATr training."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from data.dataset import TextDataset, TextDatasetval
from data.raw_iam_dataset import RawIAMDataset
from models.model import VATr, load_pretrained_ocr, load_pretrained_style_encoder
from models.stage_manager import capture_rng_state, checkpoint_payload, load_checkpoint_file, restore_rng_state, switch_generator_to_stage2
from models.vocabulary import IAM_ALPHABET, SPECIAL_ALPHABET
from util.pretrained import STYLE_ENCODER_FILENAME, ensure_pretrained_style_encoder


LOSS_NAMES = (
    "G", "D", "Dfake", "Dreal",
    "OCR_fake_raw", "alpha_ocr", "OCR_fake", "OCR_real",
    "w_fake_raw", "alpha_w", "w_fake", "w_real", "cycle",
)


def optimization_plan(no_writer_loss: bool, batch_index: int, cfg: dict) -> tuple[str, ...]:
    """Return the optimizer phases for one batch.

    The writer-free ablation is a clean 1D:1G schedule. The writer-enabled
    path preserves the original VATr cadence.
    """
    if no_writer_loss:
        return ("d_ocr", "g_ocr")
    phases = []
    if batch_index % int(cfg.get("num_critic_gocr_train", 2)) == 0:
        phases.append("g_ocr")
    if batch_index % int(cfg.get("num_critic_docr_train", 1)) == 0:
        phases.append("d_ocr")
    if batch_index % int(cfg.get("num_critic_gwl_train", 2)) == 0:
        phases.append("g_writer")
    if batch_index % int(cfg.get("num_critic_dwl_train", 1)) == 0:
        phases.append("d_writer")
    return tuple(phases)


def scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu())
    return float(value)


class MetricsLogger:
    """Append one durable CSV row per selected batch; safe across resume."""
    fields = (
        "timestamp", "epoch", "batch", "batches", "global_step", "query_mode", "batch_seconds",
        "samples_per_second", "lr_g", "lr_d", "lr_ocr", "lr_writer", "gpu_memory_mb",
        "gpu_memory_peak_mb", *LOSS_NAMES,
    )

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict) -> None:
        needs_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields, extrasaction="ignore")
            if needs_header:
                writer.writeheader()
            writer.writerow({name: row.get(name, "") for name in self.fields})


def _gray_image(tensor: torch.Tensor, crop_ink: bool = False) -> Image.Image:
    array = ((tensor.detach().float().cpu().squeeze().clamp(-1, 1) + 1) * 127.5).numpy().astype(np.uint8)
    if array.ndim == 1:
        array = array[None, :]
    if crop_ink and array.shape[1] > 1:
        ink = np.where(array.min(axis=0) < 250)[0]
        if ink.size:
            array = array[:, :int(ink[-1]) + 1]
    return Image.fromarray(array, mode="L")


def _feature_image(vector: torch.Tensor, size: int = 32) -> Image.Image:
    """Turn one hidden vector into a normalized diagnostic heatmap."""
    vector = vector.detach().float().cpu()
    width = 32 if vector.numel() % 32 == 0 else vector.numel()
    array = vector.reshape(-1, width).numpy()
    low, high = float(array.min()), float(array.max())
    if high > low:
        array = (array - low) * (255.0 / (high - low))
    else:
        array = np.zeros_like(array)
    return Image.fromarray(array.astype(np.uint8), mode="L").resize(
        (size, size), Image.Resampling.NEAREST
    )


def _mean_token_cosine(tokens: torch.Tensor) -> float:
    tokens = tokens.detach().float()
    if tokens.shape[0] < 2:
        return 1.0
    tokens = torch.nn.functional.normalize(tokens, dim=-1)
    similarities = tokens @ tokens.transpose(0, 1)
    mask = ~torch.eye(tokens.shape[0], dtype=torch.bool, device=tokens.device)
    return float(similarities[mask].mean().cpu())


def save_token_preview(model: VATr, data: dict, path: str | Path, device: str,
                       max_samples: int = 2, last_layers: int = 3,
                       max_chars: int = 16) -> None:
    """Save a per-character table for GNU/query/late-decoder collapse diagnosis."""
    rng_state = capture_rng_state()
    was_training = model.training
    model.eval()
    count = min(max_samples, len(data["label"]))
    labels = data["label"][:count]
    styles = data["simg"][:count].to(device)
    encoded, lengths, _ = model.netconverter.encode(labels)
    encoded = encoded[:count].to(device)
    try:
        with torch.no_grad():
            diagnostic = model.netG.inspect_character_layers(styles, encoded, last_layers)

        query_tokens = diagnostic["query_tokens"]
        decoder_tokens = diagnostic["decoder_tokens"]
        layer_images = diagnostic["layer_images"]
        layer_numbers = diagnostic["layer_numbers"]
        glyphs = diagnostic["glyphs"]
        resolution = int(model.args.resolution)
        cell = 32
        label_width = 118
        row_height = 36
        sample_canvases = []

        for sample_index in range(count):
            label = labels[sample_index].decode("utf-8") if isinstance(labels[sample_index], bytes) else str(labels[sample_index])
            length = min(int(lengths[sample_index]), len(label), max_chars)
            if length == 0:
                continue
            chars = list(label[:length])
            rows = []

            if glyphs is not None:
                glyph_cells = []
                for char_index in range(length):
                    glyph = glyphs[sample_index, char_index].detach().float().cpu().squeeze().numpy()
                    if glyph.max() > 1:
                        glyph = glyph / glyph.max()
                    glyph_cells.append(Image.fromarray(((1.0 - glyph) * 255).astype(np.uint8), mode="L"))
                rows.append(("GNU glyph", glyph_cells))

            query_cosine = _mean_token_cosine(query_tokens[sample_index, :length])
            rows.append((f"query feat c={query_cosine:.2f}", [
                _feature_image(query_tokens[sample_index, char_index])
                for char_index in range(length)
            ]))

            for layer_offset, layer_number in enumerate(layer_numbers):
                tokens = decoder_tokens[layer_offset, sample_index, :length]
                cosine = _mean_token_cosine(tokens)
                rows.append((f"L{layer_number} feat c={cosine:.2f}", [
                    _feature_image(tokens[char_index]) for char_index in range(length)
                ]))
                rendered_cells = []
                rendered = layer_images[layer_offset, sample_index, 0]
                for char_index in range(length):
                    patch = rendered[:, char_index * resolution:(char_index + 1) * resolution]
                    rendered_cells.append(_gray_image(patch).resize((cell, cell), Image.Resampling.BILINEAR))
                rows.append((f"L{layer_number} render", rendered_cells))

            header_height = 42
            canvas = Image.new("L", (label_width + length * cell, header_height + len(rows) * row_height), 255)
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 3), f"target: {label[:length]}", fill=0)
            for char_index, char in enumerate(chars):
                x = label_width + char_index * cell
                draw.rectangle((x, 20, x + cell - 1, header_height - 1), outline=180)
                draw.text((x + 12, 23), char, fill=0)
            for row_index, (row_name, images) in enumerate(rows):
                y = header_height + row_index * row_height
                draw.text((4, y + 11), row_name, fill=0)
                for char_index, image in enumerate(images):
                    x = label_width + char_index * cell
                    canvas.paste(image, (x, y + 2))
                    draw.rectangle((x, y + 2, x + cell - 1, y + cell + 1), outline=180)
            sample_canvases.append(canvas)

        if not sample_canvases:
            return
        gap = 8
        output = Image.new(
            "L",
            (max(canvas.width for canvas in sample_canvases),
             sum(canvas.height for canvas in sample_canvases) + gap * (len(sample_canvases) - 1)),
            255,
        )
        y = 0
        for canvas in sample_canvases:
            output.paste(canvas, (0, y))
            y += canvas.height + gap
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        output.save(path)
    finally:
        model.train(was_training)
        restore_rng_state(rng_state)


def save_preview(model: VATr, data: dict, path: str | Path, device: str, max_samples: int = 4) -> None:
    """Save style/real/fake panels without changing model mode or RNG state."""
    rng_state = capture_rng_state()
    was_training = model.training
    model.eval()
    count = min(max_samples, len(data["label"]))
    labels = data["label"][:count]
    styles = data["simg"][:count].to(device)
    encoded, lengths, positions = model.netconverter.encode(labels)
    encoded = encoded.to(device).unsqueeze(1)
    try:
        with torch.no_grad():
            generated = model.netG.Eval(styles, encoded, positions)[0].detach().float().cpu()

        panels = []
        for index in range(count):
            style_parts = []
            for style_index in range(min(3, styles.shape[1])):
                width = int(data["swids"][index, style_index])
                style_parts.append(_gray_image(styles[index, style_index, :, :width]))
            style_width = sum(image.width for image in style_parts) + 4 * max(0, len(style_parts) - 1)
            style_strip = Image.new("L", (max(1, style_width), 32), 255)
            x = 0
            for image in style_parts:
                style_strip.paste(image, (x, 0)); x += image.width + 4
            real = _gray_image(data["img"][index], crop_ink=True)
            fake_width = int(lengths[index]) * int(model.args.resolution)
            fake = _gray_image(generated[index, 0, :, :fake_width])
            label = labels[index].decode("utf-8") if isinstance(labels[index], bytes) else str(labels[index])
            panels.append((label, style_strip, real, fake))

        canvas_width = max(max(image.width for image in panel[1:]) for panel in panels) + 90
        panel_height = 18 + 3 * 36 + 8
        canvas = Image.new("L", (canvas_width, panel_height * len(panels)), 255)
        draw = ImageDraw.Draw(canvas)
        for index, (label, style, real, fake) in enumerate(panels):
            top = index * panel_height
            draw.text((4, top + 2), f"target: {label}", fill=0)
            for row, (name, image) in enumerate((("style", style), ("real", real), ("fake", fake))):
                y = top + 18 + row * 36
                draw.text((4, y + 9), name, fill=0)
                canvas.paste(image, (70, y))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
    finally:
        model.train(was_training)
        restore_rng_state(rng_state)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preflight(config: dict) -> None:
    if config.get("device", "cuda").startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        props = torch.cuda.get_device_properties(0)
        print(f"CUDA preflight: {props.name}, {props.total_memory / 2**30:.1f} GiB, torch={torch.__version__}")


def resolve_paths(config: dict) -> dict:
    config = dict(config)
    kaggle_root = config.get("kaggle_input", "/kaggle/input")
    if not config.get("dataset_path"):
        pickles = list(Path(kaggle_root).rglob("IAM-32.pickle"))
        raw_roots = [path.parent for path in Path(kaggle_root).rglob("train_gt.txt")
                     if (path.parent / "val_gt.txt").is_file() and (path.parent / "words").is_dir()]
        candidates = [*pickles, *raw_roots]
        if len(candidates) != 1:
            raise FileNotFoundError(f"Expected one IAM-32.pickle or raw iam_dataset below {kaggle_root}; found {candidates}")
        config["dataset_path"] = str(candidates[0])
    if config.get("use_pretrained_style_encoder", True) and not config.get("feat_model_path"):
        matches = list(Path(kaggle_root).rglob(STYLE_ENCODER_FILENAME))
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Expected at most one {STYLE_ENCODER_FILENAME!r} below {kaggle_root}; "
                f"found {len(matches)}: {matches}. Set feat_model_path explicitly."
            )
        if matches:
            config["feat_model_path"] = str(matches[0])
        elif config.get("auto_download_style_encoder", True):
            cache = Path(config.get("output_dir", "/kaggle/working/vatr-feature-map")) / "pretrained" / STYLE_ENCODER_FILENAME
            print(f"Pretrained style encoder not attached; downloading to {cache}")
            config["feat_model_path"] = ensure_pretrained_style_encoder(cache)
        else:
            raise FileNotFoundError(
                f"{STYLE_ENCODER_FILENAME} was not found below {kaggle_root} and automatic download is disabled"
            )
    if not config.get("use_pretrained_style_encoder", True):
        config["feat_model_path"] = None
    config.setdefault("unifont_path", "files/unifont.pickle")
    return config


class Args:
    """Namespace with VATr's legacy options plus the two-stage configuration."""
    def __init__(self, cfg: dict, num_writers: int, mode: str):
        defaults = dict(
            dataset="IAM", device="cuda", english_words_path="files/english_words.txt", img_height=32, resolution=16,
            batch_size=8, num_examples=1, tn_hidden_dim=512, tn_dropout=0.1, tn_nheads=8,
            tn_dim_feedforward=512, tn_enc_layers=3, tn_dec_layers=3, g_lr=5e-5, d_lr=5e-5,
            w_lr=5e-5, ocr_lr=5e-5, no_writer_loss=False, writer_loss_weight=1.0,
            no_ocr_loss=False, freeze_ocr=False, is_seq=True, num_words=3, is_cycle=False, is_kld=False,
            add_noise=False, all_chars=False,
            query_input="unifont", query_linear=True, glyph_chunk_size=128,
        )
        defaults.update(cfg)
        self.__dict__.update(defaults)
        self.alphabet = IAM_ALPHABET
        self.special_alphabet = SPECIAL_ALPHABET
        self.vocab_size = len(IAM_ALPHABET)
        self.num_writers = num_writers
        self.feature_mode = mode


def save(path: Path, model: VATr, epoch: int, global_step: int, scaler, scheduler, config: dict) -> None:
    payload = checkpoint_payload(model, epoch, global_step, scaler, scheduler, config=config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_training_state(path: str | Path, model: VATr, scaler, scheduler, device: str) -> tuple[int, int]:
    checkpoint = load_checkpoint_file(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    for optimizer, state in zip(model.optimizers, checkpoint.get("optimizers", [])):
        optimizer.load_state_dict(state)
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("global_step", 0))


def download_hf_checkpoint(repo_id: str, filename: str | None, repo_type: str,
                           local_dir: str | Path) -> Path:
    """Download one checkpoint from the Hub without pulling the whole repository."""
    try:
        import huggingface_hub
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face checkpoint download requires huggingface_hub. "
            "Run scripts/kaggle_setup.sh first."
        ) from error
    if not filename:
        files = huggingface_hub.HfApi().list_repo_files(repo_id=repo_id, repo_type=repo_type)
        preferred = ("checkpoints/latest.pth", "latest.pth")
        filename = next((candidate for candidate in preferred if candidate in files), None)
        if filename is None:
            candidates = [candidate for candidate in files if candidate.lower().endswith(".pth")]
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Could not auto-select one checkpoint from {repo_id}: found {candidates}. "
                    "Pass --hf-checkpoint-file explicitly."
                )
            filename = candidates[0]
        print(f"hf_checkpoint_file={filename}", flush=True)
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    path = huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir=str(local_dir),
    )
    checkpoint = Path(path)
    print(f"hf_checkpoint={checkpoint}", flush=True)
    return checkpoint


def make_loaders(cfg: dict):
    common = dict(collator_resolution=cfg.get("resolution", 16), num_examples=cfg.get("num_examples", 1))
    dataset_path = Path(cfg["dataset_path"])
    if dataset_path.is_dir():
        raw_common = dict(group_by=cfg.get("raw_group_by", "form"), forms_path=cfg.get("forms_path"), **common)
        train_set = RawIAMDataset(dataset_path, split="train", min_virtual_size=339, **raw_common)
        val_set = RawIAMDataset(dataset_path, split="val", min_virtual_size=161, **raw_common)
        print(f"Raw IAM: {train_set.num_writers} train style classes, {val_set.num_writers} val classes; "
              f"filtered labels={train_set.skipped_labels}/{val_set.skipped_labels}")
    else:
        train_set = TextDataset(dataset_path, min_virtual_size=339, **common)
        val_set = TextDatasetval(dataset_path, min_virtual_size=161, **common)
    loader_args = dict(batch_size=cfg.get("batch_size", 8), num_workers=cfg.get("num_workers", 2),
                       pin_memory=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, collate_fn=train_set.collate_fn, **loader_args)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, collate_fn=val_set.collate_fn, **loader_args)
    return train_set, train_loader, val_loader


def switch_stage(model: VATr, output: Path, epoch: int, global_step: int, scaler, scheduler, cfg: dict) -> None:
    save(output / "pre_switch.pth", model, epoch, global_step, scaler, scheduler, cfg)
    query = model.netG.query_embed
    was_training = model.netG.training
    model.netG.eval()
    fixed_ids = torch.arange(130, device=cfg["device"], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        before = query(fixed_ids).float()
    table = query.materialize().cpu()
    raw_table = query.materialize_raw().cpu()
    torch.save({
        "feature_maps": table,
        "raw_embeddings": raw_table,
        "shape": list(table.shape),
    }, output / "feature_maps.pt")
    model.optimizer_G = switch_generator_to_stage2(model.netG, model.optimizer_G)
    model.optimizers[0] = model.optimizer_G
    scheduler.optimizer = model.optimizer_G
    with torch.no_grad():
        after = model.netG.query_embed(fixed_ids).float()
    error = (before - after).abs().max().item()
    if error > 1e-6:
        raise RuntimeError(f"stage switch changed fixed query tokens (max error {error:.3g})")
    model.netG.train(was_training)
    save(output / "post_switch_init.pth", model, epoch, global_step, scaler, scheduler, cfg)
    print(f"Stage 2 initialized; fixed-token max error={error:.3g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kaggle.yaml")
    parser.add_argument("--mode", choices=["reference_baseline", "auto", "stage1", "stage2"])
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--stage1-checkpoint")
    parser.add_argument("--dataset-path", help="Raw iam_dataset directory or IAM-32.pickle")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--switch-epoch", type=int)
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--num-words", type=int, help="Number of generated words concatenated per training sample")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float, help="Set G, D, OCR and writer learning rates together")
    writer_group = parser.add_mutually_exclusive_group()
    writer_group.add_argument("--no-writer-loss", dest="no_writer_loss", action="store_true",
                              help="Disable writer CE on real and fake while preserving GAN update cadence")
    writer_group.add_argument("--writer-loss", dest="no_writer_loss", action="store_false",
                              help="Enable writer CE (overrides the YAML setting)")
    parser.set_defaults(no_writer_loss=None)
    parser.add_argument("--writer-loss-weight", type=float,
                        help="Scale the dynamically balanced writer objective; default 1.0")
    parser.add_argument("--ocr-checkpoint",
                        help="Compatible standalone OCR or full VATr checkpoint")
    ocr_group = parser.add_mutually_exclusive_group()
    ocr_group.add_argument("--freeze-ocr", dest="freeze_ocr", action="store_true", default=None,
                           help="Keep pretrained OCR fixed (recommended)")
    ocr_group.add_argument("--finetune-ocr", dest="freeze_ocr", action="store_false",
                           help="Continue updating OCR on real IAM words")
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--log-every", type=int, help="Write CSV and console metrics every N batches")
    parser.add_argument("--preview-every", type=int, help="Save validation preview every N epochs; 0 disables")
    parser.add_argument("--preview-samples", type=int)
    parser.add_argument("--token-preview-layers", type=int)
    parser.add_argument("--token-preview-samples", type=int)
    parser.add_argument("--token-preview-max-chars", type=int)
    parser.add_argument("--glyph-chunk-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--raw-group-by", choices=["form", "writer"])
    parser.add_argument("--forms-path", help="Official IAM forms.txt; required with raw-group-by=writer")
    parser.add_argument("--feat-model-path")
    style_group = parser.add_mutually_exclusive_group()
    style_group.add_argument("--pretrained-style-encoder", dest="pretrained_style_encoder",
                             action="store_true", default=None)
    style_group.add_argument("--no-pretrained-style-encoder", dest="pretrained_style_encoder",
                             action="store_false")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preview-only", action="store_true",
                        help="Load a checkpoint, write validation/token previews, and exit without training")
    parser.add_argument("--hf-checkpoint-repo",
                        help="Hugging Face repository containing the resume checkpoint")
    parser.add_argument("--hf-checkpoint-file",
                        help="Checkpoint path inside the Hugging Face repository")
    parser.add_argument("--hf-repo-type", choices=["model", "dataset"], default="dataset")
    cli = parser.parse_args()
    cfg = yaml.safe_load(Path(cli.config).read_text(encoding="utf-8"))
    if cli.mode:
        cfg["mode"] = cli.mode
    override_names = (
        "dataset_path", "output_dir", "batch_size", "epochs", "switch_epoch", "num_examples", "num_words",
        "num_workers", "seed", "save_every", "log_every", "preview_every", "preview_samples",
        "token_preview_layers", "token_preview_samples", "token_preview_max_chars",
        "glyph_chunk_size", "device", "raw_group_by", "writer_loss_weight",
        "forms_path", "feat_model_path", "ocr_checkpoint",
    )
    for name in override_names:
        value = getattr(cli, name)
        if value is not None:
            cfg[name] = value
    if cli.lr is not None:
        cfg.update(g_lr=cli.lr, d_lr=cli.lr, w_lr=cli.lr, ocr_lr=cli.lr)
    if cli.no_writer_loss is not None:
        cfg["no_writer_loss"] = cli.no_writer_loss
    if cli.freeze_ocr is not None:
        cfg["freeze_ocr"] = cli.freeze_ocr
    elif cli.ocr_checkpoint:
        # A pretrained OCR is most useful as a stationary teacher. Users can
        # explicitly opt back into online fine-tuning with --finetune-ocr.
        cfg["freeze_ocr"] = True
    if cli.pretrained_style_encoder is not None:
        cfg["use_pretrained_style_encoder"] = cli.pretrained_style_encoder
    if cli.no_amp:
        cfg["amp"] = False
    if cli.smoke:
        cfg.update(epochs=1, max_batches_per_epoch=1, save_every=1, log_every=1, preview_every=1)
    if cli.hf_checkpoint_repo:
        checkpoint_dir = Path(cfg.get("output_dir", "/kaggle/working/vatr-feature-map")) / "hf"
        cli.resume = str(download_hf_checkpoint(
            cli.hf_checkpoint_repo, cli.hf_checkpoint_file, cli.hf_repo_type, checkpoint_dir
        ))
    if cli.preview_only and not cli.resume:
        parser.error("--preview-only requires --resume or --hf-checkpoint-repo/--hf-checkpoint-file")
    # A resume checkpoint already contains the complete style encoder. Avoid an unrelated
    # pretrained-weight download when this run exists only to render diagnostics.
    if cli.preview_only:
        cfg["use_pretrained_style_encoder"] = False
    cfg = resolve_paths(cfg)
    cfg["device"] = cfg.get("device", "cuda")
    preflight(cfg)
    if cli.preflight_only:
        print(json.dumps(cfg, indent=2))
        return
    seed_everything(int(cfg.get("seed", 742)))
    output = Path(cfg.get("output_dir", "/kaggle/working/vatr-feature-map"))
    output.mkdir(parents=True, exist_ok=True)
    train_set, train_loader, val_loader = make_loaders(cfg)
    mode = cfg.get("mode", "auto")
    resume_path = output / "latest.pth" if cli.resume == "auto" else (Path(cli.resume) if cli.resume else None)
    resume_header = load_checkpoint_file(resume_path, map_location="cpu") if resume_path and resume_path.exists() else None
    initial_mode = "stage2" if resume_header and resume_header.get("query_mode") == "stage2" else mode
    if mode == "stage2" and resume_header is None:
        # A fresh stage-2 run first loads/materializes its stage-1 checkpoint.
        initial_mode = "stage1"
    if initial_mode == "auto":
        initial_mode = "stage1"
    args = Args(cfg, train_set.num_writers, initial_mode)
    print(
        f"writer_loss={'disabled' if args.no_writer_loss else 'enabled'} "
        f"weight={args.writer_loss_weight} "
        f"ocr={'frozen' if args.freeze_ocr else 'trainable'}",
        flush=True,
    )
    if args.freeze_ocr and not cfg.get("ocr_checkpoint") and resume_header is None and mode != "stage2":
        raise ValueError("freeze_ocr requires --ocr-checkpoint for a fresh run")
    model = VATr(args)
    amp_enabled = bool(cfg.get("amp", True) and cfg["device"].startswith("cuda"))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # compatibility with the original PyTorch 1.13 environment
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    scheduler = torch.optim.lr_scheduler.LambdaLR(model.optimizer_G, lambda _: 1.0)
    metrics = MetricsLogger(output / "metrics.csv")
    start_epoch = global_step = 0
    if resume_header is not None:
        start_epoch, global_step = load_training_state(resume_path, model, scaler, scheduler, cfg["device"])
    elif mode == "stage2":
        source = cli.stage1_checkpoint or cfg.get("stage1_checkpoint")
        if not source:
            raise ValueError("stage2 requires --stage1-checkpoint")
        start_epoch, global_step = load_training_state(source, model, scaler, scheduler, cfg["device"])
        switch_stage(model, output, start_epoch, global_step, scaler, scheduler, cfg)
    elif cfg.get("feat_model_path"):
        checkpoint = torch.load(cfg["feat_model_path"], map_location=cfg["device"])
        load_pretrained_style_encoder(model.netG.Feat_Encoder, checkpoint)
        print(f"Loaded pretrained ResNet-18 style encoder: {cfg['feat_model_path']}")
    if cfg.get("ocr_checkpoint") and resume_header is None and mode != "stage2":
        ocr_checkpoint = load_checkpoint_file(cfg["ocr_checkpoint"], map_location=cfg["device"])
        load_pretrained_ocr(model.netOCR, ocr_checkpoint)
        if args.freeze_ocr:
            model.set_requires_grad(model.netOCR, False)
            model.netOCR.eval()
        print(
            f"Loaded pretrained OCR: {cfg['ocr_checkpoint']} "
            f"({'frozen' if args.freeze_ocr else 'fine-tuned'})",
            flush=True,
        )
    if cli.preview_only:
        fixed_val = next(iter(val_loader))
        preview_dir = output / "previews"
        preview_path = preview_dir / f"checkpoint_epoch_{start_epoch:04d}.png"
        token_preview_path = preview_dir / f"tokens_checkpoint_epoch_{start_epoch:04d}.png"
        save_preview(model, fixed_val, preview_path, cfg["device"], int(cfg.get("preview_samples", 4)))
        save_token_preview(
            model, fixed_val, token_preview_path, cfg["device"],
            max_samples=int(cfg.get("token_preview_samples", 2)),
            last_layers=int(cfg.get("token_preview_layers", 3)),
            max_chars=int(cfg.get("token_preview_max_chars", 16)),
        )
        print(f"preview={preview_path}", flush=True)
        print(f"token_preview={token_preview_path}", flush=True)
        print(f"preview_complete checkpoint_epoch={start_epoch} step={global_step}", flush=True)
        return
    amp_context = (lambda: torch.autocast("cuda", dtype=torch.float16)) if scaler.is_enabled() else contextlib.nullcontext
    epochs = int(cfg.get("epochs", 7000))
    switch_epoch = int(cfg.get("switch_epoch", 3500))
    log_every = max(1, int(cfg.get("log_every", 1)))
    preview_every = max(0, int(cfg.get("preview_every", 10)))
    fixed_val = None
    if preview_every:
        rng_state = capture_rng_state()
        fixed_val = next(iter(val_loader))
        restore_rng_state(rng_state)
    for epoch in range(start_epoch, epochs):
        if mode == "auto" and epoch == switch_epoch and model.netG.query_embed.mode == "stage1":
            switch_stage(model, output, epoch, global_step, scaler, scheduler, cfg)
        for batch_index, data in enumerate(train_loader):
            batch_started = time.perf_counter()
            with amp_context():
                for phase in optimization_plan(model.args.no_writer_loss, batch_index, cfg):
                    model._set_input(data)
                    if phase == "d_ocr":
                        model.optimize_D_OCR(); model.optimize_D_OCR_step()
                    elif phase == "g_ocr":
                        model.optimize_G_only(); model.optimize_G_step()
                    elif phase == "g_writer":
                        model.optimize_G_WL(); model.optimize_G_step()
                    elif phase == "d_writer":
                        model.optimize_D_WL(); model.optimize_D_WL_step()
            if model.netG.query_embed.mode == "stage2":
                model.netG.query_embed.enforce_padding_zero()
            global_step += 1
            if (batch_index + 1) % log_every == 0 or batch_index == 0:
                elapsed = time.perf_counter() - batch_started
                losses = {name: scalar(value) for name, value in model.get_current_losses().items()
                          if name in LOSS_NAMES}
                gpu_memory = torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
                gpu_peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
                row = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "epoch": epoch + 1,
                    "batch": batch_index + 1, "batches": len(train_loader), "global_step": global_step,
                    "query_mode": model.netG.query_embed.mode, "batch_seconds": f"{elapsed:.6f}",
                    "samples_per_second": f"{len(data['label']) / max(elapsed, 1e-9):.3f}",
                    "lr_g": model.optimizer_G.param_groups[0]["lr"],
                    "lr_d": model.optimizer_D.param_groups[0]["lr"],
                    "lr_ocr": model.optimizer_OCR.param_groups[0]["lr"],
                    "lr_writer": model.optimizer_wl.param_groups[0]["lr"],
                    "gpu_memory_mb": f"{gpu_memory:.1f}", "gpu_memory_peak_mb": f"{gpu_peak:.1f}",
                    **losses,
                }
                metrics.write(row)
                loss_text = " ".join(f"{name}={value:.4g}" for name, value in losses.items())
                print(f"epoch={epoch + 1}/{epochs} batch={batch_index + 1}/{len(train_loader)} "
                      f"step={global_step} query={model.netG.query_embed.mode} {loss_text} "
                      f"sec={elapsed:.2f} gpu={gpu_memory:.0f}MB", flush=True)
            if cfg.get("max_batches_per_epoch") and batch_index + 1 >= int(cfg["max_batches_per_epoch"]):
                break
        scheduler.step()
        next_epoch = epoch + 1
        if fixed_val is not None and next_epoch % preview_every == 0:
            preview_path = output / "previews" / f"epoch_{next_epoch:04d}.png"
            save_preview(model, fixed_val, preview_path, cfg["device"], int(cfg.get("preview_samples", 4)))
            print(f"preview={preview_path}", flush=True)
            token_preview_path = output / "previews" / f"tokens_epoch_{next_epoch:04d}.png"
            save_token_preview(
                model, fixed_val, token_preview_path, cfg["device"],
                max_samples=int(cfg.get("token_preview_samples", 2)),
                last_layers=int(cfg.get("token_preview_layers", 3)),
                max_chars=int(cfg.get("token_preview_max_chars", 16)),
            )
            print(f"token_preview={token_preview_path}", flush=True)
        if next_epoch % int(cfg.get("save_every", 5)) == 0 or next_epoch == epochs:
            save(output / "latest.pth", model, next_epoch, global_step, scaler, scheduler, cfg)
        print(f"epoch_complete={next_epoch}/{epochs} step={global_step} query={model.netG.query_embed.mode}", flush=True)


if __name__ == "__main__":
    main()
