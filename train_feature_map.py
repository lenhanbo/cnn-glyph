"""Kaggle/local runner for reference, stage-1, stage-2 and automatic VATr training."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from data.dataset import TextDataset, TextDatasetval
from models.model import VATr
from models.stage_manager import checkpoint_payload, load_checkpoint_file, restore_rng_state, switch_generator_to_stage2
from models.vocabulary import IAM_ALPHABET, SPECIAL_ALPHABET


def unique_discovery(root: str | Path, filename: str) -> str:
    matches = list(Path(root).rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {filename!r} below {root}, found {len(matches)}: {matches}")
    return str(matches[0])


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
        config["dataset_path"] = unique_discovery(kaggle_root, "IAM-32.pickle")
    if not config.get("feat_model_path"):
        config["feat_model_path"] = unique_discovery(kaggle_root, "resnet_18_pretrained.pth")
    config.setdefault("unifont_path", "files/unifont.pickle")
    return config


class Args:
    """Namespace with VATr's legacy options plus the two-stage configuration."""
    def __init__(self, cfg: dict, num_writers: int, mode: str):
        defaults = dict(
            dataset="IAM", english_words_path="files/english_words.txt", img_height=32, resolution=16,
            batch_size=8, num_examples=15, tn_hidden_dim=512, tn_dropout=0.1, tn_nheads=8,
            tn_dim_feedforward=512, tn_enc_layers=3, tn_dec_layers=3, g_lr=5e-5, d_lr=5e-5,
            w_lr=5e-5, ocr_lr=5e-5, no_writer_loss=False, writer_loss_weight=1.0,
            no_ocr_loss=False, is_seq=True, is_kld=False, add_noise=False, all_chars=False,
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


def make_loaders(cfg: dict):
    common = dict(collator_resolution=cfg.get("resolution", 16), num_examples=cfg.get("num_examples", 15))
    train_set = TextDataset(cfg["dataset_path"], min_virtual_size=339, **common)
    val_set = TextDatasetval(cfg["dataset_path"], min_virtual_size=161, **common)
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
    torch.save({"feature_maps": table, "shape": list(table.shape)}, output / "feature_maps.pt")
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    cli = parser.parse_args()
    cfg = yaml.safe_load(Path(cli.config).read_text(encoding="utf-8"))
    cfg = resolve_paths(cfg)
    if cli.mode:
        cfg["mode"] = cli.mode
    if cli.smoke:
        cfg.update(epochs=1, max_batches_per_epoch=1, save_every=1)
    cfg["device"] = cfg.get("device", "cuda")
    preflight(cfg)
    if cli.preflight_only:
        print(json.dumps(cfg, indent=2))
        return
    seed_everything(int(cfg.get("seed", 742)))
    output = Path(cfg.get("output_dir", "/kaggle/working/vatr-feature-map"))
    output.mkdir(parents=True, exist_ok=True)
    train_set, train_loader, _ = make_loaders(cfg)
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
    model = VATr(args)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("amp", True) and cfg["device"].startswith("cuda")))
    scheduler = torch.optim.lr_scheduler.LambdaLR(model.optimizer_G, lambda _: 1.0)
    start_epoch = global_step = 0
    if resume_header is not None:
        start_epoch, global_step = load_training_state(resume_path, model, scaler, scheduler, cfg["device"])
    elif mode == "stage2":
        source = cli.stage1_checkpoint or cfg.get("stage1_checkpoint")
        if not source:
            raise ValueError("stage2 requires --stage1-checkpoint")
        start_epoch, global_step = load_training_state(source, model, scaler, scheduler, cfg["device"])
        switch_stage(model, output, start_epoch, global_step, scaler, scheduler, cfg)
    elif cfg.get("feat_model_path") and start_epoch == 0:
        checkpoint = torch.load(cfg["feat_model_path"], map_location=cfg["device"])
        state = checkpoint.get("model", checkpoint)
        state["conv1.weight"] = state["conv1.weight"].mean(1, keepdim=True)
        model.netG.Feat_Encoder.load_state_dict(state, strict=False)

    amp_context = (lambda: torch.autocast("cuda", dtype=torch.float16)) if scaler.is_enabled() else contextlib.nullcontext
    epochs = int(cfg.get("epochs", 7000))
    switch_epoch = int(cfg.get("switch_epoch", 3500))
    for epoch in range(start_epoch, epochs):
        if mode == "auto" and epoch == switch_epoch and model.netG.query_embed.mode == "stage1":
            switch_stage(model, output, epoch, global_step, scaler, scheduler, cfg)
        for batch_index, data in enumerate(train_loader):
            with amp_context():
                if batch_index % int(cfg.get("num_critic_gocr_train", 2)) == 0:
                    model._set_input(data); model.optimize_G_only(); model.optimize_G_step()
                if batch_index % int(cfg.get("num_critic_docr_train", 1)) == 0:
                    model._set_input(data); model.optimize_D_OCR(); model.optimize_D_OCR_step()
                if batch_index % int(cfg.get("num_critic_gwl_train", 2)) == 0:
                    model._set_input(data); model.optimize_G_WL(); model.optimize_G_step()
                if batch_index % int(cfg.get("num_critic_dwl_train", 1)) == 0:
                    model._set_input(data); model.optimize_D_WL(); model.optimize_D_WL_step()
            if model.netG.query_embed.mode == "stage2":
                model.netG.query_embed.enforce_padding_zero()
            global_step += 1
            if cfg.get("max_batches_per_epoch") and batch_index + 1 >= int(cfg["max_batches_per_epoch"]):
                break
        scheduler.step()
        next_epoch = epoch + 1
        if next_epoch % int(cfg.get("save_every", 5)) == 0 or next_epoch == epochs:
            save(output / "latest.pth", model, next_epoch, global_step, scaler, scheduler, cfg)
        print(f"epoch={next_epoch}/{epochs} step={global_step} query={model.netG.query_embed.mode}")


if __name__ == "__main__":
    main()
