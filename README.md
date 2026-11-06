# Handwritten Text Generation from Visual Archetypes
[**🤗 `HuggingFace implementation!`**](https://huggingface.co/blowing-up-groundhogs/vatrpp)

This repository contains the reference code and dataset for the paper [Handwritten Text Generation from Visual Archetypes](https://arxiv.org/abs/2303.15269).
If you find it useful, please cite it as:
```
@inproceedings{pippi2023handwritten,
  title={{Handwritten Text Generation from Visual Archetypes}},
  author={Pippi, Vittorio and Cascianelli, Silvia and Cucchiara, Rita},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

![test](https://github.com/aimagelab/VATr/blob/main/files/model_dark.png?raw=true#gh-dark-mode-only)
![test](https://github.com/aimagelab/VATr/blob/main/files/model_light.png?raw=true#gh-light-mode-only)

## Installation

```console
conda create --name vatr python=3.9
conda activate vatr
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
git clone https://github.com/aimagelab/VATr.git && cd VATr
pip install -r requirements.txt
```

From [this folder](https://drive.google.com/drive/folders/1FGJe2uCuK8T9HrFzY_Zc-KMIo0oPJGGY?usp=share_link) download `IAM-32.pickle` and place it into the `files` folder. If `resnet_18_pretrained.pth` is not attached, the runner downloads and caches the official checkpoint automatically.

```console
gdown --folder "https://drive.google.com/drive/u/2/folders/1FGJe2uCuK8T9HrFzY_Zc-KMIo0oPJGGY"
```

## Training

```console
python train.py
```

Useful arguments:
```console
python train.py
        --feat_model_path PATH  # pretrained ResNet-18 style encoder
        --dataset DATASET       # dataset to use. Default IAM
        --resume                # resume training from the last checkpoint with the same name
        --wandb                 # use wandb for logging
```

The style encoder remains VATr's ResNet-18. Its synthetic-pretrained checkpoint is loaded first and then fine-tuned end-to-end with the generator.


## Generate styled Handwtitten Text Images
[**🤗 `HuggingFace implementation!`**](https://huggingface.co/blowing-up-groundhogs/vatrpp)

To generate all samples for FID evaluation you can use the following script:

```console
python generate_fakes.py --checkpoint files/vatr.pth
```

To generate a specific text with a given input style folder containing images of handwritten single words you can use the following script:

```console
python generator.py --style-folder "files/style_samples/00" --checkpoint "files/vatr.pth" --output "files/output_00.png" --text "That's one small step for man, one giant leap for mankind ΑαΒβΓγΔδ"
```


Output for `That's one small step for man, one giant leap for mankind ΑαΒβΓγΔδ`:

![test](https://github.com/aimagelab/VATr/blob/main/files/output_00.png?raw=true)


### Implementation details
This work is partially based on the code released for [Handwriting-Transformers](https://github.com/ankanbhunia/Handwriting-Transformers)

## Two-stage 32x32 GNU feature maps

This fork keeps the pretrained ResNet-18 style encoder, VATr Transformer decoder, image decoder,
GAN, OCR loss, and writer loss. The GNU character-query backend has two additional modes:

1. `stage1`: fixed GNU glyphs are resized 16x16 -> 32x32 with exact nearest-neighbour
   replication, encoded to `[64,8,8]` by a 7.8K-parameter depthwise-separable CNN,
   then spatially attended.
2. `stage2`: the 129 CNN outputs are materialized and optimized directly as a
   `[130,64,8,8]` table. Padding row 0 remains zero. A stage-2 checkpoint contains no
   glyph bitmap or CNN and can be loaded without `unifont.pickle`.

`reference_baseline` retains the official `16x16 -> flatten -> Linear` module and old
checkpoint parameter names. The vocabulary is immutable and recorded in `vocab.json`.

Kaggle (T4/P100) setup does not reinstall torch or torchvision:

```console
pip install -r requirements-kaggle.txt
python train_feature_map.py --config configs/kaggle.yaml --preflight-only
python train_feature_map.py --config configs/kaggle.yaml --mode auto
```

The setup reuses Kaggle's installed scientific stack and only invokes pip for a
missing import. In particular, it does not pin the old `scipy==1.10.1`, which is
incompatible with current Python 3.13 Kaggle images and is not needed by the training runner.

For a fresh Kaggle notebook, attach a raw `iam_dataset` containing `words/`,
`train_gt.txt`, and `val_gt.txt` (or the legacy `IAM-32.pickle`), enable a T4/P100
accelerator, then clone this GitHub repository or import `kaggle_notebook.ipynb`
or run these equivalent cells:

```bash
!git clone --depth 1 https://github.com/lenhanbo/cnn-glyph.git /kaggle/working/cnn-glyph
%cd /kaggle/working/cnn-glyph
!bash scripts/kaggle_setup.sh
!bash scripts/kaggle_run.sh smoke
!bash scripts/kaggle_run.sh auto
```

After a notebook restart, clone the source again and use
`!bash scripts/kaggle_run.sh resume`. Persist `/kaggle/working/vatr-feature-map/latest.pth`
as a Kaggle output/version before the session ends. The setup fails early if CUDA is
disabled or if input discovery finds zero/multiple matching files.

To render diagnostics from a checkpoint stored in the Hugging Face dataset
`lenhanbo/vatrcnn`, attach the iAM dataset to the Kaggle notebook and run:

```bash
!bash scripts/kaggle_setup.sh
!bash scripts/kaggle_run.sh preview-hf
```

This downloads only the checkpoint, writes
`previews/checkpoint_epoch_XXXX.png` and
`previews/tokens_checkpoint_epoch_XXXX.png` below the configured output directory, and
exits without an optimizer step. It prefers `checkpoints/latest.pth` or `latest.pth`,
then falls back to the only `.pth` file in the repository. Override the Hub location
or disambiguate multiple checkpoints with the `HF_CHECKPOINT_REPO` and
`HF_CHECKPOINT_FILE` environment variables.

The same runner supports all requested workflows:

```console
python train_feature_map.py --config configs/kaggle.yaml --mode reference_baseline
python train_feature_map.py --config configs/kaggle.yaml --mode stage1
python train_feature_map.py --config configs/kaggle.yaml --mode stage2 --stage1-checkpoint /path/stage1.pth
python train_feature_map.py --config configs/kaggle.yaml --resume
python train_feature_map.py --config configs/kaggle.yaml --resume /path/latest.pth --preview-only
python train_feature_map.py --config configs/kaggle.yaml --smoke
```

All common settings can be overridden at launch without editing YAML. For example:

```console
bash scripts/kaggle_setup.sh --dataset-path /kaggle/input/my-iam/iam_dataset
bash scripts/kaggle_run.sh smoke --dataset-path /kaggle/input/my-iam/iam_dataset --batch-size 4
bash scripts/kaggle_run.sh auto --dataset-path /kaggle/input/my-iam/iam_dataset \
  --batch-size 8 --epochs 7000 --switch-epoch 3500 --lr 0.00005 \
  --num-workers 2 --output-dir /kaggle/working/vatr-feature-map
```

Other runtime flags include `--seed`, `--num-examples`, `--num-words`, `--save-every`,
`--log-every`, `--preview-every`, `--preview-samples`,
`--glyph-chunk-size`, `--no-amp`, `--raw-group-by`, `--forms-path`,
`--feat-model-path`, `--no-pretrained-style-encoder`, `--no-writer-loss`,
`--writer-loss`, and `--writer-loss-weight`.
Command-line values take priority over `configs/kaggle.yaml`.

For a writer-loss ablation that preserves the same G:D update cadence, use a fresh
output directory:

```bash
bash scripts/kaggle_run.sh stage1 --epochs 100 --no-writer-loss \
  --output-dir /kaggle/working/vatr-no-writer
```

With writer loss disabled, `w_fake_raw`, `alpha_w`, `w_fake`, and `w_real` remain zero;
the corresponding generator/discriminator passes optimize only their GAN objectives.

Training appends batch metrics to `OUTPUT/metrics.csv` (losses, learning rates,
throughput, batch time, stage, and CUDA memory) and prints the same values to the
Kaggle console. Validation panels containing target text, the style reference,
the real word, and the generated word are written to `OUTPUT/previews/epoch_XXXX.png`.
`OCR_fake_raw` and `w_fake_raw` are the unweighted auxiliary losses;
`alpha_ocr` and `alpha_w` are their dynamic gradient-balancing multipliers. The legacy
`OCR_fake` and `w_fake` columns remain the corresponding post-multiplier values. The
OCR and writer generator branches use `autograd.grad` only to measure these multipliers,
then each accumulates generator gradients with exactly one balanced `backward()` call.
Per-character diagnostic tables are written to `OUTPUT/previews/tokens_epoch_XXXX.png`.
They show the GNU glyph, query feature, and both feature/rendered output from the last
decoder layers; `c=` is mean off-diagonal token cosine similarity, where values near 1
indicate token collapse. Configure them with `token_preview_layers`,
`token_preview_samples`, and `token_preview_max_chars`.
CSV history is append-only across resume. Use `--log-every 10` for quieter logs or
`--preview-every 0` to disable previews.

If paths are null in the YAML, the runner requires exactly one raw IAM directory or one
`IAM-32.pickle` below `/kaggle/input`. It uses an attached `resnet_18_pretrained.pth` when
available; otherwise it downloads the official file once into `OUTPUT/pretrained/`.
The raw files do not contain writer IDs, so the default groups samples by IAM form.
For exact writer grouping, attach official `forms.txt`, set `forms_path`, and change
`raw_group_by` to `writer`. Checkpoints and switch artifacts go to
`/kaggle/working/vatr-feature-map`. The auto run saves `pre_switch.pth`,
`feature_maps.pt`, and `post_switch_init.pth`, and aborts if fixed query tokens differ by
more than `1e-6` across materialization.

Run the CPU contract tests with `python -m pytest -q`. The final batch-8 AMP smoke test
must be run on a Kaggle GPU with the command above because it exercises the full VATr
GAN and the Kaggle CUDA build.
