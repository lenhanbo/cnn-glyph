"""Download helpers for official VATr pretrained assets."""

from pathlib import Path


STYLE_ENCODER_FILENAME = "resnet_18_pretrained.pth"
STYLE_ENCODER_GDRIVE_ID = "1cmwkQel947dewifjGm9pA9HCCZD-cy0P"


def ensure_pretrained_style_encoder(path: str | Path, quiet: bool = False) -> str:
    """Return a cached official style checkpoint, downloading it atomically if absent."""
    path = Path(path)
    if path.is_file() and path.stat().st_size > 0:
        return str(path)

    try:
        import gdown
    except ImportError as error:
        raise RuntimeError("Automatic style-checkpoint download requires the 'gdown' package") from error

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists():
        partial.unlink()
    try:
        downloaded = gdown.download(
            id=STYLE_ENCODER_GDRIVE_ID,
            output=str(partial),
            quiet=quiet,
        )
        if downloaded is None or not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError("Google Drive did not return the pretrained style checkpoint")
        partial.replace(path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return str(path)
