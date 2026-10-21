"""Generate gallery thumbnails before deployment; originals remain for the lightbox.

Run from the repository root:
    uv run --with pillow python scripts/prepare_photos.py
"""

from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MAX_SIZE = (640, 640)


def thumbnail(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        image.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
        image.convert("RGB").save(destination, "WEBP", quality=80, method=6)


def main() -> None:
    count = original_bytes = thumbnail_bytes = 0
    for name in ("fotos", "fotos_conteo_directo"):
        folder = ROOT / "data" / name
        output = folder / "thumbnails"
        output.mkdir(exist_ok=True)
        for source in sorted(folder.iterdir()):
            if not source.is_file() or source.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            destination = output / f"{source.name}.webp"
            thumbnail(source, destination)
            count += 1
            original_bytes += source.stat().st_size
            thumbnail_bytes += destination.stat().st_size
    print(
        f"{count} miniaturas: {original_bytes / 2**20:.2f} MiB originales → "
        f"{thumbnail_bytes / 2**20:.2f} MiB WebP"
    )


if __name__ == "__main__":
    main()
