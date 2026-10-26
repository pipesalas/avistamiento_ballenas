# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pillow==13.0.0",
# ]
# ///
"""Build Vue and assemble the public media tree used by Firebase Hosting."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_photos import main as prepare_photos


def main() -> None:
    frontend = ROOT / "frontend"
    result = subprocess.run(["npm", "--prefix", str(frontend), "ci", "--prefer-offline"])
    if result.returncode != 0:
        with tempfile.TemporaryDirectory(prefix="npm-cache-") as cache_dir:
            subprocess.run(
                ["npm", "--prefix", str(frontend), "--cache", cache_dir, "ci"],
                check=True,
            )
    subprocess.run(["npm", "--prefix", str(frontend), "run", "build"], check=True)
    prepare_photos()
    media = frontend / "dist" / "media"
    if media.exists():
        shutil.rmtree(media)
    for folder in ("fotos", "fotos_conteo_directo"):
        shutil.copytree(
            ROOT / "data" / folder,
            media / folder,
            ignore=shutil.ignore_patterns(".*", "~$*"),
        )
    shutil.copyfile(ROOT / "data" / "logo_vuelvealoceano.png", media / "logo")
    print(f"Firebase Hosting listo: {frontend / 'dist'}")


if __name__ == "__main__":
    main()
