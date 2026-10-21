"""Build Vue and assemble the public media tree used by Firebase Hosting."""

from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_photos import main as prepare_photos


def main() -> None:
    frontend = ROOT / "frontend"
    subprocess.run(["npm", "--prefix", str(frontend), "ci"], check=True)
    subprocess.run(["npm", "--prefix", str(frontend), "run", "build"], check=True)
    prepare_photos()
    media = frontend / "dist" / "media"
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
