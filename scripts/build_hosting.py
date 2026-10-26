# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pillow==13.0.0",
#   "pandas==3.0.6",
#   "pyarrow==26.0.0",
#   "openpyxl==3.1.5",
#   "gpxpy==1.6.2",
#   "numpy==2.5.4",
# ]
# ///
"""Build Vue and assemble the public media and static API tree used by Firebase Hosting."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.store import get_store
from scripts.prepare_photos import main as prepare_photos


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


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
    dist = frontend / "dist"
    media = dist / "media"
    if media.exists():
        shutil.rmtree(media)
    for folder in ("fotos", "fotos_conteo_directo"):
        shutil.copytree(
            ROOT / "data" / folder,
            media / folder,
            ignore=shutil.ignore_patterns(".*", "~$*"),
        )
    shutil.copyfile(ROOT / "data" / "logo_vuelvealoceano.png", media / "logo")

    store = get_store()
    api_dir = dist / "api"
    if api_dir.exists():
        shutil.rmtree(api_dir)
    _write_json(api_dir / "meta", store.meta())
    _write_json(api_dir / "tracks", store.tracks())
    print(f"Firebase Hosting listo: {dist}")


if __name__ == "__main__":
    main()
