"""Download / refresh the latest official DeOldify + DDColor weights.

DeOldify: open-source weights stopped at Artistic / Stable / Video (no newer release).
DDColor: latest Hugging Face releases are modelscope (default) + artistic.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# Prefer China mirror when HF is slow; override with HF_ENDPOINT if needed.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(Path(r"D:\AI_Temp") / "hf_cache"))
os.environ.setdefault("TORCH_HOME", str(Path(r"D:\AI_Temp") / "torch_cache"))

# DeepAI often times out in CN; use HF mirrors first.
DEOLDIFY_HF = {
    "ColorizeArtistic_gen.pth": ("databuzzword/deoldify-artistic", "ColorizeArtistic_gen.pth"),
    "ColorizeStable_gen.pth": ("databuzzword/deoldify-stable", "ColorizeStable_gen.pth"),
    "ColorizeVideo_gen.pth": ("spensercai/DeOldify", "ColorizeVideo_gen.pth"),
}

DEOLDIFY_URL_FALLBACK = {
    "ColorizeArtistic_gen.pth": "https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth",
    "ColorizeVideo_gen.pth": "https://data.deepai.org/deoldify/ColorizeVideo_gen.pth",
    "ColorizeStable_gen.pth": "https://www.dropbox.com/s/axsd2g85uyixaho/ColorizeStable_gen.pth?dl=1",
}

DDCOLOR_HF = {
    # Latest recommended qualitative model
    "ddcolor_modelscope": "piddnad/ddcolor_modelscope",
    # Fewer unnatural color blobs; good companion to modelscope
    "ddcolor_artistic": "piddnad/ddcolor_artistic",
}

# Incomplete / leftover junk under models/
JUNK_NAMES = ("tmp889mh8wz", "tmpk0eqqdts")


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _download_url(url: str, dest: Path, min_bytes: int = 50_000_000) -> None:
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"[skip] {dest.name} already present ({_human(dest.stat().st_size)})")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[down] {dest.name}")
    print(f"       {url}")

    last_pct = [-1]

    def progress(block_num, block_size, total_size):
        if total_size <= 0:
            return
        pct = int(block_num * block_size * 100 / total_size)
        if pct != last_pct[0] and pct % 5 == 0:
            last_pct[0] = pct
            done = min(block_num * block_size, total_size)
            print(f"       {pct:3d}%  {_human(done)} / {_human(total_size)}", flush=True)

    try:
        urlretrieve(url, tmp, reporthook=progress)
        size = tmp.stat().st_size
        if size < min_bytes:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"download too small ({_human(size)}), likely failed")
        tmp.replace(dest)
        print(f"[ok]   {dest.name} ({_human(size)})")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"[fail] {dest.name}: {e}")
        raise


def cleanup_junk() -> None:
    for name in JUNK_NAMES:
        path = MODELS / name
        if path.exists():
            size = path.stat().st_size
            path.unlink()
            print(f"[clean] removed {name} ({_human(size)})")

    # Old ambiguous 220MB bin is usually an incomplete / tiny leftover.
    legacy = MODELS / "pytorch_model.bin"
    if legacy.exists() and legacy.stat().st_size < 400_000_000:
        bak = MODELS / "pytorch_model.bin.bak_small"
        if bak.exists():
            bak.unlink()
        legacy.rename(bak)
        print(f"[clean] moved small pytorch_model.bin -> {bak.name}")


def ensure_deoldify(force_stable: bool = False) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None

    for name, (repo_id, filename) in DEOLDIFY_HF.items():
        dest = MODELS / name
        if dest.exists() and dest.stat().st_size >= 80_000_000 and not (
            force_stable and name == "ColorizeStable_gen.pth"
        ):
            print(f"[skip] {name} already present ({_human(dest.stat().st_size)})")
            continue

        ok = False
        if hf_hub_download is not None:
            print(f"[down] HF {repo_id}/{filename}")
            try:
                cached = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(MODELS / "_hf_tmp" / "deoldify"),
                )
                cached_path = Path(cached)
                if dest.exists():
                    dest.unlink()
                cached_path.replace(dest)
                print(f"[ok]   {dest.name} ({_human(dest.stat().st_size)})")
                ok = True
            except Exception as e:
                print(f"[warn] HF failed for {name}: {e}")

        if not ok:
            url = DEOLDIFY_URL_FALLBACK.get(name)
            if not url:
                print(f"[warn] continue without {name}")
                continue
            try:
                _download_url(url, dest, min_bytes=80_000_000)
            except Exception:
                if name == "ColorizeStable_gen.pth" and dest.exists():
                    print("[warn] Stable re-download failed; keeping existing file")
                else:
                    print(f"[warn] continue without {name}")


def ensure_ddcolor() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[fail] huggingface_hub missing. Run: pip install huggingface-hub")
        sys.exit(1)

    for short_name, repo_id in DDCOLOR_HF.items():
        dest = MODELS / f"{short_name}.bin"
        if dest.exists() and dest.stat().st_size >= 400_000_000:
            print(f"[skip] {dest.name} already present ({_human(dest.stat().st_size)})")
            continue

        print(f"[down] {repo_id} -> {dest.name}")
        try:
            cached = hf_hub_download(
                repo_id=repo_id,
                filename="pytorch_model.bin",
                local_dir=str(MODELS / "_hf_tmp" / short_name),
            )
            cached_path = Path(cached)
            if dest.exists():
                dest.unlink()
            cached_path.replace(dest)
            print(f"[ok]   {dest.name} ({_human(dest.stat().st_size)})")
        except Exception as e:
            # Fallback: ModelScope large weight already on disk for modelscope variant
            ms = ROOT / "modelscope" / "damo" / "cv_ddcolor_image-colorization" / "pytorch_model.pt"
            if short_name == "ddcolor_modelscope" and ms.exists() and ms.stat().st_size >= 400_000_000:
                import shutil

                shutil.copy2(ms, dest.with_suffix(".pt"))
                print(f"[ok]   copied ModelScope weight -> {dest.with_suffix('.pt').name}")
                print(f"       HF download failed: {e}")
            else:
                print(f"[fail] {short_name}: {e}")


def sync_modelscope_copy() -> None:
    """Expose ModelScope DDColor weight under models/ with a clear name."""
    src = ROOT / "modelscope" / "damo" / "cv_ddcolor_image-colorization" / "pytorch_model.pt"
    dest = MODELS / "ddcolor_modelscope.pt"
    if not src.exists():
        return
    if dest.exists() and dest.stat().st_size == src.stat().st_size:
        print(f"[skip] {dest.name} already synced")
        return
    # Prefer hardlink to save disk; fall back to copy.
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
        print(f"[ok]   hardlinked {dest.name}")
    except OSError:
        import shutil

        shutil.copy2(src, dest)
        print(f"[ok]   copied {dest.name}")


def main() -> None:
    Path(r"D:\AI_Temp").mkdir(exist_ok=True)
    print("=== cleanup ===")
    cleanup_junk()
    print("\n=== DeOldify (official latest / only set) ===")
    ensure_deoldify()
    print("\n=== DDColor (Hugging Face latest) ===")
    ensure_ddcolor()
    sync_modelscope_copy()
    print("\n=== models/ summary ===")
    for p in sorted(MODELS.glob("*")):
        if p.is_file():
            print(f"  {_human(p.stat().st_size):>10}  {p.name}")
    print("\nDone. Use: python zhixing.py --model ddcolor")
    print("          python zhixing.py --model deoldify")
    print("          python zhixing.py --model deoldify-artistic")


if __name__ == "__main__":
    main()
