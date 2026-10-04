"""Download only inference assets; keep all model caches inside this project."""

import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import requests
from huggingface_hub import HfApi, snapshot_download

from config import EMBEDDING_MODEL, MODEL_CACHE_DIR, RERANK_MODEL


def download(model):
    print(f"Downloading {model}", flush=True)
    path = snapshot_download(
        model,
        cache_dir=MODEL_CACHE_DIR,
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "pytorch_model.bin"],
        ignore_patterns=["onnx/*", "openvino/*"],
        max_workers=4,
    )
    print(f"Ready: {model}: {path}", flush=True)
    return path


def download_ranged(model, use_ranges=True):
    """Resume verified HTTP ranges when the optimized transport stalls on a proxy."""
    info = HfApi().model_info(model, files_metadata=True)
    root = Path(MODEL_CACHE_DIR).resolve()
    repo = root / ("models--" + model.replace("/", "--"))
    snapshot = repo / "snapshots" / info.sha
    snapshot.mkdir(parents=True, exist_ok=True)
    has_safetensors = any(f.rfilename == "model.safetensors" for f in info.siblings)
    assets = [
        f
        for f in info.siblings
        if not f.rfilename.startswith(("onnx/", "openvino/"))
        and (
            f.rfilename.endswith((".json", ".txt", ".model", ".safetensors"))
            or (f.rfilename == "pytorch_model.bin" and not has_safetensors)
        )
    ]
    for asset in assets:
        target = (snapshot / asset.rfilename).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Model asset escapes project cache")
        if target.exists() and target.stat().st_size == asset.size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{model}/resolve/{info.sha}/{asset.rfilename}"
        if asset.size < 32 * 1024 * 1024:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            if len(response.content) != asset.size:
                raise ValueError("Model asset size mismatch")
            if asset.lfs and hashlib.sha256(response.content).hexdigest() != asset.lfs.sha256:
                raise ValueError("Downloaded model SHA256 mismatch")
            target.write_bytes(response.content)
            continue
        if not use_ranges:
            partial = target.with_suffix(target.suffix + ".download")
            resumed = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={resumed}-"} if resumed else {}
            print(f"Streaming {model}/{asset.rfilename}: {asset.size / 1024**2:.0f} MB", flush=True)
            with requests.get(url, headers=headers, stream=True, timeout=60) as result:
                result.raise_for_status()
                if resumed and result.status_code != 206:
                    resumed = 0
                transferred = resumed
                with partial.open("ab" if resumed else "wb") as handle:
                    for block in result.iter_content(1024 * 1024):
                        handle.write(block)
                        previous = transferred
                        transferred += len(block)
                        if transferred // (64 * 1024 * 1024) > previous // (64 * 1024 * 1024):
                            print(
                                f"  {model}: {transferred / 1024**2:.0f}/{asset.size / 1024**2:.0f} MB",
                                flush=True,
                            )
            if partial.stat().st_size != asset.size:
                raise ValueError("Model size mismatch")
            digest = hashlib.sha256()
            with partial.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    digest.update(block)
            if asset.lfs and digest.hexdigest() != asset.lfs.sha256:
                raise ValueError("Downloaded model SHA256 mismatch")
            os.replace(partial, target)
            continue
        response = requests.head(url, allow_redirects=True, timeout=60)
        response.raise_for_status()
        direct_url = response.url
        part_dir = root / "transfers" / model.replace("/", "--") / asset.rfilename
        part_dir.mkdir(parents=True, exist_ok=True)
        part_size = 8 * 1024 * 1024
        n_parts = (asset.size + part_size - 1) // part_size

        def fetch_part(index):
            start = index * part_size
            end = min(asset.size, start + part_size) - 1
            path = part_dir / f"{index:05d}.part"
            if path.exists() and path.stat().st_size == end - start + 1:
                return path
            for attempt in range(3):
                try:
                    with requests.get(
                        direct_url, headers={"Range": f"bytes={start}-{end}"}, timeout=60, stream=True
                    ) as result:
                        if (
                            result.status_code != 206
                            or result.headers.get("Content-Range") != f"bytes {start}-{end}/{asset.size}"
                        ):
                            raise ValueError("Server did not return the requested model byte range")
                        with path.open("wb") as handle:
                            for block in result.iter_content(1024 * 1024):
                                handle.write(block)
                    if path.stat().st_size != end - start + 1:
                        raise ValueError("Incomplete model byte range")
                    return path
                except (requests.RequestException, ValueError):
                    if attempt == 2:
                        raise

        print(
            f"Downloading {model}/{asset.rfilename}: {asset.size / 1024**2:.0f} MB, {n_parts} ranges",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=8) as executor:
            for i, _ in enumerate(executor.map(fetch_part, range(n_parts)), 1):
                if i % 16 == 0 or i == n_parts:
                    print(f"  {model}: {i}/{n_parts} ranges ready", flush=True)
        digest = hashlib.sha256()
        assembly = target.with_suffix(target.suffix + ".assembly")
        with assembly.open("wb") as handle:
            for i in range(n_parts):
                with (part_dir / f"{i:05d}.part").open("rb") as part:
                    while block := part.read(1024 * 1024):
                        digest.update(block)
                        handle.write(block)
        if asset.lfs and digest.hexdigest() != asset.lfs.sha256:
            raise ValueError("Downloaded model SHA256 mismatch")
        os.replace(assembly, target)
    (repo / "refs").mkdir(exist_ok=True)
    (repo / "refs" / "main").write_text(info.sha, encoding="utf-8")
    print(f"Ready: {model}", flush=True)
    return str(snapshot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=("hub", "ranged", "stream"), default="hub")
    parser.add_argument("--models", nargs="+", help="Optional explicit model IDs for a separate smoke run")
    args = parser.parse_args()
    models = list(
        dict.fromkeys(
            args.models or ["sentence-transformers/all-MiniLM-L6-v2", EMBEDDING_MODEL, RERANK_MODEL]
        )
    )
    with ThreadPoolExecutor(max_workers=3) as executor:
        if args.transport == "stream":
            list(executor.map(lambda model: download_ranged(model, use_ranges=False), models))
        else:
            list(executor.map(download_ranged if args.transport == "ranged" else download, models))
