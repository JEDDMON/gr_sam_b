#!/usr/bin/env python3
"""
Step 1: Download SAM ViT-B Checkpoint and a 100k-200k subset of PD12M (CC0 Public Domain).
Features:
- Streamed metadata reading from Hugging Face `Spawning/PD12M` (no massive metadata download needed).
- Asynchronous multi-threaded image downloading via aiohttp.
- Pre-resizes images so the longest side is 1024px (saves 70%+ NVMe space and speeds up preprocessing).
- Fully resumable: skips already downloaded images.
"""

import os
import io
import csv
import asyncio
import argparse
import urllib.request
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import aiohttp
from datasets import load_dataset

SAM_VIT_B_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

def download_sam_checkpoint(ckpt_dir: Path) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "sam_vit_b_01ec64.pth"
    if ckpt_path.exists() and ckpt_path.stat().st_size > 370_000_000:
        print(f"[✓] SAM ViT-B checkpoint already exists at {ckpt_path}")
        return ckpt_path

    print(f"[*] Downloading SAM ViT-B checkpoint (~375 MB) to {ckpt_path}...")
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc="sam_vit_b_01ec64.pth") as t:
        def reporthook(blocknum, blocksize, totalsize):
            t.total = totalsize
            t.update(blocknum * blocksize - t.n)
        urllib.request.urlretrieve(SAM_VIT_B_URL, filename=ckpt_path, reporthook=reporthook)
    print(f"[✓] Downloaded SAM ViT-B checkpoint.")
    return ckpt_path


async def fetch_and_save_image(session: aiohttp.ClientSession, url: str, out_path: Path, semaphore: asyncio.Semaphore) -> bool:
    if out_path.exists() and out_path.stat().st_size > 1024:
        return True

    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=12)) as resp:
                if resp.status != 200:
                    return False
                data = await resp.read()
            
            # Decode and validate image in memory, resize longest side to 1024 to save disk space
            img = Image.open(io.BytesIO(data)).convert("RGB")
            w, h = img.size
            if min(w, h) < 256:  # Skip tiny icons/thumbnails
                return False

            scale = 1024.0 / max(w, h)
            if scale < 1.0:
                new_w, new_h = int(round(w * scale)), int(round(h * scale))
                img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, format="JPEG", quality=92, optimize=True)
            return True
        except Exception:
            return False


async def download_pd12m_subset(out_dir: Path, target_count: int, concurrency: int = 64):
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    # Count existing valid images for seamless resume
    existing_files = sorted(images_dir.glob("*/*.jpg"))
    valid_count = len(existing_files)
    if valid_count >= target_count:
        print(f"[✓] Already have {valid_count} images in {images_dir}. Skipping download.")
        return

    print(f"[*] Streaming PD12M (CC0 Public Domain) metadata from Hugging Face...")
    print(f"[*] Target: {target_count} images | Existing: {valid_count} | Concurrency: {concurrency}")

    ds = load_dataset("Spawning/PD12M", split="train", streaming=True)
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency, ssl=False)

    pbar = tqdm(total=target_count, initial=valid_count, desc="Downloading CC0 Images", unit="img")
    
    # Open manifest in append mode
    manifest_exists = manifest_path.exists()
    manifest_file = open(manifest_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(manifest_file)
    if not manifest_exists:
        writer.writerow(["image_id", "relative_path", "source_url"])

    batch_tasks = []
    batch_meta = []
    batch_size = concurrency * 4

    async with aiohttp.ClientSession(connector=connector) as session:
        for idx, row in enumerate(ds):
            if valid_count >= target_count:
                break

            url = row.get("url")
            if not url:
                continue

            # Organize into subfolders of 1,000 images to avoid Linux ext4 directory slowdowns
            shard_id = f"{idx // 1000:05d}"
            img_name = f"pd12m_{idx:08d}.jpg"
            rel_path = f"{shard_id}/{img_name}"
            full_path = images_dir / rel_path

            if full_path.exists() and full_path.stat().st_size > 1024:
                continue

            batch_tasks.append(fetch_and_save_image(session, url, full_path, semaphore))
            batch_meta.append((idx, rel_path, url))

            if len(batch_tasks) >= batch_size:
                results = await asyncio.gather(*batch_tasks)
                for success, meta in zip(results, batch_meta):
                    if success and valid_count < target_count:
                        writer.writerow(meta)
                        valid_count += 1
                        pbar.update(1)
                manifest_file.flush()
                batch_tasks.clear()
                batch_meta.clear()

        # Process remaining tasks
        if batch_tasks and valid_count < target_count:
            results = await asyncio.gather(*batch_tasks)
            for success, meta in zip(results, batch_meta):
                if success and valid_count < target_count:
                    writer.writerow(meta)
                    valid_count += 1
                    pbar.update(1)
            manifest_file.flush()

    pbar.close()
    manifest_file.close()
    print(f"[✓] Successfully curated {valid_count} CC0 images in {images_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download SAM ViT-B and PD12M CC0 Subset")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root directory for dataset and checkpoints")
    parser.add_argument("--num_images", type=int, default=100000, help="Number of CC0 images to download (e.g., 100000 or 200000)")
    parser.add_argument("--concurrency", type=int, default=64, help="Number of concurrent async HTTP connections")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    ckpt_dir = data_root / "checkpoints"
    
    # 1. Download SAM ViT-B Teacher Checkpoint
    download_sam_checkpoint(ckpt_dir)

    # 2. Download PD12M CC0 Subset
    asyncio.run(download_pd12m_subset(data_root, args.num_images, args.concurrency))


if __name__ == "__main__":
    main()
