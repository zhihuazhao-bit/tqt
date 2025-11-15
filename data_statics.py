import json
from pathlib import Path
from typing import Dict, List

#!/usr/bin/env python3
"""
Collect image filenames per scene under train/val/test in /root/tqdm/dataset/road3d
Saves JSON files: train_scenes.json, val_scenes.json, test_scenes.json and combined scenes_all.json
"""

BASE_DIR = Path("/root/tqdm/dataset/orfdv2")
SPLITS = ("training", "validation", "testing", "")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images_for_split(split_dir: Path) -> Dict[str, List[str]]:
    scenes = {}
    if not split_dir.exists():
        return scenes
    for scene in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        imgs = []
        # collect image files recursively inside each scene directory
        for f in sorted(scene.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                imgs.append(f.name)  # store image filename only (not full path)
        scenes[scene.name] = imgs
    return scenes


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    all_data = {}
    for split in SPLITS:
        split_dir = BASE_DIR / split
        scenes = collect_images_for_split(split_dir)
        all_data[split] = scenes
        out_file = BASE_DIR / f"{split}_scenes.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(scenes, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(scenes)} scenes for '{split}' -> {out_file}")

    # save combined file
    combined_file = BASE_DIR / "scenes_all.json"
    with combined_file.open("w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"Saved combined data -> {combined_file}")


if __name__ == "__main__":
    main()