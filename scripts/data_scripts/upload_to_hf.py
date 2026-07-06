#!/usr/bin/env python3
"""Upload a local folder to a Hugging Face Hub dataset repo.

Uses HfApi.upload_folder, which (with hf_xet installed) streams uploads,
hashes/chunks in a single pass, auto-splits into multiple commits for large
folders, and resumes cleanly if interrupted (already-committed files are
skipped, already-uploaded data is deduplicated).

The Hub caps each repo directory at 10,000 files. data/fafb/blocks has
524,288 files (image_*.tif + segment_*.tif for a 64x64x64 block grid) in one
flat local directory, so --shard_by_z uploads one subdirectory per z value
(64 shards x 8,192 files each) instead of mirroring the flat layout.
"""
import argparse

from huggingface_hub import HfApi


def upload_sharded_by_z(api, cfg, z_range):
    failed = []
    for z in z_range:
        path_in_repo = f"{cfg.path_in_repo}/z{z:03d}" if cfg.path_in_repo else f"z{z:03d}"
        print(f"--- shard z={z:03d} -> {path_in_repo} ---", flush=True)
        try:
            api.upload_folder(
                repo_id=cfg.repo_id,
                repo_type=cfg.repo_type,
                folder_path=cfg.folder_path,
                path_in_repo=path_in_repo,
                allow_patterns=[f"*_z{z}_y*_x*.tif"],
            )
        except Exception as e:
            print(f"!!! shard z={z:03d} failed: {e}", flush=True)
            failed.append(z)
    if failed:
        print(f"Finished with {len(failed)} failed shard(s): {failed}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo_id", required=True, help="e.g. Arminshfard/fafb-em-blocks")
    p.add_argument("--folder_path", required=True)
    p.add_argument("--path_in_repo", default=None)
    p.add_argument("--repo_type", default="dataset", choices=["dataset", "model", "space"])
    p.add_argument("--private", action="store_true", default=False,
                   help="Only applies when the repo doesn't exist yet; ignored otherwise")
    p.add_argument("--shard_by_z", action="store_true",
                   help="Upload one subdirectory per z value instead of a flat folder "
                        "(needed for data/fafb/blocks: 524,288 files > the Hub's "
                        "10,000-files-per-directory limit)")
    p.add_argument("--z_start", type=int, default=0)
    p.add_argument("--z_end", type=int, default=64, help="exclusive")
    cfg = p.parse_args()

    api = HfApi()
    api.create_repo(repo_id=cfg.repo_id, repo_type=cfg.repo_type, private=cfg.private, exist_ok=True)

    print(f"Uploading {cfg.folder_path} -> {cfg.repo_id} ({cfg.repo_type}) "
          f"path_in_repo={cfg.path_in_repo!r} shard_by_z={cfg.shard_by_z} "
          f"z_range=[{cfg.z_start}, {cfg.z_end})", flush=True)

    if cfg.shard_by_z:
        upload_sharded_by_z(api, cfg, range(cfg.z_start, cfg.z_end))
    else:
        api.upload_folder(
            repo_id=cfg.repo_id,
            repo_type=cfg.repo_type,
            folder_path=cfg.folder_path,
            path_in_repo=cfg.path_in_repo,
        )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
