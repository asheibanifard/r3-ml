"""Print volume metadata for all OCP channels without downloading voxels."""

import yaml
from cloudvolume import CloudVolume


def inspect(name: str, url: str, use_https: bool = True):
    vol = CloudVolume(url, mip=0, use_https=use_https, progress=False)
    print(f"\n{'='*60}")
    print(f"Channel : {name}")
    print(f"URL     : {url}")
    print(f"Shape   : {vol.volume_size}  (X, Y, Z)")
    print(f"Dtype   : {vol.dtype}")
    print(f"MIP lvls: {len(vol.available_mips)}")
    print(f"Voxel   : {vol.resolution} nm")
    cf = vol.bounds
    print(f"Bounds  : x={cf.minpt[0]}–{cf.maxpt[0]}, "
          f"y={cf.minpt[1]}–{cf.maxpt[1]}, "
          f"z={cf.minpt[2]}–{cf.maxpt[2]}")


if __name__ == "__main__":
    with open("configs/ocp_kasthuri.yml") as f:
        cfg = yaml.safe_load(f)

    for ch_name, url in cfg["channels"].items():
        inspect(ch_name, url, use_https=cfg["download"]["use_https"])
