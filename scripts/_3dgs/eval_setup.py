import torch
from torch.utils.cpp_extension import load
from pathlib import Path

HERE = Path(__file__).resolve().parent

ext = load(
    name="3dgs_eval_cuda",
    sources=[str(HERE / "3dgs_eval_cuda.cu")],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
    ],
    extra_cflags=[
        "-O3",
    ],
    verbose=True,
)

print("Compiled successfully")
print(ext)