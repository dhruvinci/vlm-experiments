# SAM3.1 Blackwell runtime

This is a separate Python 3.12 environment from the Qwen extraction runtime because
official SAM3 requires NumPy `<2`.

Install `requirements-cu130.txt` first, then `requirements-sam31.txt`, then install the
pinned official SAM3 checkout in editable/no-dependencies mode. The worker independently
checks Python, GPU model, SM120 capability, driver, free VRAM/RAM/disk, repository commit,
installed core package versions, and the complete checkpoint SHA-256 before importing
Torch or SAM3.

The CUDA pairing follows the official PyTorch command for 2.12.1:
`torch==2.12.1`, `torchvision==0.27.1`, index `cu130`.

Do not install FlashAttention or enable `torch.compile`; the tracker builder explicitly
sets `use_fa3=False` and `compile=False`.
