# Frozen Blackwell mask runtime

This Python 3.12 environment is separate from the Qwen extraction environment because
the pinned official SAM3 repository requires NumPy `<2`. It supports two strictly
sequential stages: tracker-only SAM3.1 multiplex propagation, followed only after full
release by base-SAM3 image segmentation through Transformers 5.16.1.

Run `bootstrap-mask-env.sh VENV_PATH SAM_REPO_PATH`. The script refuses to mutate an
existing unverified environment, binds the official SAM checkout to revision
`8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`, installs CUDA 13.0 PyTorch separately from
the PyPI dependency set, performs exact import/version checks, and writes a marker that
binds both requirement-file hashes and the repository revision.

Do not install FlashAttention, enable `torch.compile`, use a Max-Q GPU, or share this
environment with Qwen. Both model adapters force BF16, eager execution, and the
reference/SDPA paths needed by the campaign.
