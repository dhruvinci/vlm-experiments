# Qwen3-VL actor-ownership decoder campaign

This work package tests whether frozen Qwen spatial representations plus actor-conditioned
semantic states can produce mutually exclusive `background` / `A1` / `A2` ownership fields
in high-contact grappling footage. The Qwen and SAM models remain frozen. Only a small spatial
decoder is trained locally.

## Current status

- The immutable SAM3.1/SAM3 mask launch packet is frozen at
  `launch-packet/sam31-breadth-v2/manifest.json` with SHA-256
  `86126c36091adf70e211dd6d51e6e53b1fdd4e1c1e0bfb8966f143c75e49e097`.
- Its 717 staged objects (7,100,837,883 bytes) are verified on RunPod network volume
  `0vnqaqwt1r` in `US-NC-2`.
- The exact local Qwen breadth inputs consumed by the decoder were rehashed: 600 artifacts,
  7,169,080,736 bytes, inventory SHA-256
  `10468102a91f4f0a45d911ccc05cfaa84dd2e062397aa9affc96585d6c6ca673`.
- Local synthetic end-to-end and failure-path tests pass. No new paid Pod has been launched for
  this mask/decoder phase.

The paid run remains deliberately gated on explicit spend authorization. Never launch a Pod
merely by following this README.

### Local armbar exploratory path

The existing 32-frame armbar cache can be used immediately, without loading Qwen or SAM and
without returning to RunPod, through `scripts/run_armbar_decoder_campaign.py`. This path is
explicitly labeled `exploratory_legacy_pseudo_labels`: the actor interiors are conservative
pseudo-labels and only the final trapped-arm contact region has manual ownership truth. It is
therefore useful for selecting cache layers, testing the query-to-pixel decoder, measuring the
held-out trapped-arm margin, and falsifying semantic/temporal hypotheses, but it cannot satisfy
the multi-clip north-star gate or authorize LoRA.

The armbar controller first screens RGB, frozen Qwen spatial arms, and language layers 25/60 on
the original four-frame validation split. It then refits the selected architecture on all 24
non-test frames with matched seeds and compares static, real action-conditioned, and
residual-norm-matched random models on the untouched eight-frame test set. Identity, contact,
zero, and mean controls are retained, and the real action model is re-evaluated with ordered,
reversed, shuffled, single-frame, 2 fps, 5-frame, 8 fps, and thinking-mode marker states.

Every job uses the same 4 GiB/no-swap cgroup and global RAM/VRAM circuit breakers as the breadth
campaign. Results, work items, checkpoints, logs, per-second telemetry, per-frame metrics, and
guard decisions are atomic and resumable. Only cached tensors and the small decoder enter local
CUDA memory.

## Scientific design

The breadth set contains four six-second, high-contact clips with 24 sampled frames each:
`back_seatbelt`, `guard_scramble`, `half_guard`, and `mount`. Entire clips—not neighboring
frames—are reserved for validation and outer testing.

The static screen compares:

1. RGB only;
2. Qwen full-resolution vision layer 11;
3. Qwen pooled vision layer 12;
4. final merged Qwen vision embeddings;
5. layer 11 plus merged embeddings;
6. full-resolution layers 5, 11, 18, and 26.

The semantic decoder projects the frozen A1 and A2 marker states into spatial queries and
scores them against every projected pixel. Its construction is exactly actor-swap equivariant:
swapping only the two queries swaps only the two actor logits while preserving background.
The selected static arm is combined with `identity_only`, `action_relational`, and
`contact_ownership` video-conditioned states. Language layers 25 and 60 are screened without
using the outer held-out clip.

The current breadth cache contains one clip-level ordered-4-fps marker pair per condition.
It can test whether video-conditioned semantic queries improve ownership, but it cannot by
itself establish an ordered-temporal advantage over reversed or shuffled context. That causal
temporal claim remains a separate gate.

## End-to-end order

1. Run the 24 GB RTX Pro 6000 Blackwell MIG preflight against the frozen packet.
2. Only if the packet-bound MIG sentinel passes, run the 96 GB non-Max-Q Pod. SAM3.1 temporal
   tracking runs first; its model is released before base-SAM3 image masks are loaded.
3. Terminate compute, then stream the completed network-volume prefix locally with
   `scripts/download_mask_campaign.py`. The downloader freezes the remote inventory, writes
   atomically, resumes per object, and rehashes every local file.
4. Build non-training agreement candidates with `scripts/manage_breadth_labels.py`. A human must
   resolve every contact-owner cell and explicitly attest the review before labels become
   training eligible.
5. Run `scripts/run_local_decoder_campaign.py`. It executes 40 isolated jobs serially, with one
   frame resident at a time, checkpoint resume, host/VRAM circuit breakers, and no swap inside
   child cgroups.
6. Run `scripts/render_decoder_campaign.py` to render every selected static and semantic model's
   held-out predictions. Each frame produces a source/truth/prediction/A2-minus-A1/contact-zoom
   panel and a compact probability tensor.

All path and runtime arguments are explicit; use `--help` on each entrypoint. Secrets are read
from the external environment file and are never part of this repository.

## North-star gate

A positive result requires all of the following on clip-held-out evaluation:

- action-conditioned macro actor IoU at least 0.60;
- at least +0.03 macro actor IoU or +0.10 contact accuracy over the selected static baseline;
- contact accuracy at least 0.70 and positive ownership margin in at least 75% of contact regions;
- background stability at least 0.90;
- real actor states beat both a true wrong-clip semantic state and a residual-norm-matched random
  state by at least 0.01 IoU, and beat zero/mean states by at least 0.03 IoU;
- swapping A1/A2 queries flips at least 75% of actor predictions while changing background
  probability by at most 0.01.

The report includes exact paired clip-bootstrap intervals. A valid negative result is still a
successful experiment and does not authorize a replacement Pod or LoRA training.

## Safety and verification

- Remote execution accepts only the 24 GB MIG preflight or the 96 GB RTX Pro 6000 Blackwell
  Server/Workstation Edition; Max-Q and substitute GPU families fail closed.
- The full worker has a hard runtime ceiling, a spend guard, a 30-second in-Pod watchdog, an
  independent local monitor, atomic artifacts, checksums, and at most two fresh-process retries.
- Local decoder jobs run serially in `systemd` cgroups. Each child has a 4 GiB hard RAM cap,
  no swap, a 60% PyTorch allocator ceiling, and a 30-minute wall-clock limit. The independent
  parent stops a job before host availability falls below 4 GiB, system swap has less than
  3 GiB free, or global GPU use exceeds 75%. SIGINT/SIGTERM and child OOMs kill the complete
  process group while preserving checkpoints, logs, telemetry, and the guard result.
- Qwen and SAM weights are never loaded on the local machine by this package.

Run the regression suite CPU-only with an external cgroup, for example:

```bash
systemd-run --user --scope --quiet \
  -p MemoryHigh=3G -p MemoryMax=4G -p MemorySwapMax=0 -- \
  env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$PWD/src" /path/to/python -m unittest discover -s tests -v
```
