# SAM3.1 + SAM3 breadth campaign v2

This immutable packet generates candidate ownership masks for 96 cached frames from four clips. It runs a geometry-only SAM3.1 temporal tracker first, releases it, then runs base SAM3 image PVS one frame at a time from tracker-derived geometric prompts. It never loads both models concurrently.

The 24 GB Blackwell MIG preflight must complete first. It verifies the exact environment, repository, artifacts, storage headroom, SM120 runtime, and a bounded BF16 SDPA operation without loading model weights. The full 96 GB Pod refuses to start unless the resulting sentinel is bound to this packet hash.

Outputs are candidate labels, not ground truth. Contact-region ownership must be reviewed and corrected before any decoder training. Base-SAM3 localization is not fully independent because its geometry prompts are derived from SAM3.1 masks; this limitation is recorded in the launch manifest and all output provenance.

Both Pod types have an in-Pod 30-second guard and a separate local 60-second controller. A launch or bootstrap failure writes `RUN_FATAL`; successful integrity completion writes `RUN_COMPLETE`. Either sentinel triggers termination. No replacement Pod is permitted automatically.
