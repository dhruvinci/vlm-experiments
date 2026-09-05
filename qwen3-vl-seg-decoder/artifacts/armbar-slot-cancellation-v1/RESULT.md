# Armbar actor-slot cancellation result

## Decision

Subtracting each actor's `identity_only` marker state from the corresponding action or contact
marker state does **not** reliably improve actor-conditioned ownership decoding. The experiment
completed all 24 preregistered jobs, but six of the seven scientific gates failed. Only
background stability passed.

This rules out the simplest version of the slot-cancellation hypothesis: a constant
`condition - identity` vector is not, by itself, a usable spatial correction for the trapped-arm
error. It does not justify LoRA training. The result remains exploratory because it uses one
source video and legacy pseudo-label supervision.

## Immutable provenance

- Campaign result: `decoder-runs/armbar-slot-cancellation-v1/campaign-result.json`
- Campaign-result SHA-256:
  `48f994b7e555d2692258aff363ac6c71c40186ccab13fa306554c4a18bd30f46`
- Campaign source commit: `552478e`
- Frozen representation audit SHA-256:
  `7001b12786324c92422eac5b9f370bba30dedd8738526c5d9f72c8d866ed583f`
- Reference armbar campaign SHA-256:
  `f68727ced2434a104fb35fb28436333bea79794907928ca3abdd903c8b129445`
- Supervision status: `exploratory_legacy_pseudo_labels`
- Seeds: 7, 71, 701
- Completed jobs: 24/24, with matching per-job result checksums
- Non-finite result values: 0

The language layers were fixed before this armbar evaluation by a label-free geometry audit of
the four cached breadth clips. The audit selected action layer 12 and contact layer 45 using
role-signed leave-one-clip-out alignment. No armbar test label was used to select either layer.

## Tested comparison

The spatial decoder was held fixed to the previously selected Qwen vision input: full-resolution
layer 11 plus the final merged visual embedding. Four semantic representations were trained:

1. raw `action_relational` actor states at language layer 12;
2. `action_relational - identity_only` per actor at layer 12;
3. raw `contact_ownership` actor states at language layer 45;
4. `contact_ownership - identity_only` per actor at layer 45.

Every representation was paired with a residual-norm-matched random control and trained under
the same three seeds. The image, split, optimizer, decoder architecture, and actor convention
were unchanged.

## Held-out result

Values are means across the three seeds.

| Representation | Macro actor IoU | Contact accuracy | Contact margin | Background stability |
|---|---:|---:|---:|---:|
| Raw action | 0.8510 | 0.0000 | -0.9885 | 0.9978 |
| Action minus identity | 0.8602 | 0.0000 | -0.9871 | 0.9982 |
| Matched random action-delta | 0.8579 | 0.0000 | -0.9896 | 0.9978 |
| Raw contact | **0.8597** | 0.0370 | **-0.8609** | 0.9979 |
| Contact minus identity | 0.8588 | 0.0000 | -0.9822 | 0.9975 |
| Matched random contact-delta | 0.8631 | 0.0000 | -0.9165 | 0.9977 |

Action cancellation improved IoU over raw action by only `+0.00913`, just below the registered
`+0.01` gate, and beat its random control by only `+0.00232`. The per-seed raw-IoU deltas were
`+0.01563`, `-0.00226`, and `+0.01403`, so the effect was not consistently positive.

Contact cancellation was worse: it changed IoU by `-0.00087` versus raw contact and by
`-0.00429` versus its random control. Its contact-margin change versus raw contact was
`-0.12130`. In the most informative seed, raw contact assigned 3/27 trapped-arm cells correctly
with a margin of `-0.6170`; cancellation assigned 0/27 correctly and pushed the margin to
`-0.9955`.

| Gate | Result |
|---|---|
| Action delta beats raw by at least 0.01 IoU | Fail |
| Action delta beats matched random by at least 0.01 IoU | Fail |
| Action delta improves contact margin by at least 0.05 | Fail |
| Contact delta beats raw by at least 0.01 IoU | Fail |
| Contact delta beats matched random by at least 0.01 IoU | Fail |
| Contact delta improves contact margin by at least 0.05 | Fail |
| Background stability is at least 0.90 | Pass |

## Visual audit

Checksum-bound six-panel diagnostics were rendered locally under
`diagnostics/armbar-slot-cancellation-v1/` for seed 701's raw and cancelled action/contact
models. They confirm the numerical result: global athlete contours remain strong while the
A2-minus-A1 field stays confidently on the wrong side at the yellow trapped-arm contact cells.
The delta representation changes boundaries but does not make the semantic correction local.

## Execution incident and recovery

The campaign did not exhaust host or GPU memory. After 22 jobs, the laptop automatically
suspended for several hours. On resume, one `nvidia-smi` sample was unavailable while the NVIDIA
driver was waking, so the fail-closed guard terminated that child. The machine then hard-rebooted
about one minute after resume. The previous boot contains no kernel OOM, `systemd-oomd` kill,
NVIDIA Xid, or kernel panic record.

At the incident, the child scope reported a 1.6 GiB host-memory peak, approximately 5.38 GiB
free VRAM, and over 8 GiB available host memory. The reboot interrupted epoch 26 between the
atomic tensor rename and its manifest commit. The unverified tensor and temporary manifest were
preserved under the job's `failures/` directory, epoch 25 was revalidated by SHA-256, and the job
resumed from it. A detached `systemd` service plus execution-scoped sleep inhibitor completed the
remaining two jobs.

Across the final 24 successful guarded attempts, observed host availability never fell below
7,522,308,096 bytes, GPU free memory never fell below 5,214,568,448 bytes, global GPU use never
exceeded 19.1%, and no successful attempt crossed a resource limit. The longest job took 542.2
seconds. The completed campaign occupies about 1.4 GiB.

Two regression-tested hardening changes follow from the incident:

- invalid or half-committed checkpoint artifacts are automatically quarantined before a verified
  checkpoint with the same epoch is rewritten; verified checkpoints remain immutable;
- the resource guard records and tolerates up to three consecutive GPU-telemetry gaps, while real
  RAM/VRAM violations still terminate immediately and a persistent telemetry outage still fails
  closed.

The complete regression suite passes under the same 4 GiB/no-swap cgroup.

## What this teaches us

Qwen's cached spatial features clearly contain substantial actor-ownership information, but the
global marker vectors do not provide the missing local correspondence. The four-clip geometry
audit shows that role-related directions exist in the language states; this experiment shows that
simply subtracting an identity direction does not tell a spatial decoder where the trapped limb
is.

The next decisive training experiment remains clip-held-out evaluation on reviewed masks from
the four cached breadth clips. If multi-clip training still cannot exploit real states relative to
wrong-clip and matched-random controls, the next extraction should be frame-varying,
query-conditioned spatial features or a learned cross-attention bridge—not a larger decoder and
not LoRA by default.
