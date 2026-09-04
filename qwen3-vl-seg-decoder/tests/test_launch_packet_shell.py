from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKET_ROOT = PROJECT_ROOT / "launch-packet" / "sam31-breadth-v2"


class LaunchPacketShellTests(unittest.TestCase):
    def test_shell_entrypoints_are_syntactically_valid(self) -> None:
        for path in (
            PACKET_ROOT / "launch.sh",
            PACKET_ROOT / "preflight.sh",
            PACKET_ROOT / "environment" / "bootstrap-mask-env.sh",
        ):
            completed = subprocess.run(
                ["bash", "-n", str(path)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_independent_guard_precedes_environment_bootstrap(self) -> None:
        source = (PACKET_ROOT / "launch.sh").read_text(encoding="utf-8")
        guard = source.index("-m ownership_decoder.remote_pod_guard")
        fatal_handler = source.index("mark_fatal()")
        trap = source.index("trap mark_fatal EXIT")
        system_bootstrap = source.index("apt-get update")
        bootstrap = source.index("bootstrap-mask-env.sh")
        self.assertLess(fatal_handler, trap)
        self.assertLess(trap, system_bootstrap)
        self.assertLess(system_bootstrap, guard)
        self.assertLess(guard, bootstrap)
        self.assertIn('wait "$guard_pid" || true', source[fatal_handler:trap])

    def test_launch_never_installs_or_downloads_models(self) -> None:
        source = (PACKET_ROOT / "launch.sh").read_text(encoding="utf-8")
        forbidden = ("huggingface-cli", "hf download", "git clone", "wget ", "curl ")
        for command in forbidden:
            self.assertNotIn(command, source)

    def test_mig_guard_precedes_environment_and_model_free_preflight(self) -> None:
        source = (PACKET_ROOT / "preflight.sh").read_text(encoding="utf-8")
        guard = source.index("-m ownership_decoder.remote_pod_guard")
        critical_hashes = source.index("\nverify_critical_packet_files\n")
        bootstrap = source.index("bootstrap-mask-env.sh")
        packet_verifier = source.index("verify_mask_packet.py")
        scientific_gate = source.index("run_mask_mig_preflight.py")
        self.assertLess(guard, critical_hashes)
        self.assertLess(critical_hashes, bootstrap)
        self.assertLess(bootstrap, packet_verifier)
        self.assertLess(packet_verifier, scientific_gate)
        self.assertNotIn("build_tracker_only_sam31", source)

    def test_full_launch_requires_packet_bound_mig_sentinel(self) -> None:
        source = (PACKET_ROOT / "launch.sh").read_text(encoding="utf-8")
        sentinel = source.index("MIG_PREFLIGHT_COMPLETE.json")
        packet_binding = source.index('preflight.get("packet_sha256")')
        critical_hashes = source.index("\nverify_critical_packet_files\n")
        bootstrap = source.index("bootstrap-mask-env.sh")
        packet_verifier = source.index("verify_mask_packet.py")
        campaign = source.index("supervise_sam31_breadth.py")
        self.assertLess(sentinel, packet_binding)
        self.assertLess(packet_binding, critical_hashes)
        self.assertLess(critical_hashes, bootstrap)
        self.assertLess(bootstrap, packet_verifier)
        self.assertLess(packet_verifier, campaign)


if __name__ == "__main__":
    unittest.main()
