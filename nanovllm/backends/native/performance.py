"""Measured single-sequence defaults for known native-runtime devices.

The policy is intentionally narrow: only the three devices with real-model
benchmark coverage are selected automatically.  Unknown hardware retains the
conservative baseline.  Explicit Config values always win over this module.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Iterable


def _cpuinfo() -> str:
    try:
        return Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _visible_devices() -> list[dict[str, Any]]:
    try:
        from nanovllm.backends.native.runtime import available_backends

        return list(available_backends())
    except (AttributeError, ImportError, RuntimeError):
        # The normal runner construction reports a more useful build error.
        # Keeping the baseline here lets that error originate from the runtime.
        return []


def _vulkan_name(records: Iterable[dict[str, Any]], device_index: int) -> str:
    for record in records:
        if record.get("kind") == "vulkan" and int(record.get("index", -1)) == device_index:
            return str(record.get("name", ""))
    return ""


def _selected_profile(
    backend: str,
    requested: str,
    device_name: str,
    machine: str,
    cpu_info: str,
) -> str:
    if requested != "auto":
        return requested
    name = device_name.lower()
    if backend == "native_vulkan":
        if "mali-g720" in name or "mali g720" in name:
            return "mali_g720_vulkan"
        if "nvidia" in name and "a100" in name:
            return "nvidia_a100_vulkan"
    cpu_description = cpu_info.lower()
    if (
        backend == "native_cpu"
        and machine.lower() in ("aarch64", "arm64")
        and ("cortex-a720" in cpu_description or "cpu part\t: 0xd81" in cpu_description)
    ):
        return "arm_a720_cpu"
    return "baseline"


def apply_native_performance_defaults(
    config: Any,
    *,
    device_records: Iterable[dict[str, Any]] | None = None,
    machine: str | None = None,
    cpu_info: str | None = None,
) -> str:
    """Resolve unset native performance options in place and return the profile.

    This is called before ``Scheduler`` construction, so its final MTP setting
    controls both native cache allocation and speculative scheduling.
    """

    if not str(getattr(config, "backend", "")).startswith("native_"):
        return "baseline"

    # Keep the narrow legacy runner-test/config contract intact.  Real Config
    # instances always carry the profile field; old lightweight adapters are
    # already fully explicit and must not trigger device discovery.
    if not hasattr(config, "native_performance_profile"):
        return "baseline"

    requested = str(getattr(config, "native_performance_profile", "auto"))
    records = list(device_records) if device_records is not None else _visible_devices()
    device_config = getattr(config, "device_config", None) or {}
    device_name = _vulkan_name(records, int(device_config.get("device_index", 0)))
    profile = _selected_profile(
        str(config.backend),
        requested,
        device_name,
        machine if machine is not None else platform.machine(),
        cpu_info if cpu_info is not None else _cpuinfo(),
    )

    defaults = {
        "baseline": (False, 3, False),
        "arm_a720_cpu": (True, 1, False),
        "mali_g720_vulkan": (False, 3, False),
        "nvidia_a100_vulkan": (True, 3, True),
    }
    enable_mtp, draft_tokens, vulkan_reuse = defaults[profile]
    if getattr(config, "enable_mtp", None) is None:
        config.enable_mtp = enable_mtp
    if getattr(config, "mtp_max_draft_tokens", None) is None:
        config.mtp_max_draft_tokens = draft_tokens
    if getattr(config, "native_vulkan_graph_reuse", None) is None:
        config.native_vulkan_graph_reuse = vulkan_reuse
    config.native_effective_performance_profile = profile

    if config.enable_mtp:
        if int(config.mtp_max_draft_tokens) <= 0:
            raise ValueError("mtp_max_draft_tokens must be positive")
        if int(config.max_num_batched_tokens) < int(config.max_num_seqs) * (
            int(config.mtp_max_draft_tokens) + 1
        ):
            raise ValueError("max_num_batched_tokens must fit one MTP verification batch")
    return profile
