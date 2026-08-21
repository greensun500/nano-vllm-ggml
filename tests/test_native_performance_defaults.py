from types import SimpleNamespace

from nanovllm.backends.native.performance import apply_native_performance_defaults


def make_config(**overrides):
    values = {
        "backend": "native_cpu",
        "device_config": {"device_index": 0},
        "enable_mtp": None,
        "mtp_max_draft_tokens": None,
        "native_vulkan_graph_reuse": None,
        "native_performance_profile": "auto",
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_auto_profile_selects_measured_a720_cpu_defaults():
    config = make_config()

    profile = apply_native_performance_defaults(
        config,
        device_records=[],
        machine="aarch64",
        cpu_info="model name : Cortex-A720",
    )

    assert profile == "arm_a720_cpu"
    assert config.enable_mtp is True
    assert config.mtp_max_draft_tokens == 1
    assert config.native_vulkan_graph_reuse is False


def test_auto_profile_keeps_measured_mali_g720_on_the_non_mtp_path():
    config = make_config(backend="native_vulkan")

    profile = apply_native_performance_defaults(
        config,
        device_records=[{"kind": "vulkan", "index": 0, "name": "Mali-G720-Immortalis"}],
        machine="aarch64",
        cpu_info="",
    )

    assert profile == "mali_g720_vulkan"
    assert config.enable_mtp is False
    assert config.mtp_max_draft_tokens == 3
    assert config.native_vulkan_graph_reuse is False


def test_auto_profile_selects_measured_a100_vulkan_defaults():
    config = make_config(backend="native_vulkan")

    profile = apply_native_performance_defaults(
        config,
        device_records=[{"kind": "vulkan", "index": 0, "name": "NVIDIA A100-SXM4-80GB"}],
        machine="x86_64",
        cpu_info="",
    )

    assert profile == "nvidia_a100_vulkan"
    assert config.enable_mtp is True
    assert config.mtp_max_draft_tokens == 3
    assert config.native_vulkan_graph_reuse is True


def test_explicit_values_override_the_auto_profile():
    config = make_config(
        backend="native_vulkan",
        enable_mtp=False,
        mtp_max_draft_tokens=2,
        native_vulkan_graph_reuse=False,
    )

    apply_native_performance_defaults(
        config,
        device_records=[{"kind": "vulkan", "index": 0, "name": "NVIDIA A100-SXM4-80GB"}],
        machine="x86_64",
        cpu_info="",
    )

    assert config.enable_mtp is False
    assert config.mtp_max_draft_tokens == 2
    assert config.native_vulkan_graph_reuse is False


def test_unknown_hardware_uses_the_conservative_baseline():
    config = make_config(backend="native_vulkan")

    profile = apply_native_performance_defaults(
        config,
        device_records=[{"kind": "vulkan", "index": 0, "name": "Unknown Vulkan"}],
        machine="x86_64",
        cpu_info="",
    )

    assert profile == "baseline"
    assert config.enable_mtp is False
    assert config.mtp_max_draft_tokens == 3
    assert config.native_vulkan_graph_reuse is False
