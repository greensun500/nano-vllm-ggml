# Vendored llama.cpp Revision

This directory is a source snapshot vendored into nano-vLLM. It is not a Git
submodule, so a normal clone contains the exact GGML sources used by the native
runtime.

- Upstream repository: `https://github.com/ggml-org/llama.cpp.git`
- Upstream base: `91c631b21d6e5d09e9c6659efdf6baeef5a44ddb` (`b9986`)
- Vendored revision: `8e29a9e44f40797b2173b179949780cc98ee7176`

The vendored revision includes nano-vLLM's reviewed Qwen3.5 runtime patches,
including Arm/Mali Vulkan work, the CPU repack compatibility declaration, the
Q6_K vocabulary-head experiment, and direct paged attention changes. Preserve
the upstream license and notices in this directory when updating the snapshot.
