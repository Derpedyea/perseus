#!/usr/bin/env python3
"""ggml-vulkan's per-op GPU timer (GGML_VK_PERF_LOGGER) on a CPU+GPU split, for kit step 4b-08.

The timer asserts that no compute context is open when a graph starts. With the routed experts on
the CPU, the scheduler's async copy of a split's inputs (no transfer queue on Polaris) opens one
first, so run 3 aborted after the first graph: GGML_ASSERT(ctx->compute_ctx.expired()). This
submits and waits for that context instead, which only runs when the timer is on.

usage: patch_vulkan.py LLAMA_CPP_DIR   (on a clean copy of ggml-vulkan.cpp; fails if it doesn't fit)
"""
import sys
from pathlib import Path

OLD = """        GGML_ASSERT(ctx->compute_ctx.expired());
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        ctx->query_idx = 0;
"""
NEW = """        // perseus: an async copy of a split's inputs can leave a compute context open; flush it
        if (!ctx->compute_ctx.expired()) {
            ggml_vk_synchronize(ctx);
        }
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        ctx->query_idx = 0;
"""

path = Path(sys.argv[1]) / "ggml/src/ggml-vulkan/ggml-vulkan.cpp"
src = path.read_text()
if src.count(OLD) != 1:
    sys.exit(f"patch_vulkan: expected the perf logger's assert once in {path}, found {src.count(OLD)}")
# ggml_vk_synchronize must be defined above the graph code that now calls it
if src.find("void ggml_vk_synchronize(ggml_backend_vk_context * ctx) {") > src.find(OLD):
    sys.exit("patch_vulkan: ggml_vk_synchronize is defined after the perf logger's code")
path.write_text(src.replace(OLD, NEW))
