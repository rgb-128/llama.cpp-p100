#pragma once

#include "common.cuh"

// Pascal (sm_60) native FP16-product / FP32-accumulate grouped GEMM for MUL_MAT_ID.
// Replaces emulated-DP4A MMQ for MoE prefill on GP100 (sm_60), keeping zero CPU sync
// (reuses mmid.cu tile_spec) and supporting per-expert NVFP4 x_scale fusion.
// Returns true if handled; false falls back to standard MMQ.
// Set GGML_CUDA_MMID_F16=0 to disable.
bool ggml_cuda_mmid_f16_p100_try(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * ids,
    ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion);
