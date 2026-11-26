#include "allreduce.cuh"

#if !defined(GGML_USE_MUSA)

#include "convert.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

// ---------------------------------------------------------------------------
// AllReduce for tensor-parallel inference across two GPUs (CUDA or
// ROCm/HIP).
//
// Provides an in-place sum reduction over matching tensors on two GPUs
// in the same process.  Used by the tensor-split path alongside NCCL;
// targets setups without NVLink/xGMI, where data is exchanged between the
// GPUs by staging it through pinned host memory over PCIe.
//
// Two reduction strategies are selected per call by tensor size:
//
//   * Chunked kernel path (small reductions): a single CUDA kernel both
//     stages data through pinned host memory and performs the local sum.
//     Cross-GPU synchronization happens *inside the kernel* (busy-wait on
//     a host-memory flag), which keeps launch overhead low for the
//     latency-sensitive token-generation case.
//
//   * Copy-engine path (large reductions): the transfer is split into
//     D2H + H2D cudaMemcpyAsync chunks driven by the GPU's copy engine,
//     followed by a small device-side add kernel.  Cross-GPU
//     synchronization happens *outside the kernel*, via CUDA events
//     between streams.  This keeps the compute engine free while large
//     transfers are in flight, which matters for prefill-sized tensors.
//     Reductions larger than the per-call inner cap are processed by an
//     outer chunker that issues sequential inner calls.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-GPU signal mechanism
//
// One int per (slot, rank) pair in pinned host memory.  Each AR call writes a
// strictly increasing token (= the AR call number) into its own arrival int.
// The peer spins until its read of the other's arrival int equals the token
// it expects for this call -- a mismatch means the peer hasn't arrived yet.
// Tokens never repeat over realistic call rates (32-bit int wraps in tens of
// days at thousands of ARs/sec), so arrival ints don't need to be reset
// between calls; we initialize once at pipeline init and let the values
// accumulate.
//
// There is exactly one writer (the owning GPU) and one reader (the peer), so
// we don't need atomics.  A volatile store paired with __threadfence_system()
// provides the release ordering that makes the D2H writes visible system-wide
// before the arrival token is observed.
//
// atomicAdd_system() requires hostNativeAtomicSupported, which is unavailable
// on PCIe-attached consumer GPUs without NVLink, so the volatile path is the
// portable choice.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void ggml_cuda_ar_signal_set(int * p, int token) {
    *(volatile int *)p = token;
}
static __device__ __forceinline__ int ggml_cuda_ar_signal_get(const int * p) {
    return *(const volatile int *)p;
}

static __device__ __forceinline__ void ggml_cuda_ar_signal_set_u32(uint32_t * p, uint32_t token) {
    *(volatile uint32_t *)p = token;
}
static __device__ __forceinline__ uint32_t ggml_cuda_ar_signal_get_u32(const uint32_t * p) {
    return *(const volatile uint32_t *)p;
}

static __device__ __forceinline__ float4 ggml_cuda_ar_ld_cv_f4(const float4 * p) {
    float4 r;
    asm volatile("ld.global.cv.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(p));
    return r;
}

static __device__ __forceinline__ uint4 ggml_cuda_ar_ld_cv_u4(const uint4 * p) {
    uint4 r;
    asm volatile("ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

template <int N, typename T>
struct ggml_cuda_ar_ptr_pack {
    T * ptr[N];
};

template <int N>
static __device__ __forceinline__ void ggml_cuda_ar_pascal_barrier(
        uint32_t *                                my_arr,
        const ggml_cuda_ar_ptr_pack<N, uint32_t> & arr_ptrs,
        int                                       tid,
        int                                       off,
        int                                       my_rank,
        uint32_t                                  token) {
    if (tid == 0) {
        ggml_cuda_ar_signal_set_u32(my_arr + off, token);
        __threadfence_system();
    }
    if (tid < N && tid != my_rank) {
        const uint32_t * peer_arr;
        if constexpr (N == 2) {
            peer_arr = (tid == 0) ? arr_ptrs.ptr[0] : arr_ptrs.ptr[1];
        } else {
            peer_arr = (tid == 0) ? arr_ptrs.ptr[0]
                     : (tid == 1) ? arr_ptrs.ptr[1]
                     : (tid == 2) ? arr_ptrs.ptr[2]
                                  : arr_ptrs.ptr[3];
        }
        peer_arr += off;
        while (ggml_cuda_ar_signal_get_u32(peer_arr) != token) {
            // Pascal has no __nanosleep; parallel per-peer volatile poll on mapped DDIO line.
        }
    }
}

// Fast exact round-to-nearest-even F32 <-> BF16 bit helpers for Pascal (sm_60),
// which has no hardware BF16 instructions.  Zero overflow risk (full FP32 exponent
// range up to 3.4e38) and ~3 ALU ops/elem completely hidden behind PCIe/QPI BW.
static __device__ __forceinline__ uint16_t ggml_cuda_ar_f32_to_bf16_bits(float x) {
    uint32_t u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<uint16_t>((u >> 16) | 0x0040u);
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

static __device__ __forceinline__ float ggml_cuda_ar_bf16_bits_to_f32(uint16_t h) {
    return __uint_as_float(static_cast<uint32_t>(h) << 16);
}

// Byte spacing between adjacent arrival ints.  64 bytes (one cache line)
// ensures each GPU/block's arrival slot lives on its own line, preventing
// false-sharing stalls on the polling GPU.
static constexpr size_t GGML_CUDA_AR_ARRIVAL_STRIDE = 64;

// Number of blocks the chunked kernel launches with.  Each block stripes a
// disjoint slice of the data and synchronizes through its own arrival-token
// slot so multiple SMs can pump PCIe stores in parallel.
static constexpr int GGML_CUDA_AR_KERNEL_BLOCKS = 8;

// Pascal multi-GPU (2x / 4x) fast-path constants:
//  * 4-slot ring: every AR kernel is a full cross-GPU barrier on the compute
//    stream, so by call k+2 every GPU has guaranteed exited call k.  4 slots
//    give 2 full extra barriers of margin -> ZERO CPU cudaEventSynchronize!
//  * Up to 16 blocks for hierarchical BAR-P2P prefill transfers (still only
//    16/56 SMs on GP100 -> 100% guaranteed simultaneous residency).
//  * 4 phases per (slot, rank, block) in the arrival ring so intra-socket and
//    cross-socket barriers live on disjoint cache lines.
static constexpr int    GGML_CUDA_AR_PASCAL_POOL_SIZE    = 4;
static constexpr int    GGML_CUDA_AR_PASCAL_MAX_BLOCKS   = 16;
static constexpr int    GGML_CUDA_AR_PASCAL_PHASES       = 4;
static constexpr size_t GGML_CUDA_AR_PASCAL_HIER_SLICE   = 4 * 1024 * 1024; // 4 MB BF16 half-buffer -> 16 MB F32 chunk!

// ---------------------------------------------------------------------------
// Pascal N-GPU (N=2 or N=4) Single-Kernel Zero-Sync DDIO AllReduce (F32 wire).
//
// Latency-optimal path for small/medium decode and speculative-verify tensors.
//  * Zero CUDA events, zero CPU cudaEventSynchronize, zero cudaMemsetAsync.
//  * Single cross-GPU barrier per call (out-of-place mapped pinned staging).
//  * When count_vec <= gridDim.x * blockDim.x (e.g. <= 32 KB at 8 blocks),
//    keeps own float4 in registers across the barrier (no HBM2 re-read!).
//  * Deterministic rank-order sum 0 + 1 + ... + (N-1) guarantees 100% bit-
//    identical FP32 results across every GPU.
// ---------------------------------------------------------------------------
template <int N>
static __global__ __launch_bounds__(256, 2) void ggml_cuda_ar_pascal_ddio_f32_kernel(
        float * __restrict__               data,
        float * __restrict__               my_host,
        ggml_cuda_ar_ptr_pack<N, float>    host_ptrs,
        uint32_t *                         my_arr,
        ggml_cuda_ar_ptr_pack<N, uint32_t> arr_ptrs,
        int                                count,
        int                                my_rank,
        int                                my_active,
        uint32_t                           token) {
    constexpr int ARR_STEP = static_cast<int>(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(uint32_t));
    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count >> 2;
    const int tail      = count_vec << 2;

    float4 *       data4    = reinterpret_cast<float4 *>(data);
    float4 *       my_host4 = reinterpret_cast<float4 *>(my_host);
    const bool     single   = (count_vec <= gnt);

    float4 cached = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (single) {
        if (gtid < count_vec) {
            if (my_active) {
                cached = data4[gtid];
            }
            my_host4[gtid] = cached;
        }
    } else {
        for (int i = gtid; i < count_vec; i += gnt) {
            const float4 v = my_active ? data4[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            my_host4[i] = v;
        }
    }
    if (bid == 0 && tid < count - tail) {
        my_host[tail + tid] = my_active ? data[tail + tid] : 0.0f;
    }

    __threadfence_system();
    __syncthreads();

    ggml_cuda_ar_pascal_barrier<N>(my_arr, arr_ptrs, tid, bid * ARR_STEP, my_rank, token);

    __syncthreads();
    __threadfence_system();

    if (single) {
        if (gtid < count_vec) {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            #pragma unroll
            for (int r = 0; r < N; ++r) {
                const float4 v = (r == my_rank)
                    ? cached
                    : ggml_cuda_ar_ld_cv_f4(reinterpret_cast<const float4 *>(host_ptrs.ptr[r]) + gtid);
                s0 += v.x;
                s1 += v.y;
                s2 += v.z;
                s3 += v.w;
            }
            data4[gtid] = make_float4(s0, s1, s2, s3);
        }
    } else {
        for (int i = gtid; i < count_vec; i += gnt) {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            #pragma unroll
            for (int r = 0; r < N; ++r) {
                const float4 v = (r == my_rank)
                    ? (my_active ? data4[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f))
                    : ggml_cuda_ar_ld_cv_f4(reinterpret_cast<const float4 *>(host_ptrs.ptr[r]) + i);
                s0 += v.x;
                s1 += v.y;
                s2 += v.z;
                s3 += v.w;
            }
            data4[i] = make_float4(s0, s1, s2, s3);
        }
    }
    if (bid == 0 && tid < count - tail) {
        float s = 0.0f;
        #pragma unroll
        for (int r = 0; r < N; ++r) {
            const float v = (r == my_rank)
                ? (my_active ? data[tail + tid] : 0.0f)
                : host_ptrs.ptr[r][tail + tid];
            s += v;
        }
        data[tail + tid] = s;
    }
}

// ---------------------------------------------------------------------------
// Pascal N-GPU Single-Kernel Zero-Sync DDIO AllReduce (F32 data, BF16 wire).
// Halves PCIe/QPI traffic for medium tensors (or non-P2P large tensors) with
// zero FP16 overflow risk and deterministic rank-order accumulation.
// ---------------------------------------------------------------------------
template <int N>
static __global__ __launch_bounds__(256, 2) void ggml_cuda_ar_pascal_ddio_bf16wire_kernel(
        float * __restrict__               data,
        uint16_t * __restrict__            my_host,
        ggml_cuda_ar_ptr_pack<N, uint16_t> host_ptrs,
        uint32_t *                         my_arr,
        ggml_cuda_ar_ptr_pack<N, uint32_t> arr_ptrs,
        int                                count,
        int                                my_rank,
        int                                my_active,
        uint32_t                           token) {
    constexpr int ARR_STEP = static_cast<int>(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(uint32_t));
    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count >> 3; // 8 floats <-> 16 bytes (uint4) of BF16
    const int tail      = count_vec << 3;

    float4 * data4    = reinterpret_cast<float4 *>(data);
    uint4 *  my_host4 = reinterpret_cast<uint4 *>(my_host);

    for (int i = gtid; i < count_vec; i += gnt) {
        uint4 pack = make_uint4(0u, 0u, 0u, 0u);
        if (my_active) {
            const float4 a = data4[2 * i];
            const float4 b = data4[2 * i + 1];
            pack.x = static_cast<uint32_t>(ggml_cuda_ar_f32_to_bf16_bits(a.x)) |
                    (static_cast<uint32_t>(ggml_cuda_ar_f32_to_bf16_bits(a.y)) << 16);
            pack.y = static_cast<uint32_t>(ggml_cuda_ar_f32_to_bf16_bits(a.z)) |
                    (static_cast<uint32_t>(ggml_cuda_ar_f32_to_bf16_bits(a.w)) << 16);
            pack.z = static_cast<uint32_t>(ggml_cuda_ar_f32_to_bf16_bits(b.x)) |
                    (static_cast<uint32_t>(ggml_cuda_ar_f32_to_bf16_bits(b.y)) << 16);
            pack.w = static_cast<uint32_t>(ggml_cuda_ar_f32_to_bf16_bits(b.z)) |
                    (static_cast<uint32_t>(ggml_cuda_ar_f32_to_bf16_bits(b.w)) << 16);
        }
        my_host4[i] = pack;
    }
    if (bid == 0 && tid < count - tail) {
        my_host[tail + tid] = my_active ? ggml_cuda_ar_f32_to_bf16_bits(data[tail + tid]) : 0u;
    }

    __threadfence_system();
    __syncthreads();

    ggml_cuda_ar_pascal_barrier<N>(my_arr, arr_ptrs, tid, bid * ARR_STEP, my_rank, token);

    __syncthreads();
    __threadfence_system();

    for (int i = gtid; i < count_vec; i += gnt) {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
        #pragma unroll
        for (int r = 0; r < N; ++r) {
            uint4 pack;
            if (r == my_rank) {
                if (my_active) {
                    const float4 a = data4[2 * i];
                    const float4 b = data4[2 * i + 1];
                    pack.x = static_cast<uint32_t>(ggml_cuda_ar_f32_to_bf16_bits(a.x)) |
                            (static_cast<uint32_t>(ggml_cuda_ar_f32_to_bf16_bits(a.y)) << 16);
                    pack.y = static_cast<uint32_t>(ggml_cuda_ar_f32_to_bf16_bits(a.z)) |
                            (static_cast<uint32_t>(ggml_cuda_ar_f32_to_bf16_bits(a.w)) << 16);
                    pack.z = static_cast<uint32_t>(ggml_cuda_ar_f32_to_bf16_bits(b.x)) |
                            (static_cast<uint32_t>(ggml_cuda_ar_f32_to_bf16_bits(b.y)) << 16);
                    pack.w = static_cast<uint32_t>(ggml_cuda_ar_f32_to_bf16_bits(b.z)) |
                            (static_cast<uint32_t>(ggml_cuda_ar_f32_to_bf16_bits(b.w)) << 16);
                } else {
                    pack = make_uint4(0u, 0u, 0u, 0u);
                }
            } else {
                pack = ggml_cuda_ar_ld_cv_u4(reinterpret_cast<const uint4 *>(host_ptrs.ptr[r]) + i);
            }
            s0 += ggml_cuda_ar_bf16_bits_to_f32(static_cast<uint16_t>(pack.x));
            s1 += ggml_cuda_ar_bf16_bits_to_f32(static_cast<uint16_t>(pack.x >> 16));
            s2 += ggml_cuda_ar_bf16_bits_to_f32(static_cast<uint16_t>(pack.y));
            s3 += ggml_cuda_ar_bf16_bits_to_f32(static_cast<uint16_t>(pack.y >> 16));
            s4 += ggml_cuda_ar_bf16_bits_to_f32(static_cast<uint16_t>(pack.z));
            s5 += ggml_cuda_ar_bf16_bits_to_f32(static_cast<uint16_t>(pack.z >> 16));
            s6 += ggml_cuda_ar_bf16_bits_to_f32(static_cast<uint16_t>(pack.w));
            s7 += ggml_cuda_ar_bf16_bits_to_f32(static_cast<uint16_t>(pack.w >> 16));
        }
        data4[2 * i]     = make_float4(s0, s1, s2, s3);
        data4[2 * i + 1] = make_float4(s4, s5, s6, s7);
    }
    if (bid == 0 && tid < count - tail) {
        float s = 0.0f;
        #pragma unroll
        for (int r = 0; r < N; ++r) {
            const uint16_t h = (r == my_rank)
                ? (my_active ? ggml_cuda_ar_f32_to_bf16_bits(data[tail + tid]) : 0u)
                : host_ptrs.ptr[r][tail + tid];
            s += ggml_cuda_ar_bf16_bits_to_f32(h);
        }
        data[tail + tid] = s;
    }
}

// ---------------------------------------------------------------------------
// Pascal N-GPU Single-Kernel Zero-Sync DDIO AllReduce for native F16 / BF16.
// ---------------------------------------------------------------------------
template <int N, typename T>
static __global__ __launch_bounds__(256, 2) void ggml_cuda_ar_pascal_ddio_16_kernel(
        T * __restrict__                   data,
        T * __restrict__                   my_host,
        ggml_cuda_ar_ptr_pack<N, T>        host_ptrs,
        uint32_t *                         my_arr,
        ggml_cuda_ar_ptr_pack<N, uint32_t> arr_ptrs,
        int                                count,
        int                                my_rank,
        int                                my_active,
        uint32_t                           token) {
    constexpr int ARR_STEP = static_cast<int>(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(uint32_t));
    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count >> 3; // 8 x 2B = 16B
    const int tail      = count_vec << 3;

    uint4 * data4    = reinterpret_cast<uint4 *>(data);
    uint4 * my_host4 = reinterpret_cast<uint4 *>(my_host);

    for (int i = gtid; i < count_vec; i += gnt) {
        my_host4[i] = my_active ? data4[i] : make_uint4(0u, 0u, 0u, 0u);
    }
    if (bid == 0 && tid < count - tail) {
        my_host[tail + tid] = my_active ? data[tail + tid] : ggml_cuda_cast<T>(0.0f);
    }

    __threadfence_system();
    __syncthreads();

    ggml_cuda_ar_pascal_barrier<N>(my_arr, arr_ptrs, tid, bid * ARR_STEP, my_rank, token);

    __syncthreads();
    __threadfence_system();

    for (int i = gtid; i < count_vec; i += gnt) {
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int r = 0; r < N; ++r) {
            T vals[8];
            const uint4 raw = (r == my_rank)
                ? (my_active ? data4[i] : make_uint4(0u, 0u, 0u, 0u))
                : ggml_cuda_ar_ld_cv_u4(reinterpret_cast<const uint4 *>(host_ptrs.ptr[r]) + i);
            memcpy(vals, &raw, sizeof(raw));
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                acc[k] += ggml_cuda_cast<float>(vals[k]);
            }
        }
        T out[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            out[k] = ggml_cuda_cast<T>(acc[k]);
        }
        uint4 out4;
        memcpy(&out4, out, sizeof(out4));
        data4[i] = out4;
    }
    if (bid == 0 && tid < count - tail) {
        float s = 0.0f;
        #pragma unroll
        for (int r = 0; r < N; ++r) {
            const T v = (r == my_rank)
                ? (my_active ? data[tail + tid] : ggml_cuda_cast<T>(0.0f))
                : host_ptrs.ptr[r][tail + tid];
            s += ggml_cuda_cast<float>(v);
        }
        data[tail + tid] = ggml_cuda_cast<T>(s);
    }
}

// ---------------------------------------------------------------------------
// Pascal 2-GPU Direct BAR-P2P Read-Only Reduce-Scatter + AllGather (F32).
//
// Exploits GP100 asymmetric PCIe P2P bandwidth: direct BAR1 reads (LDG.E.128)
// sustain ~10.18 GB/s vs ~7.13 GB/s for BAR writes.  Zero host RAM data copies,
// zero FP16 overflow risk, 100% bit-exact FP32, zero CUDA events.
//
// Each GPU owns a disjoint half of the tensor:
//   Phase A: pre-barrier (peer's previous kernel finished writing peer_data).
//   Phase B: read peer's half via LDG.E.128, sum into own half in HBM2.
//   Phase C: mid-barrier (both reduced halves committed in HBM2).
//   Phase D: AllGather other half via LDG.E.128 from peer's HBM2.
//   Phase E: exit-barrier (peer finished reading our half before next kernel).
// ---------------------------------------------------------------------------
static __global__ __launch_bounds__(256, 2) void ggml_cuda_ar_pascal_p2p_2gpu_f32_kernel(
        float * __restrict__       my_data,
        const float * __restrict__ peer_data,
        uint32_t *                 my_arr_base,
        const uint32_t *           peer_arr_base,
        int                        count,
        int                        is_hi,
        int                        my_active,
        int                        peer_active,
        uint32_t                   token) {
    constexpr int ARR_STEP   = static_cast<int>(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(uint32_t));
    constexpr int PHASE_STEP = GGML_CUDA_AR_PASCAL_MAX_BLOCKS * ARR_STEP;
    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count >> 2;
    const int tail      = count_vec << 2;
    const int half_vec  = (count_vec + 1) >> 1;

    const int own_base  = is_hi ? half_vec : 0;
    const int own_n     = is_hi ? (count_vec - half_vec) : half_vec;
    const int peer_base = is_hi ? 0 : half_vec;
    const int peer_n    = is_hi ? half_vec : (count_vec - half_vec);

    auto sync_peer = [&](int phase) {
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            const int off = phase * PHASE_STEP + bid * ARR_STEP;
            ggml_cuda_ar_signal_set_u32(my_arr_base + off, token);
            __threadfence_system();
            const uint32_t * p = peer_arr_base + off;
            while (ggml_cuda_ar_signal_get_u32(p) != token) {}
        }
        __syncthreads();
        __threadfence_system();
    };

    float4 *       my4   = reinterpret_cast<float4 *>(my_data);
    const float4 * peer4 = reinterpret_cast<const float4 *>(peer_data);

    // Phase A: both GPUs finished their prior compute kernel.
    sync_peer(0);

    // Phase B: Reduce-Scatter over direct BAR1 Read (LDG.E.128).
    for (int i = gtid; i < own_n; i += gnt) {
        const int idx = own_base + i;
        const float4 a = my_active   ? my4[idx]   : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4 b = peer_active ? peer4[idx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        my4[idx] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
    if (!is_hi && bid == 0 && tid < count - tail) {
        const float a = my_active   ? my_data[tail + tid]   : 0.0f;
        const float b = peer_active ? peer_data[tail + tid] : 0.0f;
        my_data[tail + tid] = a + b;
    }

    // Phase C: both reduced halves committed to HBM2.
    sync_peer(1);

    // Phase D: AllGather other half via direct BAR1 Read (LDG.E.128).
    for (int i = gtid; i < peer_n; i += gnt) {
        const int idx = peer_base + i;
        my4[idx] = peer4[idx];
    }
    if (is_hi && bid == 0 && tid < count - tail) {
        my_data[tail + tid] = peer_data[tail + tid];
    }

    // Phase E: ensure peer finished reading our half before our next kernel runs.
    sync_peer(2);
}

// ---------------------------------------------------------------------------
// Pascal 4-GPU Hierarchical Dual-Socket BAR-P2P + Mapped-RAM BF16 AllReduce.
//
// Designed for dual-socket PCIe topologies (0<->1 PHB on Socket 0,
// 2<->3 PHB on CPU1, QPI cross-socket with no cross-socket P2P):
//   Phase A: intra-socket pre-barrier (p2p_peer).
//   Phase B: intra-socket Reduce-Scatter via LDG.E.128 BAR Read (@ 10.18 GB/s),
//            storing own local sum in own HBM2 [own_base..own_base+own_n) AND
//            streaming BF16 (2x compressed, 0 overflow!) to host_large[my_rank].
//            (4x less cross-socket QPI traffic than flat 4-GPU F32!)
//   Phase C: cross-socket barrier (cross_peer).
//   Phase D: read cross-socket BF16 half from host_large[cross_peer], sum with
//            own local sum (rounded through BF16 for cross-socket bit-identity),
//            store final F32 to own HBM2 [own_base..own_base+own_n).
//   Phase E: intra-socket post-barrier (p2p_peer).
//   Phase F: intra-socket AllGather other half via LDG.E.128 BAR Read.
//   Phase G: intra-socket exit-barrier (p2p_peer).
//
// Every block bid processes the exact same stripe across all phases -> 100%
// block-independent, zero grid sync, zero CUDA events, STACK:0!
// ---------------------------------------------------------------------------
static __global__ __launch_bounds__(256, 2) void ggml_cuda_ar_pascal_p2p_hier_4gpu_kernel(
        float * __restrict__          my_data,
        const float * __restrict__    p2p_peer_data,
        uint16_t * __restrict__       my_host_bf16,
        const uint16_t * __restrict__ cross_host_bf16,
        uint32_t *                    my_arr_base,
        const uint32_t *              p2p_arr_base,
        const uint32_t *              cross_arr_base,
        int                           count,
        int                           is_hi,
        int                           my_active,
        int                           p2p_active,
        uint32_t                      token) {
    constexpr int ARR_STEP   = static_cast<int>(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(uint32_t));
    constexpr int PHASE_STEP = GGML_CUDA_AR_PASCAL_MAX_BLOCKS * ARR_STEP;
    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count >> 3; // 8 floats = 2 float4s = 1 uint4 BF16
    const int tail      = count_vec << 3;
    const int half_vec  = (count_vec + 1) >> 1;

    const int own_base  = is_hi ? half_vec : 0;
    const int own_n     = is_hi ? (count_vec - half_vec) : half_vec;
    const int peer_base = is_hi ? 0 : half_vec;
    const int peer_n    = is_hi ? half_vec : (count_vec - half_vec);

    auto sync_partner = [&](int phase, const uint32_t * partner_arr_base) {
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            const int off = phase * PHASE_STEP + bid * ARR_STEP;
            ggml_cuda_ar_signal_set_u32(my_arr_base + off, token);
            __threadfence_system();
            const uint32_t * p = partner_arr_base + off;
            while (ggml_cuda_ar_signal_get_u32(p) != token) {}
        }
        __syncthreads();
        __threadfence_system();
    };

    float4 *       my4   = reinterpret_cast<float4 *>(my_data);
    const float4 * peer4 = reinterpret_cast<const float4 *>(p2p_peer_data);
    uint4 *        my_h4 = reinterpret_cast<uint4 *>(my_host_bf16);
    const uint4 *  cr_h4 = reinterpret_cast<const uint4 *>(cross_host_bf16);

    // Phase A: intra-socket pre-barrier.
    sync_partner(0, p2p_arr_base);

    // Phase B: intra-socket Reduce-Scatter (LDG.E.128 BAR Read) -> HBM2 + BF16 mapped RAM.
    for (int i = gtid; i < own_n; i += gnt) {
        const int vidx = 2 * (own_base + i);
        const float4 a0 = my_active  ? my4[vidx]       : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4 a1 = my_active  ? my4[vidx + 1]   : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4 b0 = p2p_active ? peer4[vidx]     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4 b1 = p2p_active ? peer4[vidx + 1] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        const uint16_t h0 = ggml_cuda_ar_f32_to_bf16_bits(a0.x + b0.x);
        const uint16_t h1 = ggml_cuda_ar_f32_to_bf16_bits(a0.y + b0.y);
        const uint16_t h2 = ggml_cuda_ar_f32_to_bf16_bits(a0.z + b0.z);
        const uint16_t h3 = ggml_cuda_ar_f32_to_bf16_bits(a0.w + b0.w);
        const uint16_t h4 = ggml_cuda_ar_f32_to_bf16_bits(a1.x + b1.x);
        const uint16_t h5 = ggml_cuda_ar_f32_to_bf16_bits(a1.y + b1.y);
        const uint16_t h6 = ggml_cuda_ar_f32_to_bf16_bits(a1.z + b1.z);
        const uint16_t h7 = ggml_cuda_ar_f32_to_bf16_bits(a1.w + b1.w);

        my_h4[i] = make_uint4(
            static_cast<uint32_t>(h0) | (static_cast<uint32_t>(h1) << 16),
            static_cast<uint32_t>(h2) | (static_cast<uint32_t>(h3) << 16),
            static_cast<uint32_t>(h4) | (static_cast<uint32_t>(h5) << 16),
            static_cast<uint32_t>(h6) | (static_cast<uint32_t>(h7) << 16));

        // Store BF16-rounded local partial in own disjoint HBM2 half so both
        // sockets compute the exact same floating-point sum in Phase D.
        my4[vidx]     = make_float4(ggml_cuda_ar_bf16_bits_to_f32(h0),
                                    ggml_cuda_ar_bf16_bits_to_f32(h1),
                                    ggml_cuda_ar_bf16_bits_to_f32(h2),
                                    ggml_cuda_ar_bf16_bits_to_f32(h3));
        my4[vidx + 1] = make_float4(ggml_cuda_ar_bf16_bits_to_f32(h4),
                                    ggml_cuda_ar_bf16_bits_to_f32(h5),
                                    ggml_cuda_ar_bf16_bits_to_f32(h6),
                                    ggml_cuda_ar_bf16_bits_to_f32(h7));
    }
    if (!is_hi && bid == 0 && tid < count - tail) {
        const float a = my_active  ? my_data[tail + tid]       : 0.0f;
        const float b = p2p_active ? p2p_peer_data[tail + tid] : 0.0f;
        const uint16_t h = ggml_cuda_ar_f32_to_bf16_bits(a + b);
        my_host_bf16[(own_n << 3) + tid] = h;
        my_data[tail + tid] = ggml_cuda_ar_bf16_bits_to_f32(h);
    }

    // Phase C: cross-socket barrier.
    sync_partner(1, cross_arr_base);

    // Phase D: read cross-socket BF16 half from mapped RAM and accumulate in HBM2.
    for (int i = gtid; i < own_n; i += gnt) {
        const int vidx = 2 * (own_base + i);
        const uint4  cr = cr_h4[i];
        const float4 l0 = my4[vidx];
        const float4 l1 = my4[vidx + 1];

        my4[vidx] = make_float4(
            l0.x + ggml_cuda_ar_bf16_bits_to_f32(static_cast<uint16_t>(cr.x)),
            l0.y + ggml_cuda_ar_bf16_bits_to_f32(static_cast<uint16_t>(cr.x >> 16)),
            l0.z + ggml_cuda_ar_bf16_bits_to_f32(static_cast<uint16_t>(cr.y)),
            l0.w + ggml_cuda_ar_bf16_bits_to_f32(static_cast<uint16_t>(cr.y >> 16)));
        my4[vidx + 1] = make_float4(
            l1.x + ggml_cuda_ar_bf16_bits_to_f32(static_cast<uint16_t>(cr.z)),
            l1.y + ggml_cuda_ar_bf16_bits_to_f32(static_cast<uint16_t>(cr.z >> 16)),
            l1.z + ggml_cuda_ar_bf16_bits_to_f32(static_cast<uint16_t>(cr.w)),
            l1.w + ggml_cuda_ar_bf16_bits_to_f32(static_cast<uint16_t>(cr.w >> 16)));
    }
    if (!is_hi && bid == 0 && tid < count - tail) {
        my_data[tail + tid] += ggml_cuda_ar_bf16_bits_to_f32(cross_host_bf16[(own_n << 3) + tid]);
    }

    // Phase E: intra-socket post-barrier (both reduced halves ready in HBM2).
    sync_partner(2, p2p_arr_base);

    // Phase F: intra-socket AllGather other half via direct BAR1 Read (LDG.E.128).
    for (int i = gtid; i < peer_n; i += gnt) {
        const int vidx = 2 * (peer_base + i);
        my4[vidx]     = peer4[vidx];
        my4[vidx + 1] = peer4[vidx + 1];
    }
    if (is_hi && bid == 0 && tid < count - tail) {
        my_data[tail + tid] = p2p_peer_data[tail + tid];
    }

    // Phase G: intra-socket exit-barrier (ensure peer finished reading our half).
    sync_partner(3, p2p_arr_base);
}

// ---------------------------------------------------------------------------
// Chunked kernel AllReduce -- 2 GPUs, supports float, half, and bfloat16.
//
// Both GPUs run this kernel simultaneously on independent streams.  sendbuf
// and recvbuf live in T_dst (the caller's tensor type); host_mine / host_other
// carry data in T_wire (the on-wire type, possibly narrower than T_dst -- e.g.
// T_dst=F32 with T_wire=BF16 halves the bytes pushed across PCIe).  When
// T_dst == T_wire the casts below are no-ops.
//
// Each GPU runs three phases:
//
//   Phase 1 (all threads): cast sendbuf (T_dst) -> T_wire and store as
//                          single-instruction-width vectors into host_mine.
//                          __threadfence_system() commits these writes to host
//                          memory.
//   Phase 2 (thread 0):    write token to arrival_mine; spin until
//                          arrival_other == token.
//   Phase 3 (all threads): read T_wire vectors from host_other, cast
//                          each element to T_dst, and sum with the local
//                          sendbuf value (also rounded through T_wire so that
//                          both GPUs truncate identically -- this guarantees
//                          bit-equivalent results across the two devices).
//
// Multi-block: blocks stripe vectors across (gridDim.x * blockDim.x) global
// threads to keep multiple SMs issuing PCIe stores in parallel.  Each block
// has its own arrival-token slot (offset by blockIdx.x * ARRIVAL_STRIDE);
// thread 0 of each block signals/spins on that slot independently of other
// blocks.  Tail elements (the leftover < ELEMS_PER_VEC at the end) are
// handled only by block 0 to avoid cross-block writes to the same slots.
// ---------------------------------------------------------------------------
template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_kernel(
        const T_dst  *              sendbuf,
        T_dst        *              recvbuf,
        T_wire       * __restrict__ host_mine,
        const T_wire * __restrict__ host_other,
        int                         count,
        int *                       arrival_mine,
        int *                       arrival_other,
        int                         token) {

    // Vector unit for the wire type, sized to the arch's widest single-instruction
    // copy (16 B on Volta+).  Each phase-1 iter writes one vector to host memory;
    // each phase-3 iter reads one and produces ELEMS_PER_VEC sums.
    constexpr int ELEMS_PER_VEC = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
    constexpr int ARRIVAL_INTS  = (int)(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count / ELEMS_PER_VEC;
    const int tail      = count_vec * ELEMS_PER_VEC;

    // Phase 1: cast sendbuf (T_dst) -> host_mine (T_wire) and store as vectors.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                wire[k] = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
            }
            ggml_cuda_memcpy_1<sizeof(wire)>(&host_mine[off], wire);
        }
        if (bid == 0 && tid < count - tail) {
            host_mine[tail + tid] = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
        }
    }

    // Commit this block's host writes before signalling.
    __threadfence_system();
    __syncthreads();

    // Phase 2: thread 0 of each block signals on its own arrival slot, then
    // spins for the matching slot from peer.  Per-block tokens mean blocks
    // proceed independently -- no inter-block barrier needed.
    if (tid == 0) {
        int       * my_slot    = arrival_mine  + bid * ARRIVAL_INTS;
        const int * other_slot = arrival_other + bid * ARRIVAL_INTS;

        ggml_cuda_ar_signal_set(my_slot, token);
        __threadfence_system(); // make our signal visible system-wide

        while (ggml_cuda_ar_signal_get(other_slot) != token) {
#ifdef GGML_USE_HIP
            // Equals ~100ns at 2500 MHz (sleeps for n * [1,64] clock cycles)
            __builtin_amdgcn_s_sleep(4);
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#endif // GGML_USE_HIP
            // pre-Volta has no __nanosleep: spin on the mapped host flag (one PCIe read per poll)
        }
    }

    __syncthreads();

    // Acquire peer's host_other writes (this block's stripe of them).
    __threadfence_system();

    // Phase 3: read peer's T_wire vector, cast both sides through T_wire for
    // bit-equivalence, sum in T_dst precision, and write back to recvbuf.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            ggml_cuda_memcpy_1<sizeof(wire)>(wire, &host_other[off]);
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
                recvbuf[off + k] = ggml_cuda_cast<T_dst>(
                    ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(wire[k]));
            }
        }
        if (bid == 0 && tid < count - tail) {
            const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
            recvbuf[tail + tid] = ggml_cuda_cast<T_dst>(
                ggml_cuda_cast<float>(d_low) +
                ggml_cuda_cast<float>(host_other[tail + tid]));
        }
    }
}

// Combined load-convert-add kernel.  The peer's contribution arrives as T_src
// (which may be a lower-precision type than T_dst when the BF16 round-trip is
// active).  For bit-equivalence between the two GPUs, dst is first rounded
// through T_src's precision via ggml_cuda_cast -- peer already truncated its
// own value the same way before sending -- so both sides perform identical
// arithmetic.  When T_dst == T_src the round-trip cast is a no-op.
template <typename T_dst, typename T_src>
static __global__ void ggml_cuda_ar_add_kernel(
        T_dst       * __restrict__ dst,
        const T_src * __restrict__ src,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const T_src d_low = ggml_cuda_cast<T_src>(dst[i]);
        dst[i] = ggml_cuda_cast<T_dst>(
            ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(src[i]));
    }
}

// ---------------------------------------------------------------------------
// Pipeline structure
// ---------------------------------------------------------------------------

// Number of slots in the event / arrival ring.  Two slots is sufficient:
// lockstep guarantees the two GPUs are at most one AR (or chunk) apart, so
// slot[N%2] is always safe to reuse -- peer has already consumed slot[N%2]
// from AR N-2 by the time we get to AR N.  acquire_slot's
// cudaEventSynchronize on ev.ker for both devices makes that consumption
// explicit before we overwrite host_buf[slot] for the new AR.
static constexpr int GGML_CUDA_AR_POOL_SIZE = 2;

// Maximum chunk size (bytes per GPU) handled by one chunked kernel launch.
// Larger tensors are reduced by issuing multiple chunked launches.
static constexpr size_t GGML_CUDA_AR_MAX_BYTES = 1024 * 1024; // 1 MB

// Copy-engine path: largest tensor accepted on this path; sets host_large /
// dev_tmp allocation size.
static constexpr size_t GGML_CUDA_AR_COPY_MAX_BYTES = 32 * 1024 * 1024; // 32 MB

// AR wire size at which the copy-engine path takes over from the chunked-
// kernel path.  Override via GGML_CUDA_AR_COPY_THRESHOLD.
static constexpr size_t GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT = 1024 * 1024; // 1 MB
// Per-call CE chunk-size heuristic: chunk_bytes = clamp(nbytes / 4, MIN, MAX).
// The /4 keeps ~4 chunks in flight at any moment (good D2H/H2D overlap with
// the peer); the clamps cover the cases where nbytes/4 is too small (per-
// memcpy fixed cost dominates) or too large (chunk-level pipelining stalls).
// Env var GGML_CUDA_AR_COPY_CHUNK_BYTES can override with a fixed value.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN = 512 * 1024;       // 512 KB
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX = 2 * 1024 * 1024;  // 2 MB
// Absolute floor that an env-var override is allowed to set; this caps the
// per-slot copy-event array.  256 KB -> up to 128 chunks per 32 MB tensor.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN = 256 * 1024;
static constexpr int GGML_CUDA_AR_COPY_MAX_CHUNKS =
    static_cast<int>((GGML_CUDA_AR_COPY_MAX_BYTES + GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN - 1) /
                    GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);

struct ggml_cuda_ar_event_slot {
    cudaEvent_t app = nullptr;  // upstream computation complete
    cudaEvent_t cpy[GGML_CUDA_AR_COPY_MAX_CHUNKS] = {};  // copy-engine D2H chunks complete
    cudaEvent_t h2d = nullptr;  // copy-engine H2Ds complete (handoff AR stream -> compute stream)
    cudaEvent_t ker = nullptr;  // AllReduce kernel complete
};

// Mapped pinned host allocation: cudaHostAlloc + cudaHostGetDevicePointer
// in one place, with the host handle preserved for cudaFreeHost.  Used where
// the CPU never touches the buffer -- only the device reads/writes via the
// mapped device pointer.  Required on systems where cudaDevAttrCanUseHost-
// PointerForRegisteredMem is 0 and the host pointer can't be used as a
// device pointer.
struct ggml_cuda_ar_host_mapping {
    uint8_t * host = nullptr;   // cudaFreeHost handle; also the H-side ptr for cudaMemcpyAsync
    uint8_t * dev  = nullptr;   // device-side pointer for kernels / cudaMemset

    cudaError_t alloc(size_t bytes) {
        cudaError_t rc = cudaHostAlloc(reinterpret_cast<void **>(&host), bytes,
                                       cudaHostAllocPortable | cudaHostAllocMapped);
        if (rc != cudaSuccess) {
            host = nullptr;
            return rc;
        }
        rc = cudaHostGetDevicePointer(reinterpret_cast<void **>(&dev), host, 0);
        if (rc != cudaSuccess) {
            CUDA_CHECK(cudaFreeHost(host));
            host = nullptr;
            dev  = nullptr;
        }
        return rc;
    }

    void free() {
        if (host) {
            CUDA_CHECK(cudaFreeHost(host));
            host = nullptr;
            dev  = nullptr;
        }
    }
};

struct ggml_cuda_ar_pipeline {
    int      n_devices;
    int      devices[GGML_CUDA_MAX_DEVICES];
    size_t   buf_bytes;    // bytes per device in host_buf[]
    size_t   copy_bytes;   // bytes per device in host_large[] / dev_tmp[]
    size_t   copy_threshold;
    size_t   copy_chunk_bytes;
    size_t   bf16_threshold; // tensors >= this size (bytes) are reduced via FP32->BF16 round-trip; 0 disables
    uint64_t call_count;
    bool     pascal;       // pre-Volta fast multi-GPU mode (2x or 4x Pascal sm_60)
    bool     pascal_p2p_pairs; // true when GPUs form bidirectional P2P pairs (0<->1, 2<->3)
    bool     pascal_host_max_set;
    size_t   host_max;     // pascal: optional cap (GGML_CUDA_AR_HOST_MAX)
    size_t   p2p_threshold; // pascal: byte threshold above which BAR-P2P path is used
    size_t   pascal_bf16_threshold; // pascal: byte threshold for BF16 wire on DDIO path (0 = always F32)
    int      pascal_blocks_override; // optional GGML_CUDA_AR_PASCAL_BLOCKS override
    int      p2p_peer[GGML_CUDA_MAX_DEVICES];   // intra-socket P2P partner rank (-1 if none)
    int      cross_peer[GGML_CUDA_MAX_DEVICES]; // cross-socket partner rank for 4-GPU hier (-1 if none)
    int      is_hi_half[GGML_CUDA_MAX_DEVICES]; // 0 or 1 within P2P pair
    uint64_t pascal_call_count;
    uint32_t pascal_token;

    // Per-device resources.
    ggml_cuda_ar_host_mapping host_buf[GGML_CUDA_MAX_DEVICES];   // pinned staging (chunked kernel)
    ggml_cuda_ar_host_mapping host_large[GGML_CUDA_MAX_DEVICES]; // pinned staging (copy-engine or 4-GPU hier BF16)
    char *                    dev_tmp[GGML_CUDA_MAX_DEVICES];    // device scratch for copy-engine path
    cudaStream_t             streams[GGML_CUDA_MAX_DEVICES];   // non-blocking
    ggml_cuda_ar_event_slot  ev_pool[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_POOL_SIZE];

    // Copy-engine: per-device "I finished reading my peer's host_large"
    // event.  Indexed by RECORDER device.  Recorded same-device on streams[i]
    // after stage 2's last H2D from host_large[peer].  Waited cross-device
    // by peer's stage-1 stream before the next AR overwrites host_large[peer].
    cudaEvent_t              host_large_read_done[GGML_CUDA_MAX_DEVICES];
    bool                     host_large_read_done_valid;

    // Copy-engine: per-device "my add_kernel is done with dev_tmp" event.
    // Recorded on the compute stream after each add_kernel; the AR stream
    // waits on it before the next copy_impl's H2D overwrites dev_tmp.  Lets us
    // single-buffer dev_tmp despite add_kernel running on a separate stream.
    cudaEvent_t              dev_tmp_kernel_done[GGML_CUDA_MAX_DEVICES];
    bool                     dev_tmp_kernel_done_valid;

    // Arrival ring: ARRIVAL_STRIDE bytes between adjacent ints.  Mapped pinned
    // memory; CPU never reads/writes -- only the kernel and cudaMemset.
    // Use ggml_cuda_ar_arrival_ptr() / ggml_cuda_ar_pascal_arrival_ptr() to index.
    ggml_cuda_ar_host_mapping arrival;
};

// Base pointer for the (slot, rank) per-block token block.  The kernel adds
// blockIdx.x * (ARRIVAL_STRIDE/sizeof(int)) internally to land on its own slot.
static int * ggml_cuda_ar_arrival_ptr(const ggml_cuda_ar_pipeline * p, int slot, int rank) {
    const size_t offset = ((size_t)slot * p->n_devices + rank) *
                          GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<int *>(p->arrival.dev + offset);
}

static uint32_t * ggml_cuda_ar_pascal_arrival_ptr(const ggml_cuda_ar_pipeline * p, int slot, int rank, int phase = 0) {
    const size_t offset = (((size_t)slot * p->n_devices + rank) * GGML_CUDA_AR_PASCAL_PHASES + phase) *
                          GGML_CUDA_AR_PASCAL_MAX_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<uint32_t *>(p->arrival.dev + offset);
}

static uint32_t ggml_cuda_ar_pascal_next_token(ggml_cuda_ar_pipeline * p, int & out_slot) {
    out_slot = static_cast<int>(p->pascal_call_count % GGML_CUDA_AR_PASCAL_POOL_SIZE);
    p->pascal_call_count++;
    p->pascal_token++;
    if (p->pascal_token == 0) {
        p->pascal_token = 1;
    }
    return p->pascal_token;
}

static int ggml_cuda_ar_pascal_choose_blocks(const ggml_cuda_ar_pipeline * p, size_t chunk_bytes) {
    if (p->pascal_blocks_override > 0) {
        return std::min(p->pascal_blocks_override, GGML_CUDA_AR_PASCAL_MAX_BLOCKS);
    }
    if (chunk_bytes <= 32 * 1024) {
        return 1;
    }
    if (chunk_bytes <= 64 * 1024) {
        return 2;
    }
    if (chunk_bytes <= 128 * 1024) {
        return 4;
    }
    if (chunk_bytes <= 256 * 1024) {
        return 8;
    }
    return GGML_CUDA_AR_PASCAL_MAX_BLOCKS;
}

static uint64_t ggml_cuda_ar_env_u64(const char * name, uint64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t) parsed : default_value;
}

struct ggml_cuda_ar_slot_info {
    int slot;
    int token;
};

static ggml_cuda_ar_slot_info ggml_cuda_ar_acquire_slot(ggml_cuda_ar_pipeline * p) {
    const int  slot        = static_cast<int>(p->call_count % GGML_CUDA_AR_POOL_SIZE);
    const bool pool_lapped = p->call_count >= GGML_CUDA_AR_POOL_SIZE;
    p->call_count++;

    if (pool_lapped) {
        for (int i = 0; i < p->n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventSynchronize(p->ev_pool[i][slot].ker));
        }
    }

    return { slot, (int) p->call_count };
}

// Per-AR copy-engine chunk size: env-var override if set, else heuristic
// (clamp(nbytes/4, HEURISTIC_MIN, HEURISTIC_MAX)).
static size_t ggml_cuda_ar_chunk_bytes(const ggml_cuda_ar_pipeline * p, size_t nbytes) {
    if (p->copy_chunk_bytes > 0) {
        return p->copy_chunk_bytes;
    }
    return std::min(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX,
                    std::max(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN, nbytes / 4));
}

static void ggml_cuda_ar_wait_for_compute(
        ggml_cuda_ar_pipeline * p, ggml_backend_cuda_context * cuda_ctx, int rank, int slot) {
    ggml_cuda_ar_event_slot & ev = p->ev_pool[rank][slot];
    CUDA_CHECK(cudaEventRecord(ev.app, cuda_ctx->stream()));
    CUDA_CHECK(cudaStreamWaitEvent(p->streams[rank], ev.app));
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------

ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices) {

    if (n_devices != 2 && n_devices != 4) {
        GGML_LOG_DEBUG("%s: internal AllReduce supports n_devices=2 or 4 (got %zu); "
                       "falling back\n", __func__, n_devices);
        return nullptr;
    }

    // Detect Pascal (sm_60 / sm_61) multi-GPU configuration with distinct physical GPUs.
    // Supports both 2-GPU and 4-GPU setups (with or without intra-socket P2P pairs).
    bool pascal_fast = true;
    for (size_t i = 0; i < n_devices && pascal_fast; ++i) {
        const int cc = ggml_cuda_info().devices[devices[i]].cc;
        pascal_fast = (cc >= GGML_CUDA_CC_PASCAL && cc < GGML_CUDA_CC_VOLTA);
        for (size_t j = 0; j < i && pascal_fast; ++j) {
            if (ggml_cuda_info().devices[devices[i]].physical_device ==
                ggml_cuda_info().devices[devices[j]].physical_device) {
                pascal_fast = false;
            }
        }
    }
    if (const char * env = getenv("GGML_CUDA_AR_HOST")) {
        pascal_fast = pascal_fast && atoi(env) != 0;
    }

    if (!pascal_fast && n_devices != 2) {
        GGML_LOG_DEBUG("%s: non-Pascal internal AllReduce only supports n_devices=2 (got %zu); "
                       "falling back\n", __func__, n_devices);
        return nullptr;
    }

    for (size_t i = 0; i < n_devices; ++i) {
        const int cc = ggml_cuda_info().devices[devices[i]].cc;
        if (cc < GGML_CUDA_CC_VOLTA && !pascal_fast) {
            GGML_LOG_DEBUG("%s: internal AllReduce requires compute capability >= %d "
                           "(device %d has cc=%d); falling back\n",
                           __func__, GGML_CUDA_CC_VOLTA, devices[i], cc);
            return nullptr;
        }
    }

    auto * p = new ggml_cuda_ar_pipeline{};
    p->n_devices        = (int) n_devices;
    p->copy_bytes       = pascal_fast ? (GGML_CUDA_AR_PASCAL_POOL_SIZE * GGML_CUDA_AR_PASCAL_HIER_SLICE) : GGML_CUDA_AR_COPY_MAX_BYTES;
    p->copy_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_THRESHOLD", GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT);
    // 0 = use the per-call heuristic (default).  Non-zero env value forces a
    // fixed chunk size for diagnostics, with a floor at COPY_CHUNK_BYTES_MIN.
    p->copy_chunk_bytes = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_CHUNK_BYTES", 0);
    if (p->copy_chunk_bytes > 0 && p->copy_chunk_bytes < GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN) {
        GGML_LOG_WARN("%s: GGML_CUDA_AR_COPY_CHUNK_BYTES=%zu below minimum %zu; clamping\n",
                      __func__, p->copy_chunk_bytes, GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);
        p->copy_chunk_bytes = GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN;
    }
    // Default 1 for Volta+: BF16 round-trip is always on for F32 inputs.
    // Pascal: bit-exact FP32 wire below 128 KB (latency-bound decode), BF16 wire above.
    p->bf16_threshold         = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", pascal_fast ? 0 : 1);
    p->pascal                 = pascal_fast;
    p->pascal_host_max_set    = getenv("GGML_CUDA_AR_HOST_MAX") != nullptr;
    p->host_max               = ggml_cuda_ar_env_u64("GGML_CUDA_AR_HOST_MAX", GGML_CUDA_AR_MAX_BYTES);
    p->p2p_threshold          = ggml_cuda_ar_env_u64("GGML_CUDA_AR_P2P_THRESHOLD", 256 * 1024);
    p->pascal_bf16_threshold  = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 128 * 1024);
    p->pascal_blocks_override = (int) ggml_cuda_ar_env_u64("GGML_CUDA_AR_PASCAL_BLOCKS", 0);
    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i]    = devices[i];
        p->p2p_peer[i]   = -1;
        p->cross_peer[i] = -1;
        p->is_hi_half[i] = 0;
    }

    // On Pascal, probe bidirectional P2P topology to discover intra-socket pairs
    // (e.g. (0,1) on CPU0 and (2,3) on CPU1) and enable direct peer access for BAR1 reads.
    if (p->pascal) {
        const bool p2p_allowed = !getenv("GGML_CUDA_AR_P2P") || atoi(getenv("GGML_CUDA_AR_P2P")) != 0;
        auto can_p2p = [&](int r_a, int r_b) -> bool {
            const int pa = ggml_cuda_info().devices[p->devices[r_a]].physical_device;
            const int pb = ggml_cuda_info().devices[p->devices[r_b]].physical_device;
            int c1 = 0, c2 = 0;
            if (cudaDeviceCanAccessPeer(&c1, pa, pb) != cudaSuccess ||
                cudaDeviceCanAccessPeer(&c2, pb, pa) != cudaSuccess) {
                (void) cudaGetLastError();
                return false;
            }
            return c1 != 0 && c2 != 0;
        };

        if (p2p_allowed && n_devices == 2 && can_p2p(0, 1)) {
            p->p2p_peer[0] = 1; p->is_hi_half[0] = 0;
            p->p2p_peer[1] = 0; p->is_hi_half[1] = 1;
            p->pascal_p2p_pairs = true;
        } else if (p2p_allowed && n_devices == 4) {
            int partner0 = -1;
            for (int r = 1; r < 4; ++r) {
                if (can_p2p(0, r)) {
                    partner0 = r;
                    break;
                }
            }
            if (partner0 > 0) {
                int rem[2] = {-1, -1};
                int k = 0;
                for (int r = 1; r < 4; ++r) {
                    if (r != partner0) {
                        rem[k++] = r;
                    }
                }
                if (rem[0] >= 0 && rem[1] >= 0 && can_p2p(rem[0], rem[1])) {
                    // Pair 0: (0, partner0), Pair 1: (rem[0], rem[1]).
                    // Cross-socket partners own the same half: 0 <-> rem[0] (lo), partner0 <-> rem[1] (hi).
                    p->p2p_peer[0]        = partner0; p->cross_peer[0]        = rem[0];   p->is_hi_half[0]        = 0;
                    p->p2p_peer[partner0] = 0;        p->cross_peer[partner0] = rem[1];   p->is_hi_half[partner0] = 1;
                    p->p2p_peer[rem[0]]   = rem[1];   p->cross_peer[rem[0]]   = 0;        p->is_hi_half[rem[0]]   = 0;
                    p->p2p_peer[rem[1]]   = rem[0];   p->cross_peer[rem[1]]   = partner0; p->is_hi_half[rem[1]]   = 1;
                    p->pascal_p2p_pairs = true;
                }
            }
        }

        if (p->pascal_p2p_pairs) {
            for (size_t i = 0; i < n_devices; ++i) {
                const int peer = p->p2p_peer[i];
                const int phys_peer = ggml_cuda_info().devices[p->devices[peer]].physical_device;
                ggml_cuda_set_device(p->devices[i]);
                cudaError_t rc = cudaDeviceEnablePeerAccess(phys_peer, 0);
                if (rc == cudaErrorPeerAccessAlreadyEnabled) {
                    (void) cudaGetLastError();
                } else if (rc != cudaSuccess) {
                    (void) cudaGetLastError();
                    GGML_LOG_WARN("%s: cudaDeviceEnablePeerAccess(%d -> %d) failed (%s); disabling BAR-P2P path\n",
                                  __func__, p->devices[i], p->devices[peer], cudaGetErrorString(rc));
                    p->pascal_p2p_pairs = false;
                    break;
                }
            }
        }
    }

    // Per-device streams and event pools.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);

        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaStreamCreateWithFlags failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        p->streams[i] = stream;

        if (!p->pascal) {
            for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
                bool ok =
                    cudaEventCreateWithFlags(&p->ev_pool[i][s].app, cudaEventDisableTiming) == cudaSuccess &&
                    cudaEventCreateWithFlags(&p->ev_pool[i][s].h2d, cudaEventDisableTiming) == cudaSuccess &&
                    cudaEventCreateWithFlags(&p->ev_pool[i][s].ker, cudaEventDisableTiming) == cudaSuccess;
                for (int c = 0; ok && c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                    ok = cudaEventCreateWithFlags(&p->ev_pool[i][s].cpy[c], cudaEventDisableTiming) == cudaSuccess;
                }
                if (!ok) {
                    GGML_LOG_ERROR("%s: cudaEventCreate failed for device %d slot %d\n",
                                   __func__, p->devices[i], s);
                    ggml_cuda_ar_pipeline_free(p);
                    return nullptr;
                }
            }

            if (cudaEventCreateWithFlags(&p->host_large_read_done[i], cudaEventDisableTiming) != cudaSuccess) {
                GGML_LOG_ERROR("%s: cudaEventCreate for host_large_read_done failed for device %d\n",
                               __func__, p->devices[i]);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
            if (cudaEventCreateWithFlags(&p->dev_tmp_kernel_done[i], cudaEventDisableTiming) != cudaSuccess) {
                GGML_LOG_ERROR("%s: cudaEventCreate for dev_tmp_kernel_done failed for device %d\n",
                               __func__, p->devices[i]);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
        }
    }

    // Arrival ring: cache-line padded so each GPU/block/phase int is on its own line.
    const size_t arrival_bytes = p->pascal
        ? ((size_t) GGML_CUDA_AR_PASCAL_POOL_SIZE * n_devices *
           GGML_CUDA_AR_PASCAL_PHASES * GGML_CUDA_AR_PASCAL_MAX_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE)
        : ((size_t) GGML_CUDA_AR_POOL_SIZE * n_devices *
           GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE);
    if (p->arrival.alloc(arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: alloc for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }
    memset(p->arrival.host, 0, arrival_bytes);

    // Per-device pinned staging buffers.
    p->buf_bytes = GGML_CUDA_AR_MAX_BYTES;
    const size_t pool_depth = p->pascal ? GGML_CUDA_AR_PASCAL_POOL_SIZE : GGML_CUDA_AR_POOL_SIZE;
    const size_t host_buf_total = pool_depth * p->buf_bytes;
    for (size_t i = 0; i < n_devices; ++i) {
        if (p->host_buf[i].alloc(host_buf_total) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for staging failed (%zu bytes)\n",
                           __func__, host_buf_total);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Large staging buffers:
    //  * Volta+: host_large (32 MB) + dev_tmp (32 MB) for copy-engine path.
    //  * Pascal 4-GPU P2P: host_large (4 slots x 4 MB BF16 = 16 MB/GPU) for cross-socket half exchange; zero dev_tmp!
    //  * Pascal 2-GPU P2P or non-P2P: no host_large / dev_tmp needed!
    if (!p->pascal) {
        for (size_t i = 0; i < n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            if (p->host_large[i].alloc(p->copy_bytes) != cudaSuccess) {
                GGML_LOG_ERROR("%s: alloc for large staging failed (%zu bytes)\n",
                               __func__, p->copy_bytes);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
            if (cudaMalloc(reinterpret_cast<void **>(&p->dev_tmp[i]), p->copy_bytes) != cudaSuccess) {
                GGML_LOG_ERROR("%s: cudaMalloc for copy scratch failed (%zu bytes) on device %d\n",
                               __func__, p->copy_bytes, p->devices[i]);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
        }
    } else if (n_devices == 4 && p->pascal_p2p_pairs) {
        for (size_t i = 0; i < n_devices; ++i) {
            if (p->host_large[i].alloc(p->copy_bytes) != cudaSuccess) {
                GGML_LOG_ERROR("%s: alloc for 4-GPU hier staging failed (%zu bytes)\n",
                               __func__, p->copy_bytes);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
        }
    }

    // Eagerly warm up and self-verify every spinning Pascal kernel on all GPUs so
    // CUDA_MODULE_LOADING=LAZY can never serialize or stall a live spin barrier.
    if (p->pascal) {
        float * dummy[GGML_CUDA_MAX_DEVICES] = {};
        bool alloc_ok = true;
        for (size_t i = 0; i < n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            if (cudaMalloc(&dummy[i], 64) != cudaSuccess ||
                cudaMemsetAsync(dummy[i], 0, 64, p->streams[i]) != cudaSuccess) {
                alloc_ok = false;
                break;
            }
        }
        if (alloc_ok) {
            auto run_warmup_ddio = [&]() {
                int slot = 0;
                uint32_t tok = ggml_cuda_ar_pascal_next_token(p, slot);
                if (n_devices == 2) {
                    ggml_cuda_ar_ptr_pack<2, float> h_f32;
                    ggml_cuda_ar_ptr_pack<2, uint16_t> h_bf16;
                    ggml_cuda_ar_ptr_pack<2, half> h_f16;
                    ggml_cuda_ar_ptr_pack<2, nv_bfloat16> h_nb16;
                    ggml_cuda_ar_ptr_pack<2, uint32_t> arr;
                    for (int r = 0; r < 2; ++r) {
                        uint8_t * hb = p->host_buf[r].dev + (size_t) slot * p->buf_bytes;
                        h_f32.ptr[r]  = reinterpret_cast<float *>(hb);
                        h_bf16.ptr[r] = reinterpret_cast<uint16_t *>(hb);
                        h_f16.ptr[r]  = reinterpret_cast<half *>(hb);
                        h_nb16.ptr[r] = reinterpret_cast<nv_bfloat16 *>(hb);
                        arr.ptr[r]    = ggml_cuda_ar_pascal_arrival_ptr(p, slot, r, 0);
                    }
                    for (int r = 0; r < 2; ++r) {
                        ggml_cuda_set_device(p->devices[r]);
                        ggml_cuda_ar_pascal_ddio_f32_kernel<2><<<1, 256, 0, p->streams[r]>>>(dummy[r], h_f32.ptr[r], h_f32, arr.ptr[r], arr, 16, r, 1, tok);
                    }
                    tok = ggml_cuda_ar_pascal_next_token(p, slot);
                    for (int r = 0; r < 2; ++r) {
                        h_bf16.ptr[r] = reinterpret_cast<uint16_t *>(p->host_buf[r].dev + (size_t) slot * p->buf_bytes);
                        arr.ptr[r]    = ggml_cuda_ar_pascal_arrival_ptr(p, slot, r, 0);
                    }
                    for (int r = 0; r < 2; ++r) {
                        ggml_cuda_set_device(p->devices[r]);
                        ggml_cuda_ar_pascal_ddio_bf16wire_kernel<2><<<1, 256, 0, p->streams[r]>>>(dummy[r], h_bf16.ptr[r], h_bf16, arr.ptr[r], arr, 16, r, 1, tok);
                    }
                    tok = ggml_cuda_ar_pascal_next_token(p, slot);
                    for (int r = 0; r < 2; ++r) {
                        h_f16.ptr[r] = reinterpret_cast<half *>(p->host_buf[r].dev + (size_t) slot * p->buf_bytes);
                        arr.ptr[r]   = ggml_cuda_ar_pascal_arrival_ptr(p, slot, r, 0);
                    }
                    for (int r = 0; r < 2; ++r) {
                        ggml_cuda_set_device(p->devices[r]);
                        ggml_cuda_ar_pascal_ddio_16_kernel<2, half><<<1, 256, 0, p->streams[r]>>>(reinterpret_cast<half *>(dummy[r]), h_f16.ptr[r], h_f16, arr.ptr[r], arr, 16, r, 1, tok);
                    }
                    tok = ggml_cuda_ar_pascal_next_token(p, slot);
                    for (int r = 0; r < 2; ++r) {
                        h_nb16.ptr[r] = reinterpret_cast<nv_bfloat16 *>(p->host_buf[r].dev + (size_t) slot * p->buf_bytes);
                        arr.ptr[r]    = ggml_cuda_ar_pascal_arrival_ptr(p, slot, r, 0);
                    }
                    for (int r = 0; r < 2; ++r) {
                        ggml_cuda_set_device(p->devices[r]);
                        ggml_cuda_ar_pascal_ddio_16_kernel<2, nv_bfloat16><<<1, 256, 0, p->streams[r]>>>(reinterpret_cast<nv_bfloat16 *>(dummy[r]), h_nb16.ptr[r], h_nb16, arr.ptr[r], arr, 16, r, 1, tok);
                    }
                    if (p->pascal_p2p_pairs) {
                        tok = ggml_cuda_ar_pascal_next_token(p, slot);
                        for (int r = 0; r < 2; ++r) {
                            const int peer = p->p2p_peer[r];
                            ggml_cuda_set_device(p->devices[r]);
                            ggml_cuda_ar_pascal_p2p_2gpu_f32_kernel<<<1, 256, 0, p->streams[r]>>>(
                                dummy[r], dummy[peer],
                                ggml_cuda_ar_pascal_arrival_ptr(p, slot, r, 0),
                                ggml_cuda_ar_pascal_arrival_ptr(p, slot, peer, 0),
                                16, p->is_hi_half[r], 1, 1, tok);
                        }
                    }
                } else {
                    ggml_cuda_ar_ptr_pack<4, float> h_f32;
                    ggml_cuda_ar_ptr_pack<4, uint16_t> h_bf16;
                    ggml_cuda_ar_ptr_pack<4, half> h_f16;
                    ggml_cuda_ar_ptr_pack<4, nv_bfloat16> h_nb16;
                    ggml_cuda_ar_ptr_pack<4, uint32_t> arr;
                    for (int r = 0; r < 4; ++r) {
                        uint8_t * hb = p->host_buf[r].dev + (size_t) slot * p->buf_bytes;
                        h_f32.ptr[r]  = reinterpret_cast<float *>(hb);
                        arr.ptr[r]    = ggml_cuda_ar_pascal_arrival_ptr(p, slot, r, 0);
                    }
                    for (int r = 0; r < 4; ++r) {
                        ggml_cuda_set_device(p->devices[r]);
                        ggml_cuda_ar_pascal_ddio_f32_kernel<4><<<1, 256, 0, p->streams[r]>>>(dummy[r], h_f32.ptr[r], h_f32, arr.ptr[r], arr, 16, r, 1, tok);
                    }
                    tok = ggml_cuda_ar_pascal_next_token(p, slot);
                    for (int r = 0; r < 4; ++r) {
                        h_bf16.ptr[r] = reinterpret_cast<uint16_t *>(p->host_buf[r].dev + (size_t) slot * p->buf_bytes);
                        arr.ptr[r]    = ggml_cuda_ar_pascal_arrival_ptr(p, slot, r, 0);
                    }
                    for (int r = 0; r < 4; ++r) {
                        ggml_cuda_set_device(p->devices[r]);
                        ggml_cuda_ar_pascal_ddio_bf16wire_kernel<4><<<1, 256, 0, p->streams[r]>>>(dummy[r], h_bf16.ptr[r], h_bf16, arr.ptr[r], arr, 16, r, 1, tok);
                    }
                    tok = ggml_cuda_ar_pascal_next_token(p, slot);
                    for (int r = 0; r < 4; ++r) {
                        h_f16.ptr[r] = reinterpret_cast<half *>(p->host_buf[r].dev + (size_t) slot * p->buf_bytes);
                        arr.ptr[r]   = ggml_cuda_ar_pascal_arrival_ptr(p, slot, r, 0);
                    }
                    for (int r = 0; r < 4; ++r) {
                        ggml_cuda_set_device(p->devices[r]);
                        ggml_cuda_ar_pascal_ddio_16_kernel<4, half><<<1, 256, 0, p->streams[r]>>>(reinterpret_cast<half *>(dummy[r]), h_f16.ptr[r], h_f16, arr.ptr[r], arr, 16, r, 1, tok);
                    }
                    tok = ggml_cuda_ar_pascal_next_token(p, slot);
                    for (int r = 0; r < 4; ++r) {
                        h_nb16.ptr[r] = reinterpret_cast<nv_bfloat16 *>(p->host_buf[r].dev + (size_t) slot * p->buf_bytes);
                        arr.ptr[r]    = ggml_cuda_ar_pascal_arrival_ptr(p, slot, r, 0);
                    }
                    for (int r = 0; r < 4; ++r) {
                        ggml_cuda_set_device(p->devices[r]);
                        ggml_cuda_ar_pascal_ddio_16_kernel<4, nv_bfloat16><<<1, 256, 0, p->streams[r]>>>(reinterpret_cast<nv_bfloat16 *>(dummy[r]), h_nb16.ptr[r], h_nb16, arr.ptr[r], arr, 16, r, 1, tok);
                    }
                    if (p->pascal_p2p_pairs) {
                        tok = ggml_cuda_ar_pascal_next_token(p, slot);
                        for (int r = 0; r < 4; ++r) {
                            const int peer  = p->p2p_peer[r];
                            const int cross = p->cross_peer[r];
                            ggml_cuda_set_device(p->devices[r]);
                            ggml_cuda_ar_pascal_p2p_hier_4gpu_kernel<<<1, 256, 0, p->streams[r]>>>(
                                dummy[r], dummy[peer],
                                reinterpret_cast<uint16_t *>(p->host_large[r].dev     + (size_t) slot * GGML_CUDA_AR_PASCAL_HIER_SLICE),
                                reinterpret_cast<const uint16_t *>(p->host_large[cross].dev + (size_t) slot * GGML_CUDA_AR_PASCAL_HIER_SLICE),
                                ggml_cuda_ar_pascal_arrival_ptr(p, slot, r, 0),
                                ggml_cuda_ar_pascal_arrival_ptr(p, slot, peer, 0),
                                ggml_cuda_ar_pascal_arrival_ptr(p, slot, cross, 0),
                                16, p->is_hi_half[r], 1, 1, tok);
                        }
                    }
                }
            };
            run_warmup_ddio();
            for (size_t i = 0; i < n_devices; ++i) {
                ggml_cuda_set_device(p->devices[i]);
                CUDA_CHECK(cudaStreamSynchronize(p->streams[i]));
            }
        }
        for (size_t i = 0; i < n_devices; ++i) {
            if (dummy[i]) {
                ggml_cuda_set_device(p->devices[i]);
                CUDA_CHECK(cudaFree(dummy[i]));
            }
        }
    }

    if (p->pascal) {
        GGML_LOG_INFO("%s: initialized Pascal zero-sync AllReduce pipeline: %zu GPUs, "
                      "DDIO staging %zu KB x %d slots, BAR-P2P pairs=%s (p2p_thresh=%zu KB, bf16_thresh=%zu KB)\n",
                      __func__, n_devices, p->buf_bytes >> 10, GGML_CUDA_AR_PASCAL_POOL_SIZE,
                      p->pascal_p2p_pairs ? "ON" : "OFF",
                      p->p2p_threshold >> 10, p->pascal_bf16_threshold >> 10);
    } else {
        GGML_LOG_INFO("%s: initialized AllReduce pipeline: %zu GPUs, "
                      "%zu KB chunked kernel staging + %zu MB copy-engine staging per GPU\n",
                      __func__, n_devices, p->buf_bytes >> 10, p->copy_bytes >> 20);
    }

    return p;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * p) {
    if (!p) {
        return;
    }

    // Drain all in-flight kernels before tearing down resources.
    for (int i = 0; i < p->n_devices; ++i) {
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaStreamSynchronize(p->streams[i]));
        }
    }

    for (int i = 0; i < p->n_devices; ++i) {
        p->host_buf[i].free();
        p->host_large[i].free();
        if (p->dev_tmp[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaFree(p->dev_tmp[i]));
        }
        ggml_cuda_set_device(p->devices[i]);
        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            if (p->ev_pool[i][s].app) { CUDA_CHECK(cudaEventDestroy(p->ev_pool[i][s].app)); }
            for (int c = 0; c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                if (p->ev_pool[i][s].cpy[c]) { CUDA_CHECK(cudaEventDestroy(p->ev_pool[i][s].cpy[c])); }
            }
            if (p->ev_pool[i][s].h2d) { CUDA_CHECK(cudaEventDestroy(p->ev_pool[i][s].h2d)); }
            if (p->ev_pool[i][s].ker) { CUDA_CHECK(cudaEventDestroy(p->ev_pool[i][s].ker)); }
        }
        if (p->host_large_read_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventDestroy(p->host_large_read_done[i]));
        }
        if (p->dev_tmp_kernel_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventDestroy(p->dev_tmp_kernel_done[i]));
        }
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaStreamDestroy(p->streams[i]));
        }
    }
    p->arrival.free();
    delete p;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

// Asymmetric copy_impl: data sent over PCIe in T_src precision (one element of
// nbytes per ne element); accumulated locally into a T_dst buffer.  When
// T_src == T_dst this is the original homogeneous reduction.  When they differ
// (e.g. BF16 wire / F32 accumulator) the add kernel rounds dst through T_src
// for bit-equivalence between GPUs and we skip the otherwise-needed
// post-conversion entirely.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_impl(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne,
        size_t                  nbytes) {
    GGML_ASSERT(p->n_devices == 2);
    GGML_ASSERT(nbytes <= p->copy_bytes);
    GGML_ASSERT(ne <= std::numeric_limits<int>::max());

    const size_t chunk_bytes = ggml_cuda_ar_chunk_bytes(p, nbytes);
    GGML_ASSERT(chunk_bytes > 0);

    const int slot = ggml_cuda_ar_acquire_slot(p).slot;
    const size_t copy_chunks = (nbytes + chunk_bytes - 1) / chunk_bytes;
    GGML_ASSERT(copy_chunks <= GGML_CUDA_AR_COPY_MAX_CHUNKS);

    ggml_backend_cuda_context * cuda_ctx[2] = {};

    // Stage 1: both GPUs copy their local contribution to pinned host memory.
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);

        ggml_cuda_ar_wait_for_compute(p, cuda_ctx[i], i, slot);

        // Wait for peer's H2D from our host_large[i] (recorded in the
        // previous AR's stage 2) to complete before we overwrite host_large[i].
        // host_large_read_done[peer] = peer finished reading host_large[i].
        // No-op on the first AR -- no prior record exists.
        if (p->host_large_read_done_valid) {
            const int peer = 1 - i;
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->host_large_read_done[peer]));
        }

        if (!compute[i]) {
            CUDA_CHECK(cudaMemsetAsync(src_buf[i], 0, nbytes, p->streams[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaMemcpyAsync(
                p->host_large[i].host + offset, reinterpret_cast<char *>(src_buf[i]) + offset, this_bytes,
                cudaMemcpyDeviceToHost, p->streams[i]));
            CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].cpy[c], p->streams[i]));
        }
    }

    // Stage 2: each GPU waits for each peer D2H chunk, pulls that chunk back to
    // local device scratch (dev_tmp), then performs one device-local add over
    // the assembled peer tensor.  The H2Ds run on the AR stream (copy engine)
    // and the add_kernel runs on the caller's compute stream, so the AR stream
    // stays pure-copy and avoids an in-stream copy->compute engine switch every
    // AR.  dev_tmp is single-buffered: the AR stream waits cross-stream on the
    // prior AR's add_kernel-done event before overwriting it.
    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);

        // Wait for the previous AR's add_kernel (on the compute stream) to
        // finish reading dev_tmp before our H2D overwrites it.  No-op on the
        // first copy_impl call.
        if (p->dev_tmp_kernel_done_valid) {
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->dev_tmp_kernel_done[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->ev_pool[peer][slot].cpy[c]));
            CUDA_CHECK(cudaMemcpyAsync(
                p->dev_tmp[i] + offset, p->host_large[peer].host + offset, this_bytes,
                cudaMemcpyHostToDevice, p->streams[i]));
        }

        // Mark our reads of host_large[peer] complete so peer's next AR can
        // safely overwrite it.
        CUDA_CHECK(cudaEventRecord(p->host_large_read_done[i], p->streams[i]));

        // Hand off from AR stream (copy engine) to compute stream: compute
        // stream waits for all H2Ds to finish, then runs the add_kernel.
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].h2d, p->streams[i]));
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->ev_pool[i][slot].h2d));

        const int block_size = 256;
        int n_blocks = (int) ((ne + block_size - 1) / block_size);
        if (n_blocks > 1024) {
            n_blocks = 1024;
        }
        ggml_cuda_ar_add_kernel<T_dst, T_src><<<n_blocks, block_size, 0, cuda_ctx[i]->stream()>>>(
            dst_buf[i],
            reinterpret_cast<const T_src *>(p->dev_tmp[i]),
            (int) ne);
        CUDA_CHECK(cudaGetLastError());

        // Record dev_tmp-released on the compute stream so the next copy_impl
        // can wait for the kernel to finish before overwriting dev_tmp.  Also
        // record AR-done as ev.ker for acquire_slot's pool-wraparound sync.
        CUDA_CHECK(cudaEventRecord(p->dev_tmp_kernel_done[i], cuda_ctx[i]->stream()));
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, cuda_ctx[i]->stream()));
    }
    p->host_large_read_done_valid = true;
    p->dev_tmp_kernel_done_valid = true;

    return true;
}

// Outer-level chunker: copy_impl handles up to copy_bytes per call (limited by
// the host_large / dev_tmp allocation size).  When the full AR exceeds that,
// slice the tensor into copy_bytes-sized pieces and call copy_impl repeatedly.
// Each slice goes through its own stage 1 -> stage 2 cycle and acquires its own
// slot, so cross-AR fences and pool wraparound work the same way as for any
// other sequence of small ARs.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_outer(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne) {
    const int64_t outer_max_elems = (int64_t) (p->copy_bytes / sizeof(T_src));
    GGML_ASSERT(outer_max_elems > 0);

    bool ok = true;
    for (int64_t outer_start = 0; outer_start < ne && ok; outer_start += outer_max_elems) {
        const int64_t outer_ne     = std::min(outer_max_elems, ne - outer_start);
        const size_t  outer_nbytes = (size_t) outer_ne * sizeof(T_src);

        T_src * src[GGML_CUDA_MAX_DEVICES] = {};
        T_dst * dst[GGML_CUDA_MAX_DEVICES] = {};
        for (int i = 0; i < p->n_devices; ++i) {
            src[i] = src_buf[i] + outer_start;
            dst[i] = dst_buf[i] + outer_start;
        }
        ok = ggml_cuda_ar_allreduce_copy_impl<T_src, T_dst>(
            p, backends, src, dst, compute, outer_ne, outer_nbytes);
    }
    return ok;
}

template <int N>
static bool ggml_cuda_ar_pascal_dispatch_n(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors,
        ggml_type               input_type,
        int64_t                 ne,
        size_t                  input_nbytes,
        const bool              compute_flag[GGML_CUDA_MAX_DEVICES]) {
    cudaStream_t streams[N];
    for (int r = 0; r < N; ++r) {
        auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[r]->context);
        GGML_ASSERT(cuda_ctx->device == p->devices[r]);
        streams[r] = cuda_ctx->stream();
    }

    // Large F32 tensors on P2P-paired Pascal GPUs: use direct BAR1-Read P2P path!
    if (input_type == GGML_TYPE_F32 && p->pascal_p2p_pairs && input_nbytes > p->p2p_threshold) {
        if constexpr (N == 2) {
            constexpr int64_t MAX_ELEMS = (1LL << 28); // 1 GB per launch
            for (int64_t off = 0; off < ne; off += MAX_ELEMS) {
                const int chunk_elems = static_cast<int>(std::min(MAX_ELEMS, ne - off));
                int slot = 0;
                const uint32_t tok = ggml_cuda_ar_pascal_next_token(p, slot);
                const int nb = p->pascal_blocks_override > 0
                    ? std::min(p->pascal_blocks_override, GGML_CUDA_AR_PASCAL_MAX_BLOCKS)
                    : GGML_CUDA_AR_PASCAL_MAX_BLOCKS;
                for (int r = 0; r < 2; ++r) {
                    const int peer = p->p2p_peer[r];
                    ggml_cuda_set_device(p->devices[r]);
                    ggml_cuda_ar_pascal_p2p_2gpu_f32_kernel<<<nb, 256, 0, streams[r]>>>(
                        static_cast<float *>(tensors[r]->data) + off,
                        static_cast<const float *>(tensors[peer]->data) + off,
                        ggml_cuda_ar_pascal_arrival_ptr(p, slot, r, 0),
                        ggml_cuda_ar_pascal_arrival_ptr(p, slot, peer, 0),
                        chunk_elems,
                        p->is_hi_half[r],
                        compute_flag[r] ? 1 : 0,
                        compute_flag[peer] ? 1 : 0,
                        tok);
                    CUDA_CHECK(cudaGetLastError());
                }
            }
            return true;
        } else if constexpr (N == 4) {
            // Each GPU writes half the chunk in BF16 (2 bytes/elem) -> 1 byte per total elem,
            // so 4 MB slice holds ~4M floats (~16 MB F32) per single launch!
            constexpr int64_t MAX_ELEMS = ((static_cast<int64_t>(GGML_CUDA_AR_PASCAL_HIER_SLICE) - 64) & ~15LL);
            for (int64_t off = 0; off < ne; off += MAX_ELEMS) {
                const int chunk_elems = static_cast<int>(std::min(MAX_ELEMS, ne - off));
                int slot = 0;
                const uint32_t tok = ggml_cuda_ar_pascal_next_token(p, slot);
                const int nb = p->pascal_blocks_override > 0
                    ? std::min(p->pascal_blocks_override, GGML_CUDA_AR_PASCAL_MAX_BLOCKS)
                    : GGML_CUDA_AR_PASCAL_MAX_BLOCKS;
                for (int r = 0; r < 4; ++r) {
                    const int peer  = p->p2p_peer[r];
                    const int cross = p->cross_peer[r];
                    ggml_cuda_set_device(p->devices[r]);
                    ggml_cuda_ar_pascal_p2p_hier_4gpu_kernel<<<nb, 256, 0, streams[r]>>>(
                        static_cast<float *>(tensors[r]->data) + off,
                        static_cast<const float *>(tensors[peer]->data) + off,
                        reinterpret_cast<uint16_t *>(p->host_large[r].dev     + (size_t) slot * GGML_CUDA_AR_PASCAL_HIER_SLICE),
                        reinterpret_cast<const uint16_t *>(p->host_large[cross].dev + (size_t) slot * GGML_CUDA_AR_PASCAL_HIER_SLICE),
                        ggml_cuda_ar_pascal_arrival_ptr(p, slot, r, 0),
                        ggml_cuda_ar_pascal_arrival_ptr(p, slot, peer, 0),
                        ggml_cuda_ar_pascal_arrival_ptr(p, slot, cross, 0),
                        chunk_elems,
                        p->is_hi_half[r],
                        compute_flag[r] ? 1 : 0,
                        compute_flag[peer] ? 1 : 0,
                        tok);
                    CUDA_CHECK(cudaGetLastError());
                }
            }
            return true;
        }
    }

    // Small/medium latency-bound tensors (or non-P2P fallback): zero-sync DDIO path.
    const bool use_bf16_wire = (input_type == GGML_TYPE_F32) &&
                               (p->pascal_bf16_threshold > 0) &&
                               (input_nbytes >= p->pascal_bf16_threshold);
    const size_t wire_elem_bytes = use_bf16_wire ? sizeof(uint16_t) : ggml_type_size(input_type);
    const int64_t max_chunk_elems = static_cast<int64_t>((p->buf_bytes / wire_elem_bytes) & ~15ULL);
    const size_t input_elem_bytes = ggml_type_size(input_type);

    for (int64_t off = 0; off < ne; off += max_chunk_elems) {
        const int chunk_elems = static_cast<int>(std::min(max_chunk_elems, ne - off));
        const size_t chunk_wire_bytes = (size_t) chunk_elems * wire_elem_bytes;
        const int nb = ggml_cuda_ar_pascal_choose_blocks(p, chunk_wire_bytes);

        int slot = 0;
        const uint32_t tok = ggml_cuda_ar_pascal_next_token(p, slot);

        ggml_cuda_ar_ptr_pack<N, uint32_t> arr;
        for (int r = 0; r < N; ++r) {
            arr.ptr[r] = ggml_cuda_ar_pascal_arrival_ptr(p, slot, r, 0);
        }

        if (input_type == GGML_TYPE_F32 && !use_bf16_wire) {
            ggml_cuda_ar_ptr_pack<N, float> h_ptrs;
            for (int r = 0; r < N; ++r) {
                h_ptrs.ptr[r] = reinterpret_cast<float *>(p->host_buf[r].dev + (size_t) slot * p->buf_bytes);
            }
            for (int r = 0; r < N; ++r) {
                ggml_cuda_set_device(p->devices[r]);
                float * data = static_cast<float *>(tensors[r]->data) + off;
                ggml_cuda_ar_pascal_ddio_f32_kernel<N><<<nb, 256, 0, streams[r]>>>(
                    data, h_ptrs.ptr[r], h_ptrs, arr.ptr[r], arr, chunk_elems, r, compute_flag[r] ? 1 : 0, tok);
                CUDA_CHECK(cudaGetLastError());
            }
        } else if (input_type == GGML_TYPE_F32 && use_bf16_wire) {
            ggml_cuda_ar_ptr_pack<N, uint16_t> h_ptrs;
            for (int r = 0; r < N; ++r) {
                h_ptrs.ptr[r] = reinterpret_cast<uint16_t *>(p->host_buf[r].dev + (size_t) slot * p->buf_bytes);
            }
            for (int r = 0; r < N; ++r) {
                ggml_cuda_set_device(p->devices[r]);
                float * data = static_cast<float *>(tensors[r]->data) + off;
                ggml_cuda_ar_pascal_ddio_bf16wire_kernel<N><<<nb, 256, 0, streams[r]>>>(
                    data, h_ptrs.ptr[r], h_ptrs, arr.ptr[r], arr, chunk_elems, r, compute_flag[r] ? 1 : 0, tok);
                CUDA_CHECK(cudaGetLastError());
            }
        } else if (input_type == GGML_TYPE_F16) {
            ggml_cuda_ar_ptr_pack<N, half> h_ptrs;
            for (int r = 0; r < N; ++r) {
                h_ptrs.ptr[r] = reinterpret_cast<half *>(p->host_buf[r].dev + (size_t) slot * p->buf_bytes);
            }
            for (int r = 0; r < N; ++r) {
                ggml_cuda_set_device(p->devices[r]);
                half * data = reinterpret_cast<half *>(static_cast<char *>(tensors[r]->data) + off * (int64_t) input_elem_bytes);
                ggml_cuda_ar_pascal_ddio_16_kernel<N, half><<<nb, 256, 0, streams[r]>>>(
                    data, h_ptrs.ptr[r], h_ptrs, arr.ptr[r], arr, chunk_elems, r, compute_flag[r] ? 1 : 0, tok);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            ggml_cuda_ar_ptr_pack<N, nv_bfloat16> h_ptrs;
            for (int r = 0; r < N; ++r) {
                h_ptrs.ptr[r] = reinterpret_cast<nv_bfloat16 *>(p->host_buf[r].dev + (size_t) slot * p->buf_bytes);
            }
            for (int r = 0; r < N; ++r) {
                ggml_cuda_set_device(p->devices[r]);
                nv_bfloat16 * data = reinterpret_cast<nv_bfloat16 *>(static_cast<char *>(tensors[r]->data) + off * (int64_t) input_elem_bytes);
                ggml_cuda_ar_pascal_ddio_16_kernel<N, nv_bfloat16><<<nb, 256, 0, streams[r]>>>(
                    data, h_ptrs.ptr[r], h_ptrs, arr.ptr[r], arr, chunk_elems, r, compute_flag[r] ? 1 : 0, tok);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }
    return true;
}

bool ggml_cuda_ar_allreduce(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors) {
    GGML_ASSERT(p != nullptr);

    const int n = p->n_devices;
    GGML_ASSERT(n == 2 || (p->pascal && n == 4));

    const ggml_type input_type = tensors[0]->type;
    GGML_ASSERT(input_type == GGML_TYPE_F32 || input_type == GGML_TYPE_F16 || input_type == GGML_TYPE_BF16);

    const int64_t ne = ggml_nelements(tensors[0]);
    GGML_ASSERT(ne > 0);

    const size_t   input_nbytes = ggml_nbytes(tensors[0]);

    bool compute_flag[GGML_CUDA_MAX_DEVICES] = {};
    for (int i = 0; i < n; ++i) {
        compute_flag[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    }

    if (p->pascal) {
        if (p->pascal_host_max_set && !p->pascal_p2p_pairs && input_nbytes > p->host_max) {
            return false;
        }
        if (n == 2) {
            return ggml_cuda_ar_pascal_dispatch_n<2>(p, backends, tensors, input_type, ne, input_nbytes, compute_flag);
        }
        return ggml_cuda_ar_pascal_dispatch_n<4>(p, backends, tensors, input_type, ne, input_nbytes, compute_flag);
    }

    // BF16 round-trip: F32 inputs >= bf16_threshold are converted to BF16 for
    // the reduction (chunked or copy-engine), halving on-wire bytes. Matches
    // NCCL's behaviour. The pre-conversion zeroes inactive shards so the
    // inner paths see them as already-prepared compute tensors.
    const bool use_bf16 =
        input_type == GGML_TYPE_F32 &&
        p->bf16_threshold > 0 &&
        input_nbytes >= p->bf16_threshold;

    const ggml_type kernel_type = use_bf16 ? GGML_TYPE_BF16 : input_type;
    const size_t    type_size   = ggml_type_size(kernel_type);
    GGML_ASSERT(p->buf_bytes >= type_size);
    const size_t    nbytes      = (size_t) ne * type_size;

    // Decide between copy-engine and chunked kernel paths based on the working
    // type's actual byte count.  No upper bound: copy_outer slices reductions
    // larger than copy_bytes into copy_bytes-sized pieces.
    const bool use_copy_engine =
        p->copy_threshold > 0 &&
        nbytes >= p->copy_threshold;

    // BF16 inactive-shard zeroing: when use_bf16 is on, the combined kernel
    // (chunked kernel path) and the combined add kernel (copy_engine path)
    // both accumulate into the F32 tensor data directly, so an inactive
    // shard's accumulator must start at zero.
    if (use_bf16) {
        for (int i = 0; i < n; ++i) {
            if (!compute_flag[i]) {
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                ggml_cuda_set_device(p->devices[i]);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, (size_t) ne * sizeof(float), cuda_ctx->stream()));
            }
        }
    }

    // Pre-convert F32 -> BF16 into bf16_tmp ONLY for the copy_engine + use_bf16
    // path; the chunked kernel path's combined kernel does the conversion
    // inline as it writes to host_buf.
    ggml_cuda_pool_alloc<nv_bfloat16> bf16_tmp[GGML_CUDA_MAX_DEVICES];
    void * copy_src_ptr[GGML_CUDA_MAX_DEVICES] = {};

    if (use_copy_engine && use_bf16) {
        to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
        for (int i = 0; i < n; ++i) {
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            bf16_tmp[i].pool = &cuda_ctx->pool();
            bf16_tmp[i].alloc(ne);
            ggml_cuda_set_device(p->devices[i]);
            if (compute_flag[i]) {
                to_bf16(tensors[i]->data, bf16_tmp[i].get(), ne, cuda_ctx->stream());
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemsetAsync(bf16_tmp[i].get(), 0, nbytes, cuda_ctx->stream()));
            }
            copy_src_ptr[i] = bf16_tmp[i].get();
        }
    }

    bool ok = true;
    if (use_copy_engine) {
        // After up-front BF16 conversion, the tmp buffers already hold the
        // (possibly zeroed-for-inactive) data, so the inner path can treat
        // every shard as compute.
        bool inner_compute[GGML_CUDA_MAX_DEVICES];
        for (int i = 0; i < n; ++i) {
            inner_compute[i] = use_bf16 ? true : compute_flag[i];
        }

        // Dispatch into copy_impl with explicit src/dst types.  When use_bf16
        // is on, the wire type is BF16 (src = bf16_tmp) and the accumulator
        // is F32 (dst = tensors[i]->data); the combined add kernel rounds dst
        // through BF16 for bit-equivalence and writes F32 directly, so no
        // post-conversion is needed.  Otherwise src == dst (same native type).
        if (use_bf16) {
            GGML_ASSERT(kernel_type == GGML_TYPE_BF16);
            nv_bfloat16 * src[GGML_CUDA_MAX_DEVICES] = {};
            float       * dst[GGML_CUDA_MAX_DEVICES] = {};
            for (int i = 0; i < n; ++i) {
                src[i] = static_cast<nv_bfloat16 *>(copy_src_ptr[i]);
                dst[i] = static_cast<float *>(tensors[i]->data);
            }
            ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, float>(
                p, backends, src, dst, inner_compute, ne);
        } else {
            switch (kernel_type) {
                case GGML_TYPE_F32: {
                    float * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<float *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<float, float>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_BF16: {
                    nv_bfloat16 * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<nv_bfloat16 *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, nv_bfloat16>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_F16: {
                    half * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<half *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<half, half>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                default:
                    GGML_ASSERT(false);
            }
        }
    } else {
        // host_buf carries T_wire-typed data; max_chunk_elems is the count that
        // fits in one host_buf at the wire size.
        const size_t max_chunk_elems = p->buf_bytes / type_size;
        const size_t input_type_size = ggml_type_size(input_type);

        // Chunked kernel path runs entirely on the caller's compute stream:
        // since AR is a barrier here, same-stream ordering subsumes any
        // cross-stream event handshake that the copy-engine path needs, and
        // skips the cross-stream scheduling overhead that was hurting the
        // small-tensor (tg) latency on the AR-stream variant.  Only ev.ker is
        // still recorded at end-of-AR for acquire_slot's pool-wraparound check.
        for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
            const size_t remaining_elems = (size_t) (ne - chunk_start);
            const size_t chunk_elems = remaining_elems < max_chunk_elems ? remaining_elems : max_chunk_elems;
            const size_t chunk_dst_bytes  = chunk_elems * input_type_size;

            const auto [slot, token] = ggml_cuda_ar_acquire_slot(p);
            const bool last_chunk = chunk_start + (int64_t) chunk_elems == ne;

            for (int i = 0; i < n; ++i) {
                const int peer = 1 - i;  // valid for n == 2 only
                ggml_cuda_set_device(p->devices[i]);
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                cudaStream_t stream = cuda_ctx->stream();

                char * data = static_cast<char *>(tensors[i]->data) + chunk_start * (int64_t) input_type_size;

                // Match NCCL/meta-backend semantics: inactive shards contribute
                // zeros.  On the BF16 path the F32 tensor data was already
                // zeroed up-front (above), so per-chunk zeroing isn't needed.
                if (!compute_flag[i] && !use_bf16) {
                    CUDA_CHECK(cudaMemsetAsync(data, 0, chunk_dst_bytes, stream));
                }

#define LAUNCH_AR_KERNEL(T_dst, T_wire) \
                ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>( \
                    reinterpret_cast<const T_dst *>(data), \
                    reinterpret_cast<T_dst *>(data), \
                    reinterpret_cast<T_wire *>(p->host_buf[i].dev + (size_t) slot * p->buf_bytes), \
                    reinterpret_cast<const T_wire *>(p->host_buf[peer].dev + (size_t) slot * p->buf_bytes), \
                    static_cast<int>(chunk_elems), \
                    ggml_cuda_ar_arrival_ptr(p, slot, i), \
                    ggml_cuda_ar_arrival_ptr(p, slot, peer), \
                    token)

                if (use_bf16) {
                    GGML_ASSERT(input_type == GGML_TYPE_F32);
                    LAUNCH_AR_KERNEL(float, nv_bfloat16);
                } else {
                    switch (input_type) {
                        case GGML_TYPE_F32:  LAUNCH_AR_KERNEL(float,       float);       break;
                        case GGML_TYPE_F16:  LAUNCH_AR_KERNEL(half,        half);        break;
                        case GGML_TYPE_BF16: LAUNCH_AR_KERNEL(nv_bfloat16, nv_bfloat16); break;
                        default: GGML_ASSERT(false);
                    }
                }

#undef LAUNCH_AR_KERNEL
                CUDA_CHECK(cudaGetLastError());

                if (last_chunk) {
                    CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, stream));
                }
            }
        }
    }

    return ok;
}

#else // defined(GGML_USE_MUSA)

// MUSA lacks the host-mapped pinned-memory APIs (cudaHostAllocPortable
// / cudaHostAllocMapped / cudaHostGetDevicePointer) and a device-side
// sleep intrinsic that this implementation relies on, so the internal
// AllReduce is unavailable there. The dispatcher in ggml-cuda.cu treats
// a nullptr pipeline as "init failed" and silently falls back to the meta
// backend's generic AllReduce.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int *, size_t) {
    return nullptr;
}
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline *) {
}
bool ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **) {
    return false;
}

#endif // !defined(GGML_USE_MUSA)
