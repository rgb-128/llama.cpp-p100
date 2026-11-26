#include "mmid-f16-p100.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "convert.cuh"
#include "mmid.cuh"
#include "vecdotq.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace {

constexpr int MMID_BK         = 32;
constexpr int MMID_BK2        = MMID_BK / 2; // 16 half2 steps per K-tile
constexpr int MMID_FOLD_TILES = 4;           // fold every 4 K-tiles = 128 K-elements (64 half2 steps)

static __device__ __forceinline__ uint32_t mmid_ld2(const uint8_t * p) {
    const uint16_t * q = reinterpret_cast<const uint16_t *>(p);
    return (uint32_t) __ldg(q) | ((uint32_t) __ldg(q + 1) << 16);
}

static __device__ __forceinline__ uint32_t mmid_ld4(const uint8_t * p) {
    return __ldg(reinterpret_cast<const uint32_t *>(p));
}

static __device__ __forceinline__ void mmid_ld8u(const uint8_t * p, uint32_t & a0, uint32_t & a1) {
    const uint32_t * q = reinterpret_cast<const uint32_t *>(reinterpret_cast<uintptr_t>(p) & ~uintptr_t(3));
    const uint32_t sh = (uint32_t) (reinterpret_cast<uintptr_t>(p) & 3u) * 8u;
    const uint32_t w0 = __ldg(q);
    const uint32_t w1 = __ldg(q + 1);
    const uint32_t w2 = sh ? __ldg(q + 2) : 0u;
    a0 = __funnelshift_r(w0, w1, sh);
    a1 = __funnelshift_r(w1, w2, sh);
}

static __device__ __forceinline__ float mmid_h(const uint8_t * p) {
    const uint16_t bits = __ldg(reinterpret_cast<const uint16_t *>(p));
    return __half2float(__ushort_as_half(bits));
}

static __device__ __forceinline__ uint32_t mmid_spread4(const uint32_t b) {
    return ((b & 1u) << 4) | ((b & 2u) << 11) | ((b & 4u) << 18) | ((b & 8u) << 25);
}

static __device__ __forceinline__ uint32_t mmid_iq4(const uint32_t v) {
    const uint32_t x   = v & 0x07070707u;
    const uint32_t y   = x | (x >> 4);
    const uint32_t sel = (y & 0xFFu) | ((y >> 8) & 0xFF00u);
    const uint32_t lo  = __byte_perm(0x3f2d1801u, 0x766a5d4fu, sel);
    const uint32_t hi  = __byte_perm(0xa6998d81u, 0xf1d9c5b5u, sel);
    const uint32_t m   = ((v >> 3) & 0x01010101u) * 0xFFu;
    return (lo & ~m) | (hi & m);
}

static __device__ __forceinline__ uint32_t mmid_fp4(const uint32_t v) {
    const uint32_t x   = v & 0x07070707u;
    const uint32_t y   = x | (x >> 4);
    const uint32_t sel = (y & 0xFFu) | ((y >> 8) & 0xFF00u);
    const uint32_t lo  = __byte_perm(0x83828180u, 0x8c888684u, sel);
    const uint32_t hi  = __byte_perm(0x7d7e7f80u, 0x74787a7cu, sel);
    const uint32_t m   = ((v >> 3) & 0x01010101u) * 0xFFu;
    return (lo & ~m) | (hi & m);
}

static __device__ __forceinline__ void mmid_sm_k4(const int j, const uint8_t * q, int & d, int & m) {
    if (j < 4) {
        d = __ldg(q + j) & 63;
        m = __ldg(q + j + 4) & 63;
    } else {
        const uint8_t qj4 = __ldg(q + j + 4);
        d = (qj4 & 0xF) | ((__ldg(q + j - 4) >> 6) << 4);
        m = (qj4 >>  4) | ((__ldg(q + j)     >> 6) << 4);
    }
}

static __device__ __forceinline__ uint32_t mmid_sgnw(const uint32_t g, const uint32_t bits) {
    const uint32_t x   = ((bits & 0xFu) * 0x01010101u) & 0x08040201u;
    const uint32_t one = ((x + 0x7F7F7F7Fu) >> 7) & 0x01010101u;
    const uint32_t m   = (one << 8) - one;
    return ((g ^ m) + one) ^ 0x80808080u;
}

template <bool MIN>
static __device__ __forceinline__ void mmid_w8_h2(
        const uint32_t v0, const uint32_t v1,
        const __half2 kb, const __half2 s2, const __half2 mn2, __half2 (&w)[4]) {
    const uint32_t h0 = __byte_perm(v0, 0x64646464u, 0x5140u);
    const uint32_t h1 = __byte_perm(v0, 0x64646464u, 0x5342u);
    const uint32_t h2 = __byte_perm(v1, 0x64646464u, 0x5140u);
    const uint32_t h3 = __byte_perm(v1, 0x64646464u, 0x5342u);
    const __half2 q0 = __hsub2(*reinterpret_cast<const __half2 *>(&h0), kb);
    const __half2 q1 = __hsub2(*reinterpret_cast<const __half2 *>(&h1), kb);
    const __half2 q2 = __hsub2(*reinterpret_cast<const __half2 *>(&h2), kb);
    const __half2 q3 = __hsub2(*reinterpret_cast<const __half2 *>(&h3), kb);
    if constexpr (MIN) {
        w[0] = __hfma2(q0, s2, mn2);
        w[1] = __hfma2(q1, s2, mn2);
        w[2] = __hfma2(q2, s2, mn2);
        w[3] = __hfma2(q3, s2, mn2);
    } else {
        w[0] = __hmul2(q0, s2);
        w[1] = __hmul2(q1, s2);
        w[2] = __hmul2(q2, s2);
        w[3] = __hmul2(q3, s2);
    }
}

// Fast 4-instruction integer half2 -> float fold: returns (lo + hi) * 2^-112.
static __device__ __forceinline__ float mmid_fold_h(const half2 p) {
    const half2   s = __hadd2(p, __lowhigh2highlow(p));
    const int32_t v = ((int32_t) *reinterpret_cast<const uint32_t *>(&s)) >> 3;
    return __int_as_float(v & 0x8FFFE000);
}

// Per-quant-type 16-weight on-the-fly unpack (zero extra VRAM fallback path).
template <ggml_type T> struct mmid_f16_t;

template <> struct mmid_f16_t<GGML_TYPE_Q4_0> {
    static constexpr int QK = 32, BS = 18;
    static constexpr float WSCALE = 8.0f;
    static __device__ __forceinline__ void w16(const uint8_t * b, const int sub0, __half2 (&w0)[4], __half2 (&w1)[4]) {
        const uint8_t * qs = b + 2;
        const int sh = 4*(sub0 >> 1);
        const __half2 kb = __float2half2_rn(1024.0f + 8.0f);
        const __half2 s2 = __float2half2_rn(mmid_h(b) * WSCALE);
        const __half2 z2 = __float2half2_rn(0.0f);
        mmid_w8_h2<false>((mmid_ld2(qs)      >> sh) & 0x0F0F0F0Fu, (mmid_ld2(qs +  4) >> sh) & 0x0F0F0F0Fu, kb, s2, z2, w0);
        mmid_w8_h2<false>((mmid_ld2(qs +  8) >> sh) & 0x0F0F0F0Fu, (mmid_ld2(qs + 12) >> sh) & 0x0F0F0F0Fu, kb, s2, z2, w1);
    }
};

template <> struct mmid_f16_t<GGML_TYPE_Q4_1> {
    static constexpr int QK = 32, BS = 20;
    static constexpr float WSCALE = 8.0f;
    static __device__ __forceinline__ void w16(const uint8_t * b, const int sub0, __half2 (&w0)[4], __half2 (&w1)[4]) {
        const uint8_t * qs = b + 4;
        const int sh = 4*(sub0 >> 1);
        const __half2 kb  = __float2half2_rn(1024.0f);
        const __half2 s2  = __float2half2_rn(mmid_h(b)     * WSCALE);
        const __half2 mn2 = __float2half2_rn(mmid_h(b + 2) * WSCALE);
        mmid_w8_h2<true>((mmid_ld4(qs)      >> sh) & 0x0F0F0F0Fu, (mmid_ld4(qs +  4) >> sh) & 0x0F0F0F0Fu, kb, s2, mn2, w0);
        mmid_w8_h2<true>((mmid_ld4(qs +  8) >> sh) & 0x0F0F0F0Fu, (mmid_ld4(qs + 12) >> sh) & 0x0F0F0F0Fu, kb, s2, mn2, w1);
    }
};

template <> struct mmid_f16_t<GGML_TYPE_Q5_0> {
    static constexpr int QK = 32, BS = 22;
    static constexpr float WSCALE = 8.0f;
    static __device__ __forceinline__ void w16(const uint8_t * b, const int sub0, __half2 (&w0)[4], __half2 (&w1)[4]) {
        const uint8_t * qs = b + 6;
        const int sh = 4*(sub0 >> 1);
        const uint32_t hb = mmid_ld2(b + 2) >> (8*sub0);
        const __half2 kb = __float2half2_rn(1024.0f + 16.0f);
        const __half2 s2 = __float2half2_rn(mmid_h(b) * WSCALE);
        const __half2 z2 = __float2half2_rn(0.0f);
        mmid_w8_h2<false>(((mmid_ld2(qs)      >> sh) & 0x0F0F0F0Fu) | mmid_spread4(hb),
                          ((mmid_ld2(qs +  4) >> sh) & 0x0F0F0F0Fu) | mmid_spread4(hb >> 4), kb, s2, z2, w0);
        mmid_w8_h2<false>(((mmid_ld2(qs +  8) >> sh) & 0x0F0F0F0Fu) | mmid_spread4(hb >> 8),
                          ((mmid_ld2(qs + 12) >> sh) & 0x0F0F0F0Fu) | mmid_spread4(hb >> 12), kb, s2, z2, w1);
    }
};

template <> struct mmid_f16_t<GGML_TYPE_Q5_1> {
    static constexpr int QK = 32, BS = 24;
    static constexpr float WSCALE = 8.0f;
    static __device__ __forceinline__ void w16(const uint8_t * b, const int sub0, __half2 (&w0)[4], __half2 (&w1)[4]) {
        const uint8_t * qs = b + 8;
        const int sh = 4*(sub0 >> 1);
        const uint32_t hb = mmid_ld4(b + 4) >> (8*sub0);
        const __half2 kb  = __float2half2_rn(1024.0f);
        const __half2 s2  = __float2half2_rn(mmid_h(b)     * WSCALE);
        const __half2 mn2 = __float2half2_rn(mmid_h(b + 2) * WSCALE);
        mmid_w8_h2<true>(((mmid_ld4(qs)      >> sh) & 0x0F0F0F0Fu) | mmid_spread4(hb),
                         ((mmid_ld4(qs +  4) >> sh) & 0x0F0F0F0Fu) | mmid_spread4(hb >> 4), kb, s2, mn2, w0);
        mmid_w8_h2<true>(((mmid_ld4(qs +  8) >> sh) & 0x0F0F0F0Fu) | mmid_spread4(hb >> 8),
                         ((mmid_ld4(qs + 12) >> sh) & 0x0F0F0F0Fu) | mmid_spread4(hb >> 12), kb, s2, mn2, w1);
    }
};

template <> struct mmid_f16_t<GGML_TYPE_Q8_0> {
    static constexpr int QK = 32, BS = 34;
    static constexpr float WSCALE = 8.0f;
    static __device__ __forceinline__ void w16(const uint8_t * b, const int sub0, __half2 (&w0)[4], __half2 (&w1)[4]) {
        const uint8_t * qs = b + 2 + 8*sub0;
        const __half2 kb = __float2half2_rn(1024.0f + 128.0f);
        const __half2 s2 = __float2half2_rn(mmid_h(b) * WSCALE);
        const __half2 z2 = __float2half2_rn(0.0f);
        mmid_w8_h2<false>(mmid_ld2(qs)      ^ 0x80808080u, mmid_ld2(qs +  4) ^ 0x80808080u, kb, s2, z2, w0);
        mmid_w8_h2<false>(mmid_ld2(qs +  8) ^ 0x80808080u, mmid_ld2(qs + 12) ^ 0x80808080u, kb, s2, z2, w1);
    }
};

template <> struct mmid_f16_t<GGML_TYPE_IQ4_NL> {
    static constexpr int QK = 32, BS = 18;
    static constexpr float WSCALE = 8.0f;
    static __device__ __forceinline__ void w16(const uint8_t * b, const int sub0, __half2 (&w0)[4], __half2 (&w1)[4]) {
        const uint8_t * qs = b + 2;
        const int sh = 4*(sub0 >> 1);
        const __half2 kb = __float2half2_rn(1024.0f + 128.0f);
        const __half2 s2 = __float2half2_rn(mmid_h(b) * WSCALE);
        const __half2 z2 = __float2half2_rn(0.0f);
        mmid_w8_h2<false>(mmid_iq4((mmid_ld2(qs)      >> sh) & 0x0F0F0F0Fu),
                          mmid_iq4((mmid_ld2(qs +  4) >> sh) & 0x0F0F0F0Fu), kb, s2, z2, w0);
        mmid_w8_h2<false>(mmid_iq4((mmid_ld2(qs +  8) >> sh) & 0x0F0F0F0Fu),
                          mmid_iq4((mmid_ld2(qs + 12) >> sh) & 0x0F0F0F0Fu), kb, s2, z2, w1);
    }
};

template <> struct mmid_f16_t<GGML_TYPE_IQ4_XS> {
    static constexpr int QK = 256, BS = 136;
    static constexpr float WSCALE = 8.0f;
    static __device__ __forceinline__ void w16(const uint8_t * b, const int sub0, __half2 (&w0)[4], __half2 (&w1)[4]) {
        const int ib = sub0 >> 2, t = sub0 & 3;
        const uint8_t * qs = b + 8 + 16*ib;
        const int sh = 4*(t >> 1);
        const uint16_t sh16 = __ldg(reinterpret_cast<const uint16_t *>(b + 2));
        const uint8_t  sl8  = __ldg(b + 4 + (ib >> 1));
        const int ls = ((sl8 >> (4*(ib & 1))) & 0xF) | (((sh16 >> (2*ib)) & 3) << 4);
        const __half2 kb = __float2half2_rn(1024.0f + 128.0f);
        const __half2 s2 = __float2half2_rn(mmid_h(b) * (float) (ls - 32) * WSCALE);
        const __half2 z2 = __float2half2_rn(0.0f);
        mmid_w8_h2<false>(mmid_iq4((mmid_ld4(qs)      >> sh) & 0x0F0F0F0Fu),
                          mmid_iq4((mmid_ld4(qs +  4) >> sh) & 0x0F0F0F0Fu), kb, s2, z2, w0);
        mmid_w8_h2<false>(mmid_iq4((mmid_ld4(qs +  8) >> sh) & 0x0F0F0F0Fu),
                          mmid_iq4((mmid_ld4(qs + 12) >> sh) & 0x0F0F0F0Fu), kb, s2, z2, w1);
    }
};

template <> struct mmid_f16_t<GGML_TYPE_Q2_K> {
    static constexpr int QK = 256, BS = 84;
    static constexpr float WSCALE = 8.0f;
    static __device__ __forceinline__ void w16(const uint8_t * b, const int sub0, __half2 (&w0)[4], __half2 (&w1)[4]) {
        const int n = sub0 >> 4, j = (sub0 >> 2) & 3, l0 = 8*(sub0 & 3);
        const uint8_t * qs = b + 16 + 32*n + l0;
        const int sc = __ldg(b + 8*n + 2*j + (l0 >= 16));
        const __half2 kb  = __float2half2_rn(1024.0f);
        const __half2 s2  = __float2half2_rn( mmid_h(b + 80) * (float) (sc & 0xF) * WSCALE);
        const __half2 mn2 = __float2half2_rn(-mmid_h(b + 82) * (float) (sc >>  4) * WSCALE);
        mmid_w8_h2<true>((mmid_ld4(qs)      >> (2*j)) & 0x03030303u, (mmid_ld4(qs +  4) >> (2*j)) & 0x03030303u, kb, s2, mn2, w0);
        mmid_w8_h2<true>((mmid_ld4(qs +  8) >> (2*j)) & 0x03030303u, (mmid_ld4(qs + 12) >> (2*j)) & 0x03030303u, kb, s2, mn2, w1);
    }
};

template <> struct mmid_f16_t<GGML_TYPE_Q3_K> {
    static constexpr int QK = 256, BS = 110;
    static constexpr float WSCALE = 8.0f;
    static __device__ __forceinline__ void w16(const uint8_t * b, const int sub0, __half2 (&w0)[4], __half2 (&w1)[4]) {
        const int n = sub0 >> 4, j = (sub0 >> 2) & 3, l0 = 8*(sub0 & 3);
        const uint8_t * qs = b + 32 + 32*n + l0;
        const uint8_t * hm = b + l0;
        const int hs = 4*n + j;
        const int is = 8*n + 2*j + (l0 >= 16);
        const uint8_t * sc = b + 96;
        const int s6 = (is < 8 ? (__ldg(sc + is) & 0xF) : (__ldg(sc + is - 8) >> 4)) |
                       (((__ldg(sc + 8 + (is & 3)) >> (2*(is >> 2))) & 3) << 4);
        const __half2 kb = __float2half2_rn(1024.0f + 4.0f);
        const __half2 s2 = __float2half2_rn(mmid_h(b + 108) * (float) (s6 - 32) * WSCALE);
        const __half2 z2 = __float2half2_rn(0.0f);
        mmid_w8_h2<false>(((mmid_ld2(qs)      >> (2*j)) & 0x03030303u) | (((mmid_ld2(hm)      >> hs) & 0x01010101u) << 2),
                          ((mmid_ld2(qs +  4) >> (2*j)) & 0x03030303u) | (((mmid_ld2(hm +  4) >> hs) & 0x01010101u) << 2), kb, s2, z2, w0);
        mmid_w8_h2<false>(((mmid_ld2(qs +  8) >> (2*j)) & 0x03030303u) | (((mmid_ld2(hm +  8) >> hs) & 0x01010101u) << 2),
                          ((mmid_ld2(qs + 12) >> (2*j)) & 0x03030303u) | (((mmid_ld2(hm + 12) >> hs) & 0x01010101u) << 2), kb, s2, z2, w1);
    }
};

template <> struct mmid_f16_t<GGML_TYPE_Q4_K> {
    static constexpr int QK = 256, BS = 144;
    static constexpr float WSCALE = 8.0f;
    static __device__ __forceinline__ void w16(const uint8_t * b, const int sub0, __half2 (&w0)[4], __half2 (&w1)[4]) {
        const int j = sub0 >> 3, t = sub0 & 7, is = 2*j + (t >> 2), sh = 4*(t >> 2);
        const uint8_t * qs = b + 16 + 32*j + 8*(t & 3);
        int sc, m;
        mmid_sm_k4(is, b + 4, sc, m);
        const __half2 kb  = __float2half2_rn(1024.0f);
        const __half2 s2  = __float2half2_rn( mmid_h(b)     * (float) sc * WSCALE);
        const __half2 mn2 = __float2half2_rn(-mmid_h(b + 2) * (float) m  * WSCALE);
        mmid_w8_h2<true>((mmid_ld4(qs)      >> sh) & 0x0F0F0F0Fu, (mmid_ld4(qs +  4) >> sh) & 0x0F0F0F0Fu, kb, s2, mn2, w0);
        mmid_w8_h2<true>((mmid_ld4(qs +  8) >> sh) & 0x0F0F0F0Fu, (mmid_ld4(qs + 12) >> sh) & 0x0F0F0F0Fu, kb, s2, mn2, w1);
    }
};

template <> struct mmid_f16_t<GGML_TYPE_Q5_K> {
    static constexpr int QK = 256, BS = 176;
    static constexpr float WSCALE = 8.0f;
    static __device__ __forceinline__ void w16(const uint8_t * b, const int sub0, __half2 (&w0)[4], __half2 (&w1)[4]) {
        const int j = sub0 >> 3, t = sub0 & 7, is = 2*j + (t >> 2), sh = 4*(t >> 2);
        const uint8_t * qs = b + 48 + 32*j + 8*(t & 3);
        const uint8_t * qh = b + 16 + 8*(t & 3);
        int sc, m;
        mmid_sm_k4(is, b + 4, sc, m);
        const __half2 kb  = __float2half2_rn(1024.0f);
        const __half2 s2  = __float2half2_rn( mmid_h(b)     * (float) sc * WSCALE);
        const __half2 mn2 = __float2half2_rn(-mmid_h(b + 2) * (float) m  * WSCALE);
        mmid_w8_h2<true>(((mmid_ld4(qs)      >> sh) & 0x0F0F0F0Fu) | (((mmid_ld4(qh)      >> is) & 0x01010101u) << 4),
                         ((mmid_ld4(qs +  4) >> sh) & 0x0F0F0F0Fu) | (((mmid_ld4(qh +  4) >> is) & 0x01010101u) << 4), kb, s2, mn2, w0);
        mmid_w8_h2<true>(((mmid_ld4(qs +  8) >> sh) & 0x0F0F0F0Fu) | (((mmid_ld4(qh +  8) >> is) & 0x01010101u) << 4),
                         ((mmid_ld4(qs + 12) >> sh) & 0x0F0F0F0Fu) | (((mmid_ld4(qh + 12) >> is) & 0x01010101u) << 4), kb, s2, mn2, w1);
    }
};

template <> struct mmid_f16_t<GGML_TYPE_Q6_K> {
    static constexpr int QK = 256, BS = 210;
    static constexpr float WSCALE = 8.0f;
    static __device__ __forceinline__ void w16(const uint8_t * b, const int sub0, __half2 (&w0)[4], __half2 (&w1)[4]) {
        const int ip   = sub0 >> 4;
        const int q32  = (sub0 >> 2) & 3;
        const int l8   = (sub0 & 3) * 8;
        const int is   = 8*ip + 2*q32 + (l8 >= 16);
        const int ql_s = 4*(q32 >> 1);
        const int qh_s = 2*q32;
        const uint8_t * ql = b + 64*ip + 32*(q32 & 1) + l8;
        const uint8_t * qh = b + 128 + 32*ip + l8;
        const int8_t sc = static_cast<int8_t>(__ldg(b + 192 + is));
        const __half2 kb = __float2half2_rn(1024.0f + 32.0f);
        const __half2 s2 = __float2half2_rn(mmid_h(b + 208) * (float) sc * WSCALE);
        const __half2 z2 = __float2half2_rn(0.0f);
        mmid_w8_h2<false>(((mmid_ld2(ql)      >> ql_s) & 0x0F0F0F0Fu) | (((mmid_ld2(qh)      >> qh_s) & 0x03030303u) << 4),
                          ((mmid_ld2(ql +  4) >> ql_s) & 0x0F0F0F0Fu) | (((mmid_ld2(qh +  4) >> qh_s) & 0x03030303u) << 4), kb, s2, z2, w0);
        mmid_w8_h2<false>(((mmid_ld2(ql +  8) >> ql_s) & 0x0F0F0F0Fu) | (((mmid_ld2(qh +  8) >> qh_s) & 0x03030303u) << 4),
                          ((mmid_ld2(ql + 12) >> ql_s) & 0x0F0F0F0Fu) | (((mmid_ld2(qh + 12) >> qh_s) & 0x03030303u) << 4), kb, s2, z2, w1);
    }
};

template <> struct mmid_f16_t<GGML_TYPE_IQ3_S> {
    static constexpr int QK = 256, BS = 110;
    static constexpr float WSCALE = 8.0f;
    static __device__ __forceinline__ void w16(const uint8_t * b, const int sub0, __half2 (&w0)[4], __half2 (&w1)[4]) {
        const int ib = sub0 >> 2, l = sub0 & 3;
        const uint8_t * qs = b + 2 + 8*ib + 2*l;
        const uint32_t qh  = __ldg(b + 66 + ib);
        const uint32_t sg0 = __ldg(b + 74 + 4*ib + l);
        const uint32_t sg1 = __ldg(b + 74 + 4*ib + l + 1);
        const int ls = (__ldg(b + 106 + (ib >> 1)) >> (4*(ib & 1))) & 0xF;
        const __half2 kb = __float2half2_rn(1024.0f + 128.0f);
        const __half2 s2 = __float2half2_rn(mmid_h(b) * (float) (2*ls + 1) * WSCALE);
        const __half2 z2 = __float2half2_rn(0.0f);
        const uint32_t idx0 = __ldg(qs)     | ((qh << (8 - 2*l)) & 0x100u);
        const uint32_t idx1 = __ldg(qs + 1) | ((qh << (7 - 2*l)) & 0x100u);
        const uint32_t idx2 = __ldg(qs + 2) | ((qh << (6 - 2*l)) & 0x100u);
        const uint32_t idx3 = __ldg(qs + 3) | ((qh << (5 - 2*l)) & 0x100u);
        mmid_w8_h2<false>(mmid_sgnw(iq3s_grid[idx0], sg0), mmid_sgnw(iq3s_grid[idx1], sg0 >> 4), kb, s2, z2, w0);
        mmid_w8_h2<false>(mmid_sgnw(iq3s_grid[idx2], sg1), mmid_sgnw(iq3s_grid[idx3], sg1 >> 4), kb, s2, z2, w1);
    }
};

template <> struct mmid_f16_t<GGML_TYPE_MXFP4> {
    static constexpr int QK = 32, BS = 17;
    static constexpr float WSCALE = 1.0f;
    static __device__ __forceinline__ void w16(const uint8_t * b, const int sub0, __half2 (&w0)[4], __half2 (&w1)[4]) {
        const int sh = 4*(sub0 >> 1);
        uint32_t a0, a1, a2, a3;
        mmid_ld8u(b + 1, a0, a1);
        mmid_ld8u(b + 9, a2, a3);
        const __half2 kb = __float2half2_rn(1024.0f + 128.0f);
        const __half2 s2 = __float2half2_rn(ggml_cuda_e8m0_to_fp32(__ldg(b)) * 0.5f);
        const __half2 z2 = __float2half2_rn(0.0f);
        mmid_w8_h2<false>(mmid_fp4((a0 >> sh) & 0x0F0F0F0Fu), mmid_fp4((a1 >> sh) & 0x0F0F0F0Fu), kb, s2, z2, w0);
        mmid_w8_h2<false>(mmid_fp4((a2 >> sh) & 0x0F0F0F0Fu), mmid_fp4((a3 >> sh) & 0x0F0F0F0Fu), kb, s2, z2, w1);
    }
};

template <> struct mmid_f16_t<GGML_TYPE_NVFP4> {
    static constexpr int QK = 64, BS = 36;
    static constexpr float WSCALE = 1.0f;
    static __device__ __forceinline__ void w16(const uint8_t * b, const int sub0, __half2 (&w0)[4], __half2 (&w1)[4]) {
        const int sb = sub0 >> 1;
        const uint8_t * qs = b + 4 + 8*sb;
        const uint32_t q0 = mmid_ld4(qs);
        const uint32_t q1 = mmid_ld4(qs + 4);
        const __half2 kb = __float2half2_rn(1024.0f + 128.0f);
        const __half2 s2 = __float2half2_rn(ggml_cuda_ue4m3_to_fp32(__ldg(b + sb)));
        const __half2 z2 = __float2half2_rn(0.0f);
        mmid_w8_h2<false>(mmid_fp4( q0       & 0x0F0F0F0Fu), mmid_fp4( q1       & 0x0F0F0F0Fu), kb, s2, z2, w0);
        mmid_w8_h2<false>(mmid_fp4((q0 >> 4) & 0x0F0F0F0Fu), mmid_fp4((q1 >> 4) & 0x0F0F0F0Fu), kb, s2, z2, w1);
    }
};

// Gather + power-of-2 column prescale for non-broadcast MUL_MAT_ID inputs (e.g. down_exps).
__global__ void __launch_bounds__(256) mmid_f16_prescale_gather(
        const char * __restrict__ src1_base, const size_t nb11,
        const int32_t * __restrict__ ids_src1,
        half * __restrict__ X16, float * __restrict__ cs, const int K, const int xexp) {
    const int compact_row = blockIdx.x;
    const int src_row = ids_src1[compact_row];
    const float * x = reinterpret_cast<const float *>(src1_base + (size_t) src_row * nb11);
    const bool aligned = ((reinterpret_cast<uintptr_t>(x) & 15u) == 0) && ((K & 3) == 0);
    float m = 0.0f;
    if (aligned) {
        const float4 * x4 = reinterpret_cast<const float4 *>(x);
        const int K4 = K >> 2;
        for (int k = threadIdx.x; k < K4; k += 256) {
            const float4 v = __ldg(x4 + k);
            m = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
    } else {
        for (int k = threadIdx.x; k < K; k += 256) {
            m = fmaxf(m, fabsf(__ldg(x + k)));
        }
    }
    __shared__ float sm[32];
    m = warp_reduce_max(m);
    if ((threadIdx.x & 31) == 0) {
        sm[threadIdx.x >> 5] = m;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        m = threadIdx.x < 8 ? sm[threadIdx.x] : 0.0f;
        m = warp_reduce_max(m);
        if (threadIdx.x == 0) {
            sm[0] = m;
        }
    }
    __syncthreads();
    m = sm[0];
    const int   e = (m > 0.0f && isfinite(m)) ? (ilogbf(m) - xexp) : 0;
    const float s = ldexpf(1.0f, -e);
    half * y = X16 + (int64_t) compact_row * K;
    if (aligned) {
        const float4 * x4 = reinterpret_cast<const float4 *>(x);
        uint2 * y4 = reinterpret_cast<uint2 *>(y);
        const int K4 = K >> 2;
        for (int k = threadIdx.x; k < K4; k += 256) {
            const float4 v = __ldg(x4 + k);
            const __half2 h0 = __floats2half2_rn(v.x * s, v.y * s);
            const __half2 h1 = __floats2half2_rn(v.z * s, v.w * s);
            y4[k] = make_uint2(*reinterpret_cast<const uint32_t *>(&h0),
                               *reinterpret_cast<const uint32_t *>(&h1));
        }
    } else {
        for (int k = threadIdx.x; k < K; k += 256) {
            y[k] = __float2half(__ldg(x + k) * s);
        }
    }
    if (threadIdx.x == 0) {
        cs[compact_row] = ldexpf(1.0f, e) * 0x1p112f;
    }
}

// Deduplicated prescale + scatter for broadcast MUL_MAT_ID inputs (ne11 == 1: gate_exps / up_exps).
__global__ void __launch_bounds__(256) mmid_f16_prescale_scatter(
        const char * __restrict__ src1_base, const size_t nb12,
        const int32_t * __restrict__ ids_inv,
        half * __restrict__ X16, float * __restrict__ cs,
        const int K, const int n_expert_used, const int xexp) {
    const int it = blockIdx.x;
    const float * x = reinterpret_cast<const float *>(src1_base + (size_t) it * nb12);
    const bool aligned = ((reinterpret_cast<uintptr_t>(x) & 15u) == 0) && ((K & 3) == 0);
    float m = 0.0f;
    if (aligned) {
        const float4 * x4 = reinterpret_cast<const float4 *>(x);
        const int K4 = K >> 2;
        for (int k = threadIdx.x; k < K4; k += 256) {
            const float4 v = __ldg(x4 + k);
            m = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
    } else {
        for (int k = threadIdx.x; k < K; k += 256) {
            m = fmaxf(m, fabsf(__ldg(x + k)));
        }
    }
    __shared__ float sm[32];
    m = warp_reduce_max(m);
    if ((threadIdx.x & 31) == 0) {
        sm[threadIdx.x >> 5] = m;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        m = threadIdx.x < 8 ? sm[threadIdx.x] : 0.0f;
        m = warp_reduce_max(m);
        if (threadIdx.x == 0) {
            sm[0] = m;
        }
    }
    __syncthreads();
    m = sm[0];
    const int   e   = (m > 0.0f && isfinite(m)) ? (ilogbf(m) - xexp) : 0;
    const float inv = ldexpf(1.0f, -e);
    const float sc  = ldexpf(1.0f, e) * 0x1p112f;

    __shared__ int32_t dst_rows[64];
    for (int iex = threadIdx.x; iex < n_expert_used; iex += 256) {
        const int32_t crow = ids_inv[it * n_expert_used + iex];
        if (iex < 64) {
            dst_rows[iex] = crow;
        }
        cs[crow] = sc;
    }
    __syncthreads();

    if (aligned) {
        const float4 * x4 = reinterpret_cast<const float4 *>(x);
        const int K4 = K >> 2;
        for (int k = threadIdx.x; k < K4; k += 256) {
            const float4 v = __ldg(x4 + k);
            const __half2 h0 = __floats2half2_rn(v.x * inv, v.y * inv);
            const __half2 h1 = __floats2half2_rn(v.z * inv, v.w * inv);
            const uint2 hv4 = make_uint2(*reinterpret_cast<const uint32_t *>(&h0),
                                         *reinterpret_cast<const uint32_t *>(&h1));
            for (int iex = 0; iex < n_expert_used; ++iex) {
                const int32_t crow = (iex < 64) ? dst_rows[iex] : ids_inv[it * n_expert_used + iex];
                reinterpret_cast<uint2 *>(X16 + (int64_t) crow * K)[k] = hv4;
            }
        }
    } else {
        for (int k = threadIdx.x; k < K; k += 256) {
            const half hv = __float2half(__ldg(x + k) * inv);
            for (int iex = 0; iex < n_expert_used; ++iex) {
                const int32_t crow = (iex < 64) ? dst_rows[iex] : ids_inv[it * n_expert_used + iex];
                X16[(int64_t) crow * K + k] = hv;
            }
        }
    }
}

// ===================================================================================================
// Path 1: Pure FP16 grouped GEMM (weights pre-dequantized via ggml_get_to_fp16_cuda).
// Works for ALL quant types (current and future) at ~13-15 TFLOPS on GP100.
//   BN =  16: BM =  64, 128 threads (16x8  -> 4x2 tile/thread, 4 blocks/SM)
//   BN =  32: BM =  64, 128 threads (16x8  -> 4x4 tile/thread, 4 blocks/SM)
//   BN =  64: BM = 128, 256 threads (16x16 -> 8x4 tile/thread, 2 blocks/SM)
//   BN = 128: BM = 128, 256 threads (16x16 -> 8x8 tile/thread, 1 block/SM)
// ===================================================================================================
template <int BN>
__global__ void __launch_bounds__(BN <= 32 ? 128 : 256, BN <= 32 ? 4 : (BN == 64 ? 2 : 1)) mmid_f16_pure_kernel(
        const half * __restrict__    W16,
        const half * __restrict__    X16,
        const float * __restrict__   cs,
        const int32_t * __restrict__ ids_dst,
        const ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const float * __restrict__   x_scale,
        float * __restrict__         dst,
        const int64_t                s1,
        const int64_t                s2,
        const int                    M,
        const int                    K,
        const int                    n_expert_used,
        const int                    m_blocks) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
    constexpr int BM     = (BN <= 32) ? 64  : 128;
    constexpr int NT     = (BN <= 32) ? 128 : 256;
    constexpr int TX_DIM = (BN <= 32) ? 8   : 16;
    constexpr int TM     = BM / 16;             // 4 for BN=16/32, 8 for BN=64/128
    constexpr int TN     = BN / TX_DIM;         // 2 for BN=16, 4 for BN=32/64, 8 for BN=128
    constexpr int BS_PAD = (BN == 16) ? 18 : BN;

    const int bid      = blockIdx.x;
    const int tile_idx = bid / m_blocks;
    const int m_idx    = bid - tile_idx * m_blocks;

    const ggml_cuda_mm_id_tile_spec ts = tile_spec[tile_idx];
    const int expert   = ts.expert;
    const int col_low  = ts.col_low;
    const int col_high = ts.col_high;
    if (col_low >= col_high) {
        return;
    }

    __shared__ __align__(16) uint32_t As[2][MMID_BK2][BM];
    __shared__ __align__(16) uint32_t Bs[2][MMID_BK2][BS_PAD];

    const int t  = threadIdx.x;
    const int tx = t & (TX_DIM - 1);
    const int ty = t / TX_DIM; // 0..15
    const int m0 = m_idx * BM;
    const int ncols_tile = col_high - col_low;

    const half * W_exp = W16 + (int64_t) expert * (int64_t) M * (int64_t) K;

    float acc[TM][TN];
    half2 h[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
            h[i][j]   = make_half2(0.0f, 0.0f);
        }
    }

    uint4 ra[2];
    uint4 rb0, rb1;
    const bool bt0 = (BN != 16) || (t < 64);

    auto gload = [&](const int k0) {
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            const int l = t + NT * w;
            const int r = l >> 2;
            const int c = l & 3;
            ra[w] = (m0 + r < M)
                ? __ldg(reinterpret_cast<const uint4 *>(W_exp + (int64_t) (m0 + r) * K + k0 + c * 8))
                : make_uint4(0u, 0u, 0u, 0u);
        }
        if (bt0) {
            const int r = t >> 2;
            const int c = t & 3;
            rb0 = (r < ncols_tile)
                ? __ldg(reinterpret_cast<const uint4 *>(X16 + (int64_t) (col_low + r) * K + k0 + c * 8))
                : make_uint4(0u, 0u, 0u, 0u);
        }
        if constexpr (BN == 128) {
            const int l = t + 256;
            const int r = l >> 2;
            const int c = l & 3;
            rb1 = (r < ncols_tile)
                ? __ldg(reinterpret_cast<const uint4 *>(X16 + (int64_t) (col_low + r) * K + k0 + c * 8))
                : make_uint4(0u, 0u, 0u, 0u);
        }
    };

    auto sstore = [&](const int buf) {
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            const int l  = t + NT * w;
            const int r  = l >> 2;
            const int c  = l & 3;
            const int rs = r ^ (c << 3);
            As[buf][c * 4 + 0][rs] = ra[w].x;
            As[buf][c * 4 + 1][rs] = ra[w].y;
            As[buf][c * 4 + 2][rs] = ra[w].z;
            As[buf][c * 4 + 3][rs] = ra[w].w;
        }
        if (bt0) {
            const int r  = t >> 2;
            const int c  = t & 3;
            const int rs = (BN == 16) ? r : (r ^ (c << 3));
            Bs[buf][c * 4 + 0][rs] = rb0.x;
            Bs[buf][c * 4 + 1][rs] = rb0.y;
            Bs[buf][c * 4 + 2][rs] = rb0.z;
            Bs[buf][c * 4 + 3][rs] = rb0.w;
        }
        if constexpr (BN == 128) {
            const int l  = t + 256;
            const int r  = l >> 2;
            const int c  = l & 3;
            const int rs = r ^ (c << 3);
            Bs[buf][c * 4 + 0][rs] = rb1.x;
            Bs[buf][c * 4 + 1][rs] = rb1.y;
            Bs[buf][c * 4 + 2][rs] = rb1.z;
            Bs[buf][c * 4 + 3][rs] = rb1.w;
        }
    };

    const int nt = K / MMID_BK;
    gload(0);
    sstore(0);
    __syncthreads();

    for (int it = 0; it < nt; ++it) {
        const int buf = it & 1;
        if (it + 1 < nt) {
            gload((it + 1) * MMID_BK);
        }
        const bool restart = (it % MMID_FOLD_TILES) == 0;
#pragma unroll
        for (int k2 = 0; k2 < MMID_BK2; ++k2) {
            const int sw = (k2 >> 2) << 3;
            uint32_t a[TM];
            const uint4 a0 = *reinterpret_cast<const uint4 *>(&As[buf][k2][(ty * 4) ^ sw]);
            a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
            if constexpr (TM == 8) {
                const uint4 a1 = *reinterpret_cast<const uint4 *>(&As[buf][k2][(64 + ty * 4) ^ sw]);
                a[4] = a1.x; a[5] = a1.y; a[6] = a1.z; a[7] = a1.w;
            }
            uint32_t b[TN];
            if constexpr (BN == 128) {
                const uint4 b0 = *reinterpret_cast<const uint4 *>(&Bs[buf][k2][(tx * 4) ^ sw]);
                const uint4 b1 = *reinterpret_cast<const uint4 *>(&Bs[buf][k2][(64 + tx * 4) ^ sw]);
                b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
                b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
            } else if constexpr (BN == 64 || BN == 32) {
                const uint4 b0 = *reinterpret_cast<const uint4 *>(&Bs[buf][k2][(tx * 4) ^ sw]);
                b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
            } else {
                const uint2 b0 = *reinterpret_cast<const uint2 *>(&Bs[buf][k2][tx * 2]);
                b[0] = b0.x; b[1] = b0.y;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const half2 ai = *reinterpret_cast<const half2 *>(&a[i]);
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const half2 bj = *reinterpret_cast<const half2 *>(&b[j]);
                    h[i][j] = (k2 == 0 && restart) ? __hmul2(ai, bj) : __hfma2(ai, bj, h[i][j]);
                }
            }
        }
        if (((it + 1) % MMID_FOLD_TILES) == 0 || it + 1 == nt) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += mmid_fold_h(h[i][j]);
                }
            }
        }
        if (it + 1 < nt) {
            sstore(buf ^ 1);
        }
        __syncthreads();
    }

    const float escale = x_scale ? __ldg(x_scale + expert) : 1.0f;
    constexpr int NH = TM / 4;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int col_off = (BN == 128) ? (j < 4 ? tx * 4 + j : 64 + tx * 4 + (j - 4)) : (tx * TN + j);
        if (col_off >= ncols_tile) {
            continue;
        }
        const int col = col_low + col_off;
        const float s = cs[col] * escale;
        const int32_t d_id = ids_dst[col];
        const int32_t tok  = d_id / n_expert_used;
        const int32_t iex  = d_id - tok * n_expert_used;
        float * out_base = dst + (int64_t) tok * s2 + (int64_t) iex * s1;
#pragma unroll
        for (int ih = 0; ih < NH; ++ih) {
            const int m = m0 + ih * 64 + ty * 4;
            if (m >= M) {
                continue;
            }
            float * out = out_base + m;
            if (m + 3 < M && (reinterpret_cast<uintptr_t>(out) & 15u) == 0) {
                *reinterpret_cast<float4 *>(out) = make_float4(
                    acc[ih * 4 + 0][j] * s, acc[ih * 4 + 1][j] * s,
                    acc[ih * 4 + 2][j] * s, acc[ih * 4 + 3][j] * s);
            } else {
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    if (m + ii < M) {
                        out[ii] = acc[ih * 4 + ii][j] * s;
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(W16, X16, cs, ids_dst, tile_spec, x_scale, dst, s1, s2, M, K, n_expert_used, m_blocks);
#endif
}

// ===================================================================================================
// Path 2: Zero-VRAM on-the-fly dequant grouped GEMM (used when VRAM is tight or batch is sparse).
//   BN = 16: BM =  64, 128 threads (16x8  -> 4x2 tile/thread, 4 blocks/SM)
//   BN = 32: BM =  64, 128 threads (16x8  -> 4x4 tile/thread, 4 blocks/SM)
//   BN = 64: BM = 128, 256 threads (16x16 -> 8x4 tile/thread, 2 blocks/SM)
// ===================================================================================================
template <ggml_type T, int BN>
__global__ void __launch_bounds__(BN <= 32 ? 128 : 256, BN <= 32 ? 4 : 2) mmid_f16_otf_kernel(
        const uint8_t * __restrict__ W,
        const size_t                 nb01,
        const size_t                 nb02,
        const half * __restrict__    X16,
        const float * __restrict__   cs,
        const int32_t * __restrict__ ids_dst,
        const ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const float * __restrict__   x_scale,
        float * __restrict__         dst,
        const int64_t                s1,
        const int64_t                s2,
        const int                    M,
        const int                    K,
        const int                    n_expert_used,
        const int                    m_blocks) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
    using tr = mmid_f16_t<T>;
    constexpr int BM     = (BN <= 32) ? 64  : 128;
    constexpr int TX_DIM = (BN <= 32) ? 8   : 16;
    constexpr int TM     = BM / 16;             // 4 for BN=16/32, 8 for BN=64
    constexpr int TN     = BN / TX_DIM;         // 2 for BN=16, 4 for BN=32/64
    constexpr int BS_PAD = (BN == 16) ? 18 : BN;
    constexpr int SUBS   = tr::QK / 8;

    const int bid      = blockIdx.x;
    const int tile_idx = bid / m_blocks;
    const int m_idx    = bid - tile_idx * m_blocks;

    const ggml_cuda_mm_id_tile_spec ts = tile_spec[tile_idx];
    const int expert   = ts.expert;
    const int col_low  = ts.col_low;
    const int col_high = ts.col_high;
    if (col_low >= col_high) {
        return;
    }

    __shared__ __align__(16) uint32_t As[2][MMID_BK2][BM];
    __shared__ __align__(16) uint32_t Bs[2][MMID_BK2][BS_PAD];

    const int t  = threadIdx.x;
    const int tx = t & (TX_DIM - 1);
    const int ty = t / TX_DIM; // 0..15
    const int m0 = m_idx * BM;
    const int ncols_tile = col_high - col_low;

    const uint8_t * W_exp = W + (size_t) expert * nb02;

    float acc[TM][TN];
    half2 h[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
            h[i][j]   = make_half2(0.0f, 0.0f);
        }
    }

    const int wr    = t >> 1;
    const int wc0   = (t & 1) << 1; // 0 or 2
    const bool wr_ok = (m0 + wr < M);
    const uint8_t * W_row = W_exp + (size_t) (m0 + wr) * nb01;

    uint4 ra0, ra1;
    uint4 rb0;
    const bool bt0 = (BN != 16) || (t < 64);

    auto gload = [&](const int k0) {
        if (wr_ok) {
            const int ch0  = (k0 >> 3) + wc0;
            const int blk  = ch0 / SUBS;
            const int sub0 = ch0 % SUBS;
            __half2 wh0[4], wh1[4];
            tr::w16(W_row + (size_t) blk * tr::BS, sub0, wh0, wh1);
            ra0 = make_uint4(*reinterpret_cast<const uint32_t *>(&wh0[0]),
                             *reinterpret_cast<const uint32_t *>(&wh0[1]),
                             *reinterpret_cast<const uint32_t *>(&wh0[2]),
                             *reinterpret_cast<const uint32_t *>(&wh0[3]));
            ra1 = make_uint4(*reinterpret_cast<const uint32_t *>(&wh1[0]),
                             *reinterpret_cast<const uint32_t *>(&wh1[1]),
                             *reinterpret_cast<const uint32_t *>(&wh1[2]),
                             *reinterpret_cast<const uint32_t *>(&wh1[3]));
        } else {
            ra0 = make_uint4(0u, 0u, 0u, 0u);
            ra1 = make_uint4(0u, 0u, 0u, 0u);
        }
        if (bt0) {
            const int r = t >> 2;
            const int c = t & 3;
            rb0 = (r < ncols_tile)
                ? __ldg(reinterpret_cast<const uint4 *>(X16 + (int64_t) (col_low + r) * K + k0 + c * 8))
                : make_uint4(0u, 0u, 0u, 0u);
        }
    };

    auto sstore = [&](const int buf) {
        const int rs0 = wr ^ (wc0 << 3);
        const int rs1 = wr ^ ((wc0 + 1) << 3);
        As[buf][wc0 * 4 + 0][rs0]       = ra0.x;
        As[buf][wc0 * 4 + 1][rs0]       = ra0.y;
        As[buf][wc0 * 4 + 2][rs0]       = ra0.z;
        As[buf][wc0 * 4 + 3][rs0]       = ra0.w;
        As[buf][(wc0 + 1) * 4 + 0][rs1] = ra1.x;
        As[buf][(wc0 + 1) * 4 + 1][rs1] = ra1.y;
        As[buf][(wc0 + 1) * 4 + 2][rs1] = ra1.z;
        As[buf][(wc0 + 1) * 4 + 3][rs1] = ra1.w;
        if (bt0) {
            const int r  = t >> 2;
            const int c  = t & 3;
            const int rs = (BN == 16) ? r : (r ^ (c << 3));
            Bs[buf][c * 4 + 0][rs] = rb0.x;
            Bs[buf][c * 4 + 1][rs] = rb0.y;
            Bs[buf][c * 4 + 2][rs] = rb0.z;
            Bs[buf][c * 4 + 3][rs] = rb0.w;
        }
    };

    const int nt = K / MMID_BK;
    gload(0);
    sstore(0);
    __syncthreads();

    for (int it = 0; it < nt; ++it) {
        const int buf = it & 1;
        if (it + 1 < nt) {
            gload((it + 1) * MMID_BK);
        }
        const bool restart = (it % MMID_FOLD_TILES) == 0;
#pragma unroll
        for (int k2 = 0; k2 < MMID_BK2; ++k2) {
            const int sw = (k2 >> 2) << 3;
            uint32_t a[TM];
            const uint4 a0 = *reinterpret_cast<const uint4 *>(&As[buf][k2][(ty * 4) ^ sw]);
            a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
            if constexpr (TM == 8) {
                const uint4 a1 = *reinterpret_cast<const uint4 *>(&As[buf][k2][(64 + ty * 4) ^ sw]);
                a[4] = a1.x; a[5] = a1.y; a[6] = a1.z; a[7] = a1.w;
            }
            uint32_t b[TN];
            if constexpr (BN == 64 || BN == 32) {
                const uint4 b0 = *reinterpret_cast<const uint4 *>(&Bs[buf][k2][(tx * 4) ^ sw]);
                b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
            } else {
                const uint2 b0 = *reinterpret_cast<const uint2 *>(&Bs[buf][k2][tx * 2]);
                b[0] = b0.x; b[1] = b0.y;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const half2 ai = *reinterpret_cast<const half2 *>(&a[i]);
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const half2 bj = *reinterpret_cast<const half2 *>(&b[j]);
                    h[i][j] = (k2 == 0 && restart) ? __hmul2(ai, bj) : __hfma2(ai, bj, h[i][j]);
                }
            }
        }
        if (((it + 1) % MMID_FOLD_TILES) == 0 || it + 1 == nt) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += mmid_fold_h(h[i][j]);
                }
            }
        }
        if (it + 1 < nt) {
            sstore(buf ^ 1);
        }
        __syncthreads();
    }

    const float escale = (x_scale ? __ldg(x_scale + expert) : 1.0f) * (1.0f / tr::WSCALE);
    constexpr int NH = TM / 4;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int col_off = tx * TN + j;
        if (col_off >= ncols_tile) {
            continue;
        }
        const int col = col_low + col_off;
        const float s = cs[col] * escale;
        const int32_t d_id = ids_dst[col];
        const int32_t tok  = d_id / n_expert_used;
        const int32_t iex  = d_id - tok * n_expert_used;
        float * out_base = dst + (int64_t) tok * s2 + (int64_t) iex * s1;
#pragma unroll
        for (int ih = 0; ih < NH; ++ih) {
            const int m = m0 + ih * 64 + ty * 4;
            if (m >= M) {
                continue;
            }
            float * out = out_base + m;
            if (m + 3 < M && (reinterpret_cast<uintptr_t>(out) & 15u) == 0) {
                *reinterpret_cast<float4 *>(out) = make_float4(
                    acc[ih * 4 + 0][j] * s, acc[ih * 4 + 1][j] * s,
                    acc[ih * 4 + 2][j] * s, acc[ih * 4 + 3][j] * s);
            } else {
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    if (m + ii < M) {
                        out[ii] = acc[ih * 4 + ii][j] * s;
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(W, nb01, nb02, X16, cs, ids_dst, tile_spec, x_scale, dst, s1, s2, M, K, n_expert_used, m_blocks);
#endif
}

static bool mmid_f16_otf_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            return true;
        default:
            return false;
    }
}

template <int BN>
static void mmid_f16_otf_launch(
        const ggml_type type,
        const unsigned int grid_x,
        const uint8_t * W, const size_t nb01, const size_t nb02,
        const half * X16, const float * cs,
        const int32_t * ids_dst, const ggml_cuda_mm_id_tile_spec * tile_spec,
        const float * x_scale, float * dst,
        const int64_t s1, const int64_t s2,
        const int M, const int K, const int n_expert_used, const int m_blocks,
        cudaStream_t stream) {
    constexpr int NT = (BN <= 32) ? 128 : 256;
#define MMID_CASE(GT) \
    case GT: \
        mmid_f16_otf_kernel<GT, BN><<<grid_x, NT, 0, stream>>>( \
            W, nb01, nb02, X16, cs, ids_dst, tile_spec, x_scale, dst, s1, s2, M, K, n_expert_used, m_blocks); \
        break;
    switch (type) {
        MMID_CASE(GGML_TYPE_Q4_0)
        MMID_CASE(GGML_TYPE_Q4_1)
        MMID_CASE(GGML_TYPE_Q5_0)
        MMID_CASE(GGML_TYPE_Q5_1)
        MMID_CASE(GGML_TYPE_Q8_0)
        MMID_CASE(GGML_TYPE_Q2_K)
        MMID_CASE(GGML_TYPE_Q3_K)
        MMID_CASE(GGML_TYPE_Q4_K)
        MMID_CASE(GGML_TYPE_Q5_K)
        MMID_CASE(GGML_TYPE_Q6_K)
        MMID_CASE(GGML_TYPE_IQ4_NL)
        MMID_CASE(GGML_TYPE_IQ4_XS)
        MMID_CASE(GGML_TYPE_IQ3_S)
        MMID_CASE(GGML_TYPE_MXFP4)
        MMID_CASE(GGML_TYPE_NVFP4)
        default:
            GGML_ABORT("mmid_f16_p100: unsupported type %s", ggml_type_name(type));
    }
#undef MMID_CASE
}

} // namespace

bool ggml_cuda_mmid_f16_p100_try(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    static const bool enabled = [] {
        const char * s = std::getenv("GGML_CUDA_MMID_F16");
        return !s || std::atoi(s) != 0;
    }();
    if (!enabled) {
        return false;
    }

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    // Target GP100 (sm_60) specifically: fast native FP16 HFMA2, no __dp4a.
    if (cc != GGML_CUDA_CC_PASCAL || !fast_fp16_available(cc)) {
        return false;
    }
    if (!ids || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }
    if (fusion && (fusion->gate || fusion->x_bias || fusion->gate_bias || fusion->gate_scale ||
                   fusion->shared_up || fusion->shared_gate || fusion->shared_dst)) {
        return false;
    }
    const float * x_scale = nullptr;
    if (fusion && fusion->x_scale) {
        if (src0->type != GGML_TYPE_NVFP4 || fusion->x_scale->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(fusion->x_scale) || ggml_nelements(fusion->x_scale) != src0->ne[2]) {
            return false;
        }
        x_scale = static_cast<const float *>(fusion->x_scale->data);
    }

    const int64_t K  = src0->ne[0];
    const int64_t M  = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows   = ne12 * n_expert_used;

    if (K <= 0 || (K % MMID_BK) != 0 || (K % ggml_blck_size(src0->type)) != 0 ||
            M <= 0 || M > INT_MAX || K > INT_MAX ||
            ne02 <= 0 || ne12 <= 0 || ne_get_rows <= 0 || ne_get_rows > INT_MAX ||
            src0->ne[3] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1 ||
            src0->nb[0] != ggml_type_size(src0->type) ||
            src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            ids->nb[0]  != sizeof(int32_t) ||
            (src1->nb[2] % src1->nb[1]) != 0 || (dst->nb[2] % dst->nb[1]) != 0) {
        return false;
    }

    const int64_t ncols_opt = (ne_get_rows + ne02 - 1) / ne02;

    // Path 2 (zero-VRAM on-the-fly FP16x2 dequant GEMM) beats full-expert pre-dequant on all
    // 15 primary quant types because compressed weight traffic is 3-4x smaller than FP16.
    // Path 1 (ggml_get_to_fp16_cuda + pure FP16 GEMM) serves as automatic acceleration for all
    // other quant types (IQ1/IQ2/IQ3_XXS/Q1_0/Q2_0/FP8 and any future ggml quant types) when
    // VRAM allows, or can be forced via GGML_CUDA_MMID_PREDEQ=1.
    static const int predeq_env = [] {
        const char * s = std::getenv("GGML_CUDA_MMID_PREDEQ");
        return s ? std::atoi(s) : -1;
    }();
    static size_t w16_hwm[GGML_CUDA_MAX_DEVICES] = {0};

    const bool otf_ok = mmid_f16_otf_supported(src0->type);
    const to_fp16_cuda_t to_fp16 = ggml_is_contiguous(src0) ? ggml_get_to_fp16_cuda(src0->type) : nullptr;
    const size_t w16_nelem = (size_t) ne02 * (size_t) M * (size_t) K;
    const size_t w16_bytes = w16_nelem * sizeof(half);

    bool use_predeq = false;
    if (to_fp16 != nullptr && predeq_env != 0 && (predeq_env == 1 || !otf_ok)) {
        if (w16_bytes <= w16_hwm[id]) {
            use_predeq = true;
        } else {
            size_t free_b = 0, total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess &&
                    free_b + w16_hwm[id] >= w16_bytes + (512ULL << 20)) {
                use_predeq = true;
                w16_hwm[id] = w16_bytes;
            }
        }
    }

    if (!use_predeq && !otf_ok) {
        return false;
    }

    static const int bn_env = [] {
        const char * s = std::getenv("GGML_CUDA_MMID_F16_BN");
        return s ? std::atoi(s) : 0;
    }();
    int BN;
    if (use_predeq) {
        BN = (ncols_opt <= 16) ? 16 : (ncols_opt <= 40 ? 32 : (ncols_opt <= 96 ? 64 : 128));
        if (bn_env == 16 || bn_env == 32 || bn_env == 64 || bn_env == 128) {
            BN = bn_env;
        }
    } else {
        BN = (ncols_opt <= 8) ? 16 : (ncols_opt <= 96 ? 32 : 64);
        if (bn_env == 16 || bn_env == 32 || bn_env == 64) {
            BN = bn_env;
        }
    }

    const int64_t BM       = (BN <= 32) ? 64 : 128;
    const int64_t m_blocks = (M + BM - 1) / BM;
    const int64_t ntj_64   = (ne_get_rows + BN - 1) / BN + ne02;
    const int64_t grid_64  = ntj_64 * m_blocks;
    if (grid_64 <= 0 || grid_64 > INT_MAX) {
        return false;
    }

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    const int ntj = (int) ntj_64;
    ggml_cuda_pool_alloc<ggml_cuda_mm_id_tile_spec> tile_spec(ctx.pool(), ntj);

    const bool dedup_bcast = (ne11 == 1 && n_expert_used > 1);
    const int si1  = (int) (ids->nb[1] / sizeof(int32_t));
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);

    ggml_cuda_launch_mm_ids_helper(
        static_cast<const int32_t *>(ids->data),
        ids_src1.get(), ids_dst.get(), tile_spec.get(),
        (int) ne02, (int) ne12, (int) n_expert_used, (int) ne11, BN,
        si1, sis1, dedup_bcast, stream);
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<half>  x16_alloc(ctx.pool(), (size_t) ne_get_rows * (size_t) K);
    ggml_cuda_pool_alloc<float> cs_alloc(ctx.pool(), (size_t) ne_get_rows);
    const int xexp = (use_predeq && src0->type != GGML_TYPE_NVFP4) ? 3 : 1;

    if (dedup_bcast) {
        mmid_f16_prescale_scatter<<<(unsigned int) ne12, 256, 0, stream>>>(
            static_cast<const char *>(src1->data), src1->nb[2],
            ids_src1.get(), x16_alloc.get(), cs_alloc.get(),
            (int) K, (int) n_expert_used, xexp);
    } else {
        mmid_f16_prescale_gather<<<(unsigned int) ne_get_rows, 256, 0, stream>>>(
            static_cast<const char *>(src1->data), src1->nb[1],
            ids_src1.get(), x16_alloc.get(), cs_alloc.get(),
            (int) K, xexp);
    }
    CUDA_CHECK(cudaGetLastError());

    const unsigned int grid_x = (unsigned int) grid_64;
    const int64_t s1 = (int64_t) (dst->nb[1] / sizeof(float));
    const int64_t s2 = (int64_t) (dst->nb[2] / sizeof(float));
    float * dst_d = static_cast<float *>(dst->data);

    if (use_predeq) {
        ggml_cuda_pool_alloc<half> w16_alloc(ctx.pool(), w16_nelem);
        to_fp16(src0->data, w16_alloc.get(), (int64_t) w16_nelem, stream);
        CUDA_CHECK(cudaGetLastError());

        if (BN == 16) {
            mmid_f16_pure_kernel<16><<<grid_x, 128, 0, stream>>>(
                w16_alloc.get(), x16_alloc.get(), cs_alloc.get(), ids_dst.get(), tile_spec.get(),
                x_scale, dst_d, s1, s2, (int) M, (int) K, (int) n_expert_used, (int) m_blocks);
        } else if (BN == 32) {
            mmid_f16_pure_kernel<32><<<grid_x, 128, 0, stream>>>(
                w16_alloc.get(), x16_alloc.get(), cs_alloc.get(), ids_dst.get(), tile_spec.get(),
                x_scale, dst_d, s1, s2, (int) M, (int) K, (int) n_expert_used, (int) m_blocks);
        } else if (BN == 64) {
            mmid_f16_pure_kernel<64><<<grid_x, 256, 0, stream>>>(
                w16_alloc.get(), x16_alloc.get(), cs_alloc.get(), ids_dst.get(), tile_spec.get(),
                x_scale, dst_d, s1, s2, (int) M, (int) K, (int) n_expert_used, (int) m_blocks);
        } else {
            mmid_f16_pure_kernel<128><<<grid_x, 256, 0, stream>>>(
                w16_alloc.get(), x16_alloc.get(), cs_alloc.get(), ids_dst.get(), tile_spec.get(),
                x_scale, dst_d, s1, s2, (int) M, (int) K, (int) n_expert_used, (int) m_blocks);
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    const uint8_t * W = static_cast<const uint8_t *>(src0->data);
    if (BN == 16) {
        mmid_f16_otf_launch<16>(
            src0->type, grid_x, W, src0->nb[1], src0->nb[2],
            x16_alloc.get(), cs_alloc.get(), ids_dst.get(), tile_spec.get(),
            x_scale, dst_d, s1, s2, (int) M, (int) K, (int) n_expert_used, (int) m_blocks, stream);
    } else if (BN == 32) {
        mmid_f16_otf_launch<32>(
            src0->type, grid_x, W, src0->nb[1], src0->nb[2],
            x16_alloc.get(), cs_alloc.get(), ids_dst.get(), tile_spec.get(),
            x_scale, dst_d, s1, s2, (int) M, (int) K, (int) n_expert_used, (int) m_blocks, stream);
    } else {
        mmid_f16_otf_launch<64>(
            src0->type, grid_x, W, src0->nb[1], src0->nb[2],
            x16_alloc.get(), cs_alloc.get(), ids_dst.get(), tile_spec.get(),
            x_scale, dst_d, s1, s2, (int) M, (int) K, (int) n_expert_used, (int) m_blocks, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_mmid_f16_p100_try(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_UNUSED_VARS(ctx, src0, src1, ids, dst, fusion);
    return false;
}

#endif
