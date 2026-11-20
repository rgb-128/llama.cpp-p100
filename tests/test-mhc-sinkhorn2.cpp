// CPU comparison against the original ggml graph, including strided logits and
// the [1, n_tokens] views broadcast by SenseNova Loop's build_merge.
// Standalone build (after building ggml into build-cpu):
// c++ -std=c++17 -Iggml/include tests/test-mhc-sinkhorn2.cpp -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lggml -lggml-cpu -lggml-base -o build-cpu/bin/test-mhc-sinkhorn2
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static bool check(ggml_backend_t backend, int64_t n_rows, int32_t n_iter, bool view, int n_threads) {
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead()*2048 + ggml_graph_overhead(),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx);
    ggml_tensor * logits = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, view ? 8 : 4, n_rows);
    ggml_tensor * mix = view ? ggml_view_2d(ctx.get(), logits, 4, n_rows, logits->nb[1], 4*sizeof(float)) : logits;
    ggml_tensor * fused = ggml_mhc_sinkhorn2(ctx.get(), mix, n_iter);
    GGML_ASSERT(fused != mix && ggml_is_contiguous(fused) && ggml_are_same_shape(fused, mix));

    std::array<ggml_tensor *, 4> old;
    for (int i = 0; i < 4; ++i) {
        old[i] = ggml_view_2d(ctx.get(), mix, 1, n_rows, mix->nb[1], i*sizeof(float));
    }
    auto normalize = [&](ggml_tensor * & x, ggml_tensor * & y) {
        ggml_tensor * delta = ggml_sub(ctx.get(), x, y);
        x = ggml_neg(ctx.get(), ggml_softplus(ctx.get(), ggml_neg(ctx.get(), delta)));
        y = ggml_neg(ctx.get(), ggml_softplus(ctx.get(), delta));
    };
    for (int32_t i = 0; i < n_iter; ++i) {
        normalize(old[0], old[1]);
        normalize(old[2], old[3]);
        normalize(old[0], old[2]);
        normalize(old[1], old[3]);
    }
    for (auto & x : old) {
        x = ggml_exp(ctx.get(), x);
    }
    ggml_tensor * reference = ggml_concat(ctx.get(),
        ggml_concat(ctx.get(), old[0], old[1], 0), ggml_concat(ctx.get(), old[2], old[3], 0), 0);

    // Exercise the output views and the same broadcasts as build_merge.
    ggml_tensor * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 13, n_rows);
    ggml_tensor * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 13, n_rows);
    std::array<ggml_tensor *, 2> merged, merged_ref;
    for (int i = 0; i < 2; ++i) {
        ggml_tensor * x = ggml_view_2d(ctx.get(), fused, 1, n_rows, fused->nb[1], (2*i + 0)*sizeof(float));
        ggml_tensor * y = ggml_view_2d(ctx.get(), fused, 1, n_rows, fused->nb[1], (2*i + 1)*sizeof(float));
        merged[i] = ggml_add(ctx.get(), ggml_mul(ctx.get(), a, x), ggml_mul(ctx.get(), b, y));
        merged_ref[i] = ggml_add(ctx.get(), ggml_mul(ctx.get(), a, old[2*i]), ggml_mul(ctx.get(), b, old[2*i + 1]));
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, fused);
    ggml_build_forward_expand(graph, reference);
    for (int i = 0; i < 2; ++i) {
        ggml_build_forward_expand(graph, merged[i]);
        ggml_build_forward_expand(graph, merged_ref[i]);
    }
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-80.0f, 80.0f);
    std::vector<float> input(ggml_nelements(logits));
    for (float & x : input) {
        x = dist(rng);
    }
    // Guarantee saturation/underflow coverage even for the one-row case.
    const size_t offset = view ? 4 : 0;
    input[offset + 0] = -80.0f;
    input[offset + 1] =  80.0f;
    input[offset + 2] =  80.0f;
    input[offset + 3] = -80.0f;
    ggml_backend_tensor_set(logits, input.data(), 0, input.size()*sizeof(float));
    std::vector<float> streams(ggml_nelements(a));
    for (ggml_tensor * stream : {a, b}) {
        for (float & x : streams) {
            x = dist(rng) / 80.0f;
        }
        ggml_backend_tensor_set(stream, streams.data(), 0, streams.size()*sizeof(float));
    }
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    bool identical = true;
    float max_abs_diff = 0.0f;
    auto compare = [&](ggml_tensor * x, ggml_tensor * y) {
        GGML_ASSERT(ggml_are_same_shape(x, y) && ggml_is_contiguous(x) && ggml_is_contiguous(y));
        std::vector<float> actual(ggml_nelements(x)), expected(actual.size());
        ggml_backend_tensor_get(x, actual.data(), 0, actual.size()*sizeof(float));
        ggml_backend_tensor_get(y, expected.data(), 0, expected.size()*sizeof(float));
        identical &= std::memcmp(actual.data(), expected.data(), actual.size()*sizeof(float)) == 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            identical &= std::isfinite(actual[i]) && std::isfinite(expected[i]);
            max_abs_diff = std::max(max_abs_diff, std::fabs(actual[i] - expected[i]));
        }
    };
    compare(fused, reference);
    for (int i = 0; i < 2; ++i) {
        compare(merged[i], merged_ref[i]);
    }
    std::vector<float> after(input.size());
    ggml_backend_tensor_get(logits, after.data(), 0, after.size()*sizeof(float));
    identical &= std::memcmp(input.data(), after.data(), input.size()*sizeof(float)) == 0;
    printf("n_rows=%lld n_iter=%d view=%d threads=%d: max abs diff = %.9g, bit-identical = %s\n",
           (long long) n_rows, n_iter, view, n_threads, max_abs_diff, identical ? "yes" : "NO");
    return identical;
}

int main() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    GGML_ASSERT(backend);
    bool ok = true;
    for (int n_threads : {1, 4}) {
        for (int64_t n_rows : {1, 7, 32, 257}) {
            for (int32_t n_iter : {0, 1, 3, 20}) {
                for (bool view : {false, true}) {
                    ok &= check(backend.get(), n_rows, n_iter, view, n_threads);
                }
            }
        }
    }
    return ok ? 0 : 1;
}
