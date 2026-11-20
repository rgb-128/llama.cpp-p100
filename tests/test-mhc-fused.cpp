// Bitwise CPU comparison of the complete two-stream mHC route, including the
// original Sinkhorn loop, output slice views, and last-layer get_rows cropping.
// Standalone build: link this file with ggml, ggml-cpu and ggml-base (as for
// test-mhc-sinkhorn2.cpp). No model files or CUDA toolkit are required.
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

struct route {
    ggml_tensor * input;
    std::array<ggml_tensor *, 2> out;
    ggml_cgraph * graph;
};

// mode 0: LOOP_MHC_UNFUSED=1; 1: previous Sinkhorn-only fusion; 2: all fused.
static route build(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, ggml_tensor * branch,
                   ggml_tensor * logits, ggml_tensor * ids, bool with_mix, int n_iter, float scale, int mode) {
    auto row = [&](ggml_tensor * t, int i) {
        return ggml_view_2d(ctx, t, 1, t->ne[1], t->nb[1], i*sizeof(float));
    };
    auto crop = [&](ggml_tensor * t) { return ids ? ggml_get_rows(ctx, t, ids) : t; };
    route result{};
    result.input = mode == 2 ? ggml_mhc_pre(ctx, a, b, logits, scale) :
        ggml_scale(ctx, ggml_add(ctx, ggml_mul(ctx, a, ggml_sigmoid(ctx, row(logits, 0))),
                                     ggml_mul(ctx, b, ggml_sigmoid(ctx, row(logits, 1)))), scale);
    std::array<ggml_tensor *, 4> m{};
    ggml_tensor * mix = nullptr;
    if (with_mix) {
        if (mode == 0) {
            for (int i = 0; i < 4; ++i) {
                m[i] = row(logits, 4 + i);
            }
            auto norm = [&](ggml_tensor * & x, ggml_tensor * & y) {
                ggml_tensor * d = ggml_sub(ctx, x, y);
                x = ggml_neg(ctx, ggml_softplus(ctx, ggml_neg(ctx, d)));
                y = ggml_neg(ctx, ggml_softplus(ctx, d));
            };
            for (int i = 0; i < n_iter; ++i) {
                norm(m[0], m[1]); norm(m[2], m[3]); norm(m[0], m[2]); norm(m[1], m[3]);
            }
            for (auto & x : m) { x = ggml_exp(ctx, x); }
        } else {
            mix = ggml_mhc_sinkhorn2(ctx, ggml_view_2d(ctx, logits, 4, a->ne[1], logits->nb[1], 4*sizeof(float)), n_iter);
            if (mode == 1) {
                for (int i = 0; i < 4; ++i) { m[i] = row(mix, i); }
            }
        }
    }
    if (mode == 2) {
        ggml_tensor * merged = ggml_mhc_merge(ctx, crop(a), crop(b), crop(branch), crop(logits), mix ? crop(mix) : nullptr);
        for (int i = 0; i < 2; ++i) {
            result.out[i] = ggml_view_2d(ctx, merged, merged->ne[0], merged->ne[1], merged->nb[1], i*merged->nb[2]);
        }
    } else {
        ggml_tensor * p0 = crop(ggml_scale(ctx, ggml_sigmoid(ctx, row(logits, 2)), 2.0f));
        ggml_tensor * p1 = crop(ggml_scale(ctx, ggml_sigmoid(ctx, row(logits, 3)), 2.0f));
        a = crop(a); b = crop(b); branch = crop(branch);
        if (with_mix) { for (auto & x : m) { x = crop(x); } }
        ggml_tensor * na = with_mix ? ggml_add(ctx, ggml_mul(ctx, a, m[0]), ggml_mul(ctx, b, m[1])) : a;
        ggml_tensor * nb = with_mix ? ggml_add(ctx, ggml_mul(ctx, a, m[2]), ggml_mul(ctx, b, m[3])) : b;
        result.out[0] = ggml_add(ctx, na, ggml_mul(ctx, branch, p0));
        result.out[1] = ggml_add(ctx, nb, ggml_mul(ctx, branch, p1));
    }
    result.graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(result.graph, result.input);
    for (auto * out : result.out) { ggml_build_forward_expand(result.graph, out); }
    return result;
}

static int compute_nodes(ggml_cgraph * graph) {
    int n = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto op = ggml_graph_node(graph, i)->op;
        n += op != GGML_OP_VIEW && op != GGML_OP_RESHAPE && op != GGML_OP_PERMUTE && op != GGML_OP_TRANSPOSE;
    }
    return n;
}

static bool check(ggml_backend_t backend, int64_t n_embd, int64_t n_tok, bool with_mix, int64_t n_logits,
                  int n_threads, bool cropped, int n_iter, float scale) {
    ggml_init_params params{ggml_tensor_overhead()*4096 + 4*ggml_graph_overhead(), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx);
    // Contiguous slices at different offsets, as produced by the preceding merge.
    auto * streams = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_embd, n_tok, 2);
    auto * a = ggml_view_2d(ctx.get(), streams, n_embd, n_tok, streams->nb[1], 0);
    auto * b = ggml_view_2d(ctx.get(), streams, n_embd, n_tok, streams->nb[1], streams->nb[2]);
    auto * branch = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_tok);
    auto * logits = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_logits, n_tok);
    const int64_t n_out = (n_tok + 1)/2;
    auto * ids = cropped ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_out) : nullptr;
    std::array<route, 3> routes;
    for (int mode = 0; mode < 3; ++mode) {
        routes[mode] = build(ctx.get(), a, b, branch, logits, ids, with_mix, n_iter, scale, mode);
    }
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<std::vector<float>> originals;
    for (auto * t : {streams, branch, logits}) {
        std::vector<float> data(ggml_nelements(t));
        for (auto & x : data) { x = dist(rng); }
        // Saturated sigmoid and strongly asymmetric mixing, plus ordinary logits.
        if (t == logits) {
            data[0] = -80.0f; data[1] = 80.0f;
            data[2] = 80.0f; data[3] = -80.0f;
            if (with_mix) {
                data[4] = -80.0f; data[5] = 80.0f; data[6] = 80.0f; data[7] = -80.0f;
            }
        }
        ggml_backend_tensor_set(t, data.data(), 0, data.size()*sizeof(float));
        originals.push_back(std::move(data));
    }
    if (ids) {
        std::vector<int32_t> indices(n_out);
        for (int64_t i = 0; i < n_out; ++i) { indices[i] = n_tok - 1 - 2*i; }
        ggml_backend_tensor_set(ids, indices.data(), 0, indices.size()*sizeof(int32_t));
    }
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    bool ok = true;
    for (const auto & route : routes) {
        for (int i = 0; i < ggml_graph_n_nodes(route.graph); ++i) {
            ok &= ggml_backend_supports_op(backend, ggml_graph_node(route.graph, i));
        }
        GGML_ASSERT(ggml_backend_graph_compute(backend, route.graph) == GGML_STATUS_SUCCESS);
    }
    float max_diff = 0.0f;
    auto compare = [&](ggml_tensor * actual, ggml_tensor * expected) {
        GGML_ASSERT(ggml_are_same_shape(actual, expected) && ggml_is_contiguous(actual) && ggml_is_contiguous(expected));
        std::vector<float> x(ggml_nelements(actual)), y(x.size());
        ggml_backend_tensor_get(actual, x.data(), 0, x.size()*sizeof(float));
        ggml_backend_tensor_get(expected, y.data(), 0, y.size()*sizeof(float));
        ok &= std::memcmp(x.data(), y.data(), x.size()*sizeof(float)) == 0;
        for (size_t i = 0; i < x.size(); ++i) {
            ok &= std::isfinite(x[i]) && std::isfinite(y[i]);
            max_diff = std::max(max_diff, std::fabs(x[i] - y[i]));
        }
    };
    for (int mode : {1, 2}) {
        compare(routes[mode].input, routes[0].input);
        for (int i = 0; i < 2; ++i) { compare(routes[mode].out[i], routes[0].out[i]); }
    }
    size_t j = 0;
    for (auto * t : {streams, branch, logits}) {
        std::vector<float> data(ggml_nelements(t));
        ggml_backend_tensor_get(t, data.data(), 0, data.size()*sizeof(float));
        ok &= std::memcmp(data.data(), originals[j++].data(), data.size()*sizeof(float)) == 0;
    }
    printf("embd=%lld tok=%lld mix=%d logits=%lld threads=%d crop=%d iter=%d scale=%.7g: max abs diff=%.9g bit-identical=%s\n",
           (long long) n_embd, (long long) n_tok, with_mix, (long long) n_logits,
           n_threads, cropped, n_iter, scale, max_diff, ok ? "yes" : "NO");
    if (n_embd == 13 && n_tok == 5 && n_threads == 1 && !cropped && n_iter == 3 && scale == 0.5f) {
        printf("  route nodes (unfused / Sinkhorn-only / all fused): %d / %d / %d; compute nodes: %d / %d / %d\n",
               ggml_graph_n_nodes(routes[0].graph), ggml_graph_n_nodes(routes[1].graph), ggml_graph_n_nodes(routes[2].graph),
               compute_nodes(routes[0].graph), compute_nodes(routes[1].graph), compute_nodes(routes[2].graph));
    }
    return ok;
}

static void check_support(ggml_backend_t backend) {
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead()*32, nullptr, true}));
    auto * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 13, 5);
    auto * b = ggml_dup_tensor(ctx.get(), a);
    auto * branch = ggml_dup_tensor(ctx.get(), a);
    auto * logits = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 8, 5);
    auto * mix = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 5);
    auto * pre = ggml_mhc_pre(ctx.get(), a, b, logits, 0.5f);
    auto * merge = ggml_mhc_merge(ctx.get(), a, b, branch, logits, mix);
    auto rejects = [&](ggml_tensor * t, auto & field, auto bad) {
        auto saved = field;
        field = bad;
        GGML_ASSERT(!ggml_backend_supports_op(backend, t));
        field = saved;
    };
    GGML_ASSERT(ggml_backend_supports_op(backend, pre) && ggml_backend_supports_op(backend, merge));
    for (auto * t : {pre, merge}) {
        rejects(t, a->type, GGML_TYPE_F16);
        rejects(t, b->ne[1], int64_t(4));
        rejects(t, a->nb[1], size_t(56));
        rejects(t, logits->ne[0], int64_t(7));
        rejects(t, logits->ne[1], int64_t(4));
        rejects(t, logits->nb[1], size_t(36));
        rejects(t, t->type, GGML_TYPE_F16);
        rejects(t, t->ne[1], int64_t(4));
    }
    rejects(merge, branch->type, GGML_TYPE_F16);
    rejects(merge, branch->ne[0], int64_t(12));
    rejects(merge, mix->type, GGML_TYPE_F16);
    rejects(merge, mix->ne[0], int64_t(3));
    rejects(merge, mix->ne[1], int64_t(4));
    rejects(merge, mix->nb[1], size_t(20));
    rejects(merge, merge->ne[2], int64_t(1));
    puts("supports_op: invalid types, shapes, strides rejected");
}

int main() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    GGML_ASSERT(backend);
    check_support(backend.get());
    bool ok = true;
    for (int n_threads : {1, 4}) {
        for (int64_t n_embd : {13, 4096}) {
            for (int64_t n_tok : {1, 2, 5, 33}) {
                for (bool cropped : {false, true}) {
                    for (float scale : {0.5f, 1.234567f}) {
                        for (int n_iter : {1, 3, 20}) {
                            ok &= check(backend.get(), n_embd, n_tok, true, 8, n_threads, cropped, n_iter, scale);
                        }
                        ok &= check(backend.get(), n_embd, n_tok, false, 8, n_threads, cropped, 3, scale);
                        ok &= check(backend.get(), n_embd, n_tok, false, 4, n_threads, cropped, 3, scale);
                    }
                }
            }
        }
    }
    return ok ? 0 : 1;
}
