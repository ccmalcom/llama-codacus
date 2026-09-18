#include "ggml-moe-h2d-stats.h"
#include "ggml-impl.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <vector>

namespace {

struct stats {
    double   bucket[GGML_MOE_H2D_N_BUCKETS] = {0, 0, 0, 0};
    uint64_t copies    = 0;
    uint64_t ranges    = 0;
    uint64_t bytes     = 0;
    uint64_t passes    = 0;
    double   pass_secs = 0;
    // per-copy h2d time, kept so the distribution can be reported rather than just a mean
    std::vector<float> copy_ms;
    std::vector<float> range_kb;
    // routing digest
    uint64_t digest       = 0xcbf29ce484222325ull;
    uint64_t digest_folds = 0;
    uint64_t experts_sum  = 0;
    // destination aliasing: the last few copy destinations, and how often a copy's
    // destination overlapped one of the k copies before it
    const void * prev_dst[4]  = {nullptr, nullptr, nullptr, nullptr};
    size_t       prev_size[4] = {0, 0, 0, 0};
    uint64_t     overlap_at[4] = {0, 0, 0, 0};
    uint64_t     distinct_checked = 0;
    uint64_t     drain_after_slot = 0;
};

stats g;

int env_flag(const char * name) {
    const char * s = getenv(name);
    return s ? atoi(s) : 0;
}

int stats_on  = -1;
int digest_on = -1;

inline void fnv(uint64_t & h, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        h ^= (v >> (i * 8)) & 0xff;
        h *= 0x100000001b3ull;
    }
}

const char * bucket_name(int b) {
    switch (b) {
        case GGML_MOE_H2D_WAIT_PREV: return "wait_prev_compute";
        case GGML_MOE_H2D_IDS:       return "ids_readback";
        case GGML_MOE_H2D_PREFETCH:  return "prefetch_wait";
        case GGML_MOE_H2D_COPY:      return "h2d_issue";
    }
    return "?";
}

float pct(const std::vector<float> & v, double q) {
    if (v.empty()) return 0;
    size_t i = (size_t)(q * (v.size() - 1));
    return v[i];
}

bool reported = false;

// llama-server does not reach ggml_backend_sched_free on SIGTERM, so the report has to
// come out of a static destructor the way the prefetch pool's counters do. Declared
// after `g` so it is destroyed before it; the report call in ggml_backend_sched_free is
// kept for callers that do free cleanly, and `reported` makes the second one a no-op.
struct reporter {
    ~reporter() { ggml_moe_h2d_report(); }
};
reporter g_reporter;

} // namespace

int ggml_moe_h2d_stats_enabled(void) {
    if (stats_on < 0) {
        stats_on = env_flag("GGML_MOE_H2D_STATS");
    }
    return stats_on;
}

int ggml_moe_route_digest_enabled(void) {
    if (digest_on < 0) {
        digest_on = env_flag("GGML_MOE_ROUTE_DIGEST");
    }
    return digest_on;
}

double ggml_moe_h2d_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

void ggml_moe_h2d_add(enum ggml_moe_h2d_bucket b, double secs) {
    g.bucket[b] += secs;
    if (b == GGML_MOE_H2D_COPY) {
        g.copy_ms.push_back((float) (secs * 1e3));
    }
}

void ggml_moe_h2d_copy_done(size_t n_ranges, size_t bytes,
                            const void * dst, size_t dst_bytes) {
    g.copies += 1;
    g.ranges += n_ranges;
    g.bytes  += bytes;

    const char * a0 = (const char *) dst;
    const char * a1 = a0 + dst_bytes;
    for (int k = 0; k < 4; k++) {
        if (!g.prev_dst[k]) {
            continue;
        }
        const char * b0 = (const char *) g.prev_dst[k];
        const char * b1 = b0 + g.prev_size[k];
        if (a0 < b1 && b0 < a1) {
            g.overlap_at[k] += 1;
        }
    }
    g.distinct_checked += 1;
    for (int k = 3; k > 0; k--) {
        g.prev_dst[k]  = g.prev_dst[k - 1];
        g.prev_size[k] = g.prev_size[k - 1];
    }
    g.prev_dst[0]  = dst;
    g.prev_size[0] = dst_bytes;
}

void ggml_moe_h2d_note_drain_after_slot(void) {
    g.drain_after_slot += 1;
}

void ggml_moe_h2d_pass(double secs) {
    g.passes    += 1;
    g.pass_secs += secs;
}

void ggml_moe_route_fold(const uint32_t * mask, int n_words, int n_expert) {
    uint64_t n = 0;
    for (int i = 0; i < n_words; i++) {
        fnv(g.digest, mask[i]);
        uint32_t w = mask[i];
        while (w) { n += w & 1u; w >>= 1; }
    }
    fnv(g.digest, (uint64_t) n_expert);
    g.digest_folds += 1;
    g.experts_sum  += n;
    if (ggml_moe_route_digest_enabled() >= 2) {
        fprintf(stderr, "moe-route: fold %llu experts %llu digest %016llx\n",
                      (unsigned long long) g.digest_folds, (unsigned long long) n,
                      (unsigned long long) g.digest);
    }
}

void ggml_moe_route_fold_range(int32_t first_id, int32_t last_id, size_t bytes) {
    fnv(g.digest, (uint64_t) (uint32_t) first_id);
    fnv(g.digest, (uint64_t) (uint32_t) last_id);
    fnv(g.digest, (uint64_t) bytes);
    if (ggml_moe_h2d_stats_enabled()) {
        g.range_kb.push_back((float) (bytes / 1024.0));
    }
}

void ggml_moe_h2d_report(void) {
    if (reported) {
        return;
    }
    reported = true;
    if (ggml_moe_route_digest_enabled() && g.digest_folds) {
        fprintf(stderr, "moe-route-digest: %016llx folds=%llu experts_mean=%.1f\n",
                      (unsigned long long) g.digest,
                      (unsigned long long) g.digest_folds,
                      (double) g.experts_sum / (double) g.digest_folds);
    }
    if (!ggml_moe_h2d_stats_enabled() || g.copies == 0) {
        return;
    }
    double total = 0;
    for (int b = 0; b < GGML_MOE_H2D_N_BUCKETS; b++) {
        total += g.bucket[b];
    }
    fprintf(stderr, "moe-h2d: %llu copies, %llu ranges, %.2f GiB, %llu passes, "
                  "%.1f s in the expert path of %.1f s of compute_splits\n",
                  (unsigned long long) g.copies, (unsigned long long) g.ranges,
                  g.bytes / 1073741824.0, (unsigned long long) g.passes,
                  total, g.pass_secs);
    for (int b = 0; b < GGML_MOE_H2D_N_BUCKETS; b++) {
        fprintf(stderr, "moe-h2d:   %-18s %8.1f s  %5.1f %% of the expert path  "
                      "%5.1f %% of compute_splits  %7.2f ms per copy\n",
                      bucket_name(b), g.bucket[b],
                      total > 0 ? 100.0 * g.bucket[b] / total : 0.0,
                      g.pass_secs > 0 ? 100.0 * g.bucket[b] / g.pass_secs : 0.0,
                      1e3 * g.bucket[b] / (double) g.copies);
    }
    if (g.bucket[GGML_MOE_H2D_COPY] > 0) {
        fprintf(stderr, "moe-h2d:   effective H2D while issuing: %.2f GB/s "
                      "(%.2f GiB over %.1f s)\n",
                      g.bytes / 1e9 / g.bucket[GGML_MOE_H2D_COPY],
                      g.bytes / 1073741824.0, g.bucket[GGML_MOE_H2D_COPY]);
    }
    std::sort(g.copy_ms.begin(), g.copy_ms.end());
    std::sort(g.range_kb.begin(), g.range_kb.end());
    fprintf(stderr, "moe-h2d:   per-copy ms  p10 %.1f  p50 %.1f  p90 %.1f  max %.1f\n",
                  pct(g.copy_ms, 0.10), pct(g.copy_ms, 0.50), pct(g.copy_ms, 0.90),
                  g.copy_ms.empty() ? 0.0f : g.copy_ms.back());
    fprintf(stderr, "moe-h2d:   range KiB    p10 %.0f  p50 %.0f  p90 %.0f  max %.0f  "
                  "mean %.0f, %.1f ranges per copy\n",
                  pct(g.range_kb, 0.10), pct(g.range_kb, 0.50), pct(g.range_kb, 0.90),
                  g.range_kb.empty() ? 0.0f : g.range_kb.back(),
                  g.ranges ? g.bytes / 1024.0 / g.ranges : 0.0,
                  g.copies ? (double) g.ranges / (double) g.copies : 0.0);

    // how far ahead a copy could be issued without overwriting a destination an earlier
    // copy's matmul is still reading. 0 % at distance k means the k-th previous expert
    // weight never shares device memory with this one, so issuing k copies ahead on a
    // second stream needs no extra buffer and no extra synchronisation.
    fprintf(stderr, "moe-h2d:   destination reuse over %llu copies: "
                  "prev-1 %.1f %%  prev-2 %.1f %%  prev-3 %.1f %%  prev-4 %.1f %%\n",
                  (unsigned long long) g.distinct_checked,
                  100.0 * g.overlap_at[0] / (double) g.distinct_checked,
                  100.0 * g.overlap_at[1] / (double) g.distinct_checked,
                  100.0 * g.overlap_at[2] / (double) g.distinct_checked,
                  100.0 * g.overlap_at[3] / (double) g.distinct_checked);
    fprintf(stderr, "moe-h2d:   compute-stream drains while a slot copy was in flight: %llu\n",
                  (unsigned long long) g.drain_after_slot);
}
