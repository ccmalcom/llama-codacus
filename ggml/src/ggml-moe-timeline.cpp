#include "ggml-moe-timeline.h"
#include "ggml-impl.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <vector>

namespace {

struct phase {
    double   bucket[GGML_MOE_TL_N_BUCKETS] = {};
    uint64_t count [GGML_MOE_TL_N_BUCKETS] = {};
    double   pass_secs = 0;
    double   tail_secs = 0;     // counted inside bucket[TAIL_SYNC] too; kept for clarity
    uint64_t passes    = 0;
    uint64_t splits    = 0;
    uint64_t nodes     = 0;
};

struct timeline {
    // the pass currently open. Buckets are accumulated here and committed on pass_end,
    // because whether a pass is prefill or decode is only known once the residency
    // policy has acted -- part way through it.
    double   cur_bucket[GGML_MOE_TL_N_BUCKETS] = {};
    uint64_t cur_count [GGML_MOE_TL_N_BUCKETS] = {};
    int      cur_is_decode = 0;
    int      last_is_decode = 0;

    phase    ph[2];             // [0] prefill / not-decode, [1] decode
    std::vector<float> decode_pass_ms;

    int      sched_n_copies = -1;
    int      sched_has_events = -1;
    uint64_t cuda_graph_calls = 0;
    uint64_t cuda_graph_used  = 0;
};

timeline g;
int on = -1;
bool reported = false;

const char * bucket_name(int b) {
    switch (b) {
        case GGML_MOE_TL_SPLIT_SYNC:    return "split_sync";
        case GGML_MOE_TL_INPUT_WAIT:    return "input_wait";
        case GGML_MOE_TL_INPUT_WAIT_MOE:return "input_wait_moe";
        case GGML_MOE_TL_IDS_READBACK:  return "ids_readback";
        case GGML_MOE_TL_TAIL_SYNC:     return "tail_sync";
        case GGML_MOE_TL_CPU_JOIN:      return "cpu_join";
        case GGML_MOE_TL_ADMIT_POLICY:  return "admit_policy";
        case GGML_MOE_TL_ADMIT_WEIGHTS: return "admit_weights";
        case GGML_MOE_TL_ADMIT_MAPS:    return "admit_maps";
        case GGML_MOE_TL_ADMIT_IDS:     return "admit_ids";
        case GGML_MOE_TL_PREFETCH_WAIT: return "prefetch_wait";
        case GGML_MOE_TL_STAGE_COPY:    return "stage_copy";
        case GGML_MOE_TL_LAUNCH:        return "launch";
    }
    return "?";
}

// which side of the CPU/GPU question a bucket answers
bool is_gpu_wait(int b) {
    return b == GGML_MOE_TL_SPLIT_SYNC || b == GGML_MOE_TL_INPUT_WAIT ||
           b == GGML_MOE_TL_INPUT_WAIT_MOE ||
           b == GGML_MOE_TL_IDS_READBACK || b == GGML_MOE_TL_TAIL_SYNC ||
           b == GGML_MOE_TL_CPU_JOIN;
}

float pct(const std::vector<float> & v, double q) {
    if (v.empty()) return 0;
    return v[(size_t)(q * (v.size() - 1))];
}

// llama-server is SIGTERMed and never reaches ggml_backend_sched_free, so the report has
// to come out of a static destructor, exactly as the round-13 H2D counters do.
struct reporter {
    ~reporter() { ggml_moe_tl_report(); }
};
reporter g_reporter;

} // namespace

int ggml_moe_tl_enabled(void) {
    if (on < 0) {
        const char * s = getenv("GGML_MOE_TIMELINE");
        on = s ? atoi(s) : 0;
    }
    return on;
}

double ggml_moe_tl_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

void ggml_moe_tl_add(enum ggml_moe_tl_bucket b, double secs) {
    g.cur_bucket[b] += secs;
    g.cur_count [b] += 1;
}

void ggml_moe_tl_mark_decode(void) {
    g.cur_is_decode = 1;
}

void ggml_moe_tl_pass_begin(void) {
    for (int b = 0; b < GGML_MOE_TL_N_BUCKETS; b++) {
        g.cur_bucket[b] = 0;
        g.cur_count [b] = 0;
    }
    g.cur_is_decode = 0;
}

void ggml_moe_tl_pass_end(double secs, int n_splits, int n_nodes) {
    phase & p = g.ph[g.cur_is_decode];
    for (int b = 0; b < GGML_MOE_TL_N_BUCKETS; b++) {
        p.bucket[b] += g.cur_bucket[b];
        p.count [b] += g.cur_count [b];
    }
    p.pass_secs += secs;
    p.passes    += 1;
    p.splits    += (uint64_t) n_splits;
    p.nodes     += (uint64_t) n_nodes;
    if (g.cur_is_decode) {
        g.decode_pass_ms.push_back((float) (secs * 1e3));
    }
    g.last_is_decode = g.cur_is_decode;
}

void ggml_moe_tl_tail(double secs) {
    phase & p = g.ph[g.last_is_decode];
    p.bucket[GGML_MOE_TL_TAIL_SYNC] += secs;
    p.count [GGML_MOE_TL_TAIL_SYNC] += 1;
    p.tail_secs += secs;
}

void ggml_moe_tl_note_sched(int n_copies, int has_events) {
    g.sched_n_copies   = n_copies;
    g.sched_has_events = has_events;
}

void ggml_moe_tl_note_cuda_graph(int used) {
    g.cuda_graph_calls += 1;
    g.cuda_graph_used  += used ? 1 : 0;
}

void ggml_moe_tl_report(void) {
    if (reported || !ggml_moe_tl_enabled()) {
        return;
    }
    reported = true;

    if (g.sched_n_copies >= 0) {
        fprintf(stderr, "moe-tl: scheduler n_copies=%d, per-backend events=%s -> an input "
                      "wait is %s\n",
                      g.sched_n_copies, g.sched_has_events ? "yes" : "NO",
                      g.sched_has_events ? "ggml_backend_event_wait (stream-side, host does "
                                           "not block)"
                                         : "ggml_backend_synchronize (the host blocks until "
                                           "the GPU has drained every kernel issued so far)");
    }
    if (g.cuda_graph_calls) {
        fprintf(stderr, "moe-tl: CUDA graph: %llu of %llu graph_compute calls replayed a "
                      "captured graph (%.1f %%)\n",
                      (unsigned long long) g.cuda_graph_used,
                      (unsigned long long) g.cuda_graph_calls,
                      100.0 * (double) g.cuda_graph_used / (double) g.cuda_graph_calls);
    }

    for (int ph = 1; ph >= 0; ph--) {
        const phase & p = g.ph[ph];
        if (p.passes == 0) {
            continue;
        }
        const char * tag = ph ? "decode" : "prefill";
        // the tail lands outside the pass it belongs to, so the denominator for a
        // per-token cost is pass + tail, which is the whole eval as the host sees it
        const double eval_secs = p.pass_secs + p.tail_secs;
        const double per = 1e3 / (double) p.passes;

        double sum = 0, gpu_wait = 0;
        for (int b = 0; b < GGML_MOE_TL_N_BUCKETS; b++) {
            sum += p.bucket[b];
            if (is_gpu_wait(b)) {
                gpu_wait += p.bucket[b];
            }
        }

        fprintf(stderr,
            "moe-tl[%s]: %llu passes, %.1f s of evals (%.2f ms each), "
            "%.1f splits and %.0f nodes per pass\n",
            tag, (unsigned long long) p.passes, eval_secs, eval_secs * per,
            (double) p.splits / (double) p.passes, (double) p.nodes / (double) p.passes);

        for (int b = 0; b < GGML_MOE_TL_N_BUCKETS; b++) {
            if (p.count[b] == 0) {
                continue;
            }
            fprintf(stderr,
                "moe-tl[%s]:   %-14s %8.2f ms/pass  %5.1f %% of eval  "
                "%9llu calls  %8.1f us each  [%s]\n",
                tag, bucket_name(b), p.bucket[b] * per,
                eval_secs > 0 ? 100.0 * p.bucket[b] / eval_secs : 0.0,
                (unsigned long long) p.count[b],
                1e6 * p.bucket[b] / (double) p.count[b],
                is_gpu_wait(b) ? "host waits on GPU" : "host busy");
        }
        const double other = eval_secs - sum;
        fprintf(stderr,
            "moe-tl[%s]:   %-14s %8.2f ms/pass  %5.1f %% of eval  "
            "(scheduler bookkeeping: everything not in a bucket above)\n",
            tag, "unbucketed", other * per,
            eval_secs > 0 ? 100.0 * other / eval_secs : 0.0);
        fprintf(stderr,
            "moe-tl[%s]:   critical path: %.2f ms/pass host waiting on GPU (%.1f %%), "
            "%.2f ms/pass host busy (%.1f %%), %.2f ms/pass unbucketed\n",
            tag, gpu_wait * per, eval_secs > 0 ? 100.0 * gpu_wait / eval_secs : 0.0,
            (sum - gpu_wait) * per,
            eval_secs > 0 ? 100.0 * (sum - gpu_wait) / eval_secs : 0.0,
            other * per);
    }

    if (!g.decode_pass_ms.empty()) {
        std::sort(g.decode_pass_ms.begin(), g.decode_pass_ms.end());
        fprintf(stderr, "moe-tl[decode]: per-pass ms  p10 %.2f  p50 %.2f  p90 %.2f  "
                      "max %.2f\n",
                      pct(g.decode_pass_ms, 0.10), pct(g.decode_pass_ms, 0.50),
                      pct(g.decode_pass_ms, 0.90), g.decode_pass_ms.back());
    }
}
