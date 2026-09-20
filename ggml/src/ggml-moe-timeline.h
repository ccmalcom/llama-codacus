#pragma once

// Round 16: a host-side wall-clock decomposition of one decode token.
//
// Round 15 fitted `token_ms = 56.4 + 80.7*(1 - hit_rate)` to three operating points with
// residuals under 0.15 ms. The slope is expert transfer. The 56.4 ms intercept is what
// this file exists to explain, and the round-15 instrumentation cannot: its four buckets
// (ggml-moe-h2d-stats.h) cover only the scheduler's *selective staging copy*, which the
// dynamic residency policy replaces. Under `GGML_MOE_CACHE_POLICY=lru` the scheduler's
// h2d_issue bucket saw 25.93 GiB while moe-cache admitted 142.03 GiB -- roughly 85 % of
// decode's transfer had no timer on it at all.
//
// This module times the whole host thread through one scheduler pass, in buckets that
// partition it rather than sampling it, and keeps prefill and decode apart. The sum of
// the buckets plus `other` is the pass wall-clock by construction, so nothing can hide.
//
// Opt-in with GGML_MOE_TIMELINE=1; every entry point is one cached-int branch otherwise.
// Nothing here allocates on the timed path, issues a device call, or changes control
// flow: the buckets must not create the serialisation they are measuring.
//
// A bucket is host time, not device time. `*_sync`/`*_wait`/`*_readback` are the host
// blocked on the GPU; `launch`, `admit_*` and `stage_copy` are the host doing work or
// blocked inside the driver. That split is the whole point: it answers whether decode is
// CPU-bound or GPU-bound without a profiler.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_moe_tl_bucket {
    // --- host blocked on the device -----------------------------------------
    GGML_MOE_TL_SPLIT_SYNC = 0,  // event_synchronize between two splits of different backends
    GGML_MOE_TL_INPUT_WAIT,      // waiting to reuse a split input's staging buffer
    GGML_MOE_TL_INPUT_WAIT_MOE,  // ... of an input the residency policy then handled, so
                                 //     nothing was written to the staging buffer at all
    GGML_MOE_TL_IDS_READBACK,    // D2H of this layer's routing ids, and the stream sync for it
    GGML_MOE_TL_TAIL_SYNC,       // ggml_backend_sched_synchronize at the end of the eval
    GGML_MOE_TL_CPU_JOIN,        // waiting for the async CPU worker to finish a CPU split
    // --- host doing work, or blocked inside the driver ----------------------
    GGML_MOE_TL_ADMIT_POLICY,    // the LRU decision: two passes over the draws, victim search
    GGML_MOE_TL_ADMIT_WEIGHTS,   // the admissions themselves: 3 pageable H2D per miss
    GGML_MOE_TL_ADMIT_MAPS,      // residency-map writes, 4 bytes each
    GGML_MOE_TL_ADMIT_IDS,       // silencing the cold chain for this step
    GGML_MOE_TL_PREFETCH_WAIT,   // blocking on the host page-population workers
    GGML_MOE_TL_STAGE_COPY,      // the scheduler's selective staging copy (the static path)
    GGML_MOE_TL_LAUNCH,          // ggml_backend_graph_compute_async: kernel dispatch
    GGML_MOE_TL_N_BUCKETS
};

int    ggml_moe_tl_enabled(void);
double ggml_moe_tl_now(void);

// one interval into a bucket of the pass currently open
void   ggml_moe_tl_add(enum ggml_moe_tl_bucket b, double secs);

// this pass is a decode step, not prefill. Called by the residency policy when it acts,
// which is exactly the batch-1 condition, established by provenance rather than guessed.
void   ggml_moe_tl_mark_decode(void);

void   ggml_moe_tl_pass_begin(void);
void   ggml_moe_tl_pass_end(double secs, int n_splits, int n_nodes);
// the end-of-eval drain, which happens after the pass closes; attributed to the phase of
// the pass that just closed.
void   ggml_moe_tl_tail(double secs);

// one-time structural facts, printed with the report rather than inferred from source:
// how many graph copies the scheduler was built with, and whether it therefore has events
// (without them every input wait is a full host-blocking stream drain, not a stream wait).
void   ggml_moe_tl_note_sched(int n_copies, int has_events);
// one ggml_backend_cuda_graph_compute call, and whether it replayed a captured CUDA graph
void   ggml_moe_tl_note_cuda_graph(int used);

void   ggml_moe_tl_report(void);

#ifdef __cplusplus
}
#endif
