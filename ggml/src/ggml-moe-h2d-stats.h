#pragma once

// Measurement-only instrumentation of the scheduler's selective expert H2D copy.
//
// Two independent, opt-in facilities, both no-ops unless their environment variable is
// set, both confined to ggml_backend_sched_compute_splits:
//
//   GGML_MOE_H2D_STATS=1   time every (layer, weight) copy of an offloaded MUL_MAT_ID
//                          into four buckets and print the totals at scheduler free.
//                          The buckets are the serialised chain the copy sits in:
//                            wait_prev   waiting for the split backend to release the
//                                        destination buffer, i.e. GPU compute backlog
//                            ids         reading the routing ids back D2H and syncing,
//                                        i.e. waiting for this layer's attention
//                            prefetch    blocking on the host page-population workers
//                            h2d         issuing the ranged copies themselves
//                          Their sum is the whole cost of the expert path per pass;
//                          what is left of wall-clock is graph launch and CPU splits.
//
//   GGML_MOE_ROUTE_DIGEST=1  fold each layer's used-expert bitset and its coalesced range
//                          list into a running 64-bit FNV-1a digest, printed at free.
//                          This is a correctness oracle for changes to the H2D path that
//                          does not depend on sampling, on the output text, or on the
//                          prefetch pool being on: two runs whose digests agree selected
//                          and copied exactly the same expert bytes in the same order.
//                          =2 also prints a per-pass digest.
//
// Neither facility allocates on the timed path or changes control flow.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_moe_h2d_bucket {
    GGML_MOE_H2D_WAIT_PREV = 0,
    GGML_MOE_H2D_IDS       = 1,
    GGML_MOE_H2D_PREFETCH  = 2,
    GGML_MOE_H2D_COPY      = 3,
    GGML_MOE_H2D_N_BUCKETS = 4,
};

// enabled() is read once from the environment; the callers guard on it so a disabled
// build path costs one predictable branch.
int    ggml_moe_h2d_stats_enabled(void);
int    ggml_moe_route_digest_enabled(void);

double ggml_moe_h2d_now(void);
void   ggml_moe_h2d_add(enum ggml_moe_h2d_bucket b, double secs);
// one completed weight copy: n ranges totalling bytes, into [dst, dst+dst_bytes).
// The destination is recorded because whether a layer's gate/up/down expert copies land
// in distinct device buffers decides whether they may be issued ahead of each other on a
// second stream: ggml_gallocr reuses device memory by graph liveness, so if up's
// destination aliases gate's, an early copy of up would overwrite gate while gate's
// matmul is still reading it. See the overlap section of the report.
void   ggml_moe_h2d_copy_done(size_t n_ranges, size_t bytes,
                              const void * dst, size_t dst_bytes);
// one scheduler pass boundary
void   ggml_moe_h2d_pass(double secs);

// fold one layer's routing into the digest: the bitset words, the popcount and the
// coalesced range list the copy will issue
void   ggml_moe_route_fold(const uint32_t * mask, int n_words, int n_expert);
void   ggml_moe_route_fold_range(int32_t first_id, int32_t last_id, size_t bytes);

// a compute-stream drain that ran while a slot copy of the same split was already in
// flight. Such a drain blocks the host until that copy lands and so cancels the overlap;
// counting it distinguishes "overlap did not help" from "overlap never happened".
void   ggml_moe_h2d_note_drain_after_slot(void);

void   ggml_moe_h2d_report(void);

#ifdef __cplusplus
}
#endif
