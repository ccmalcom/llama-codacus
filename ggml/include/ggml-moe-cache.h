#pragma once

// Round 15: a demand-filled (LRU) residency policy for the VRAM expert pack, at
// batch-1 decode only.
//
// The shipped pack (--moe-cache-slots, llama_model_base::init_moe_expert_cache) is
// frozen at load from a frequency profile. Measured on the deployed configuration it
// serves 32.0% of decode draws; a plain LRU of the same size serves 63.2%, and at 64
// slots 70.0%, with no profile at all. Decode is transfer-bound, so that is most of
// the cost.
//
// THE ONE INVARIANT THIS FILE EXISTS TO HOLD: the admission *is* the miss. An expert
// that misses is already crossing PCIe this step. Writing it into a persistent slot
// instead of the graph allocator's throwaway staging mirror costs no extra bytes --
// the destination tensor and the destination offset are independent parameters of
// ggml_backend_tensor_set_async, and llama-model.cpp's load-time fill already does
// exactly this copy (host expert e -> pack slot s). A separate promotion copy would
// double decode traffic and put every policy below CPU decode; there is no version of
// this worth building that adds a second transfer.
//
// HOW THE IDS ARE STEERED, and why it is not the obvious way. The graph derives both
// id tensors on the GPU with ggml_get_rows over the residency maps, and the graph
// allocator legitimately gives those two derived tensors the SAME address: the cold
// chain is built first, so it has consumed ids_cold before ids_hot is produced, and the
// live ranges are disjoint. Writing ids_hot from the host is therefore useless -- the
// later get_rows overwrites it. So the policy steers the hot side through
// `moe_map_hot`, which get_rows reads afterwards, and only the cold side is written
// directly (to -1, which the cold chain then consumes before the hot get_rows runs).
// The aliasing is not worked around; it is relied on, in the order the graph already
// guarantees.
//
// Decode reads its raw routing from a permanent identity map (`moe_map_ident`) that the
// graph substitutes for `moe_map_cold` at batch 1 only. That keeps the real cold map
// accurate for prefill, which therefore keeps the pack and is left byte-for-byte alone.
//
// Opt-in: inert unless llama registered the packs AND the policy is enabled. Prefill
// (node->ne[2] != 1) never takes this path.

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- registration, from llama ------------------------------------------------

// Enable the policy. Without this every entry point below is a no-op and the pack
// keeps its load-time contents.
GGML_API void ggml_moe_cache_set_enabled(int enabled);
GGML_API int  ggml_moe_cache_enabled(void);

// One call per packed layer, at model load. `host` are the CPU-resident full expert
// weights, `pack` the device-resident slot packs, `map_hot`/`map_cold` the persistent
// i32[n_expert] residency maps the graph's get_rows reads. `slot_owner_init` is the
// expert currently occupying each slot, or -1; pass NULL for a cold start.
GGML_API void ggml_moe_cache_register_layer(int il,
                                   struct ggml_tensor * host_gate,
                                   struct ggml_tensor * host_up,
                                   struct ggml_tensor * host_down,
                                   struct ggml_tensor * pack_gate,
                                   struct ggml_tensor * pack_up,
                                   struct ggml_tensor * pack_down,
                                   struct ggml_tensor * map_hot,
                                   struct ggml_tensor * map_cold,
                                   struct ggml_tensor * map_ident,
                                   int n_slots, int n_expert,
                                   const int32_t * slot_owner_init);

GGML_API void ggml_moe_cache_clear_registry(void);

// --- the hot path, from the scheduler ----------------------------------------

// Called from ggml_backend_sched's split-input loop with the routing it has already
// read back. Returns 1 if this input was handled -- residency decided, admissions
// issued, id tensors rewritten -- and the caller must then skip the staging copy
// entirely. Returns 0 if the policy does not apply, and the caller proceeds exactly
// as before.
//
// `ids_host` is the scheduler's readback of `ids_tensor`. At decode that tensor was
// derived from the identity map, so it holds this step's raw expert ids -- every draw,
// hit or miss. Residency is decided against this module's own host-side mirror, not
// against anything read back from the device.
GGML_API int ggml_moe_cache_admit(ggml_backend_t backend,
                         const struct ggml_tensor * host_weight,
                         const struct ggml_tensor * node,
                         struct ggml_tensor * ids_tensor,
                         const int32_t * ids_host, int64_t n_ids);

// Per-eval bookkeeping: the decision is made once per layer per eval, on whichever of
// the layer's three weight splits the scheduler reaches first.
GGML_API void ggml_moe_cache_begin_eval(void);

// --- accounting ---------------------------------------------------------------

struct ggml_moe_cache_stats {
    int64_t steps;          // layer-steps that took the policy path
    int64_t draws;          // routed expert slots seen (10 per layer-step)
    int64_t hits;           // already resident
    int64_t misses;         // admitted this step
    int64_t evictions;      // admissions that displaced a live expert
    int64_t experts_copied; // == misses
    int64_t bytes_h2d;      // admission bytes only, all three tensors
    int64_t empty_admits;   // layer-steps where every draw was already resident
};

GGML_API void ggml_moe_cache_get_stats(struct ggml_moe_cache_stats * out);
GGML_API void ggml_moe_cache_report(void);

#ifdef __cplusplus
}
#endif
