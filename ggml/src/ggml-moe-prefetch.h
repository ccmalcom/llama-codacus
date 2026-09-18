#pragma once

// Selective host-side page-cache population for mmap-backed MoE expert weights.
//
// Ported mechanism (not policy) from ik_llama.cpp's ggml-moe-prefetch engine as
// measured in the B5 arm of local-ai's docs/nvme-tier-results.md: a small pool of
// host worker threads that MADV_POPULATE_READ 2 MiB chunks of the expert rows the
// current split's routing actually selected, skipping chunks mincore reports as
// resident, so the scheduler's pageable H2D copy of those rows hits warm page
// cache instead of faulting them in one thread at a time.
//
// Deliberately NOT ported: whole-tensor lookahead of future layers, MADV_COLD of
// populated pages, the CPU kernel-entry hook, the decode-path node enqueue.
//
// Opt-in: the pool only exists when the scheduler was told to create it
// (GGML_MOE_HOST_PREFETCH=<n_threads>); every entry point is a no-op otherwise.

#include "ggml.h"

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// mapping registry: only tensors whose bytes lie inside a registered mmap are
// ever populated. Nothing is opened or retained beyond (base, size).
void ggml_moe_prefetch_register_mapping  (const void * base, size_t size);
void ggml_moe_prefetch_unregister_mapping(const void * base);

// worker pool control; n_threads > 0 (re)creates the pool, <= 0 shuts it down.
// Idempotent when the pool already has n_threads workers.
void ggml_moe_prefetch_set_n_threads(int n_threads);
bool ggml_moe_prefetch_enabled(void);

// bumped once per scheduler pass so a per-tensor enqueue is idempotent within
// one graph execution but re-issued on the next.
void ggml_moe_prefetch_new_epoch(void);

// enqueue the selected expert rows of weight tensor w (expert dim = ne[2],
// stride nb[2]) given a bitmask with one bit per expert, LSB-first within
// 32-bit words (the scheduler's ggml_bitset_t layout). Consecutive selected
// experts coalesce into one range; each range is extended by `tail_pad` bytes
// (clamped to the tensor) to cover the padding the H2D copy reads past the
// last expert of a run. Urgent: inserted at the queue head.
void ggml_moe_prefetch_mask(const struct ggml_tensor * w, const uint32_t * mask, int n_expert, size_t tail_pad);

// block until every job enqueued for w in the current epoch has been processed.
// Returns immediately when nothing is pending or the pool is off.
void ggml_moe_prefetch_wait(const struct ggml_tensor * w);

#ifdef __cplusplus
}
#endif
