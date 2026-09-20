#include "../include/ggml-moe-cache.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

namespace {

struct layer_state {
    int il = -1;

    const ggml_tensor * host[3] = { nullptr, nullptr, nullptr };
    ggml_tensor       * pack[3] = { nullptr, nullptr, nullptr };
    ggml_tensor       * map_hot   = nullptr;
    ggml_tensor       * map_cold  = nullptr;
    ggml_tensor       * map_ident = nullptr;

    int n_slots  = 0;
    int n_expert = 0;

    // residency, host-authoritative. expert_slot is the inverse of slot_owner and is
    // what decides hit vs miss; nothing is read back from the device to decide it.
    std::vector<int32_t> slot_owner;   // slot -> expert, or -1 if never filled
    std::vector<int32_t> expert_slot;  // expert -> slot, or -1 if not resident
    std::vector<int64_t> slot_used;    // slot -> strict recency stamp

    // the decision is made once per layer per eval
    int64_t decided_eval = -1;

    // scratch, kept to avoid per-step allocation
    std::vector<int32_t> draw;      // this step's raw expert ids
    std::vector<int32_t> minus1;    // the -1 pattern written into the cold ids
};

struct registry {
    bool enabled = false;
    int64_t eval = 0;
    int64_t clock = 0;   // strict total order for recency; never per-step

    std::vector<layer_state> layers;
    std::unordered_map<const ggml_tensor *, std::pair<int, int>> by_host; // tensor -> (layer idx, which of 3)

    ggml_moe_cache_stats stats = {};
};

registry & reg() {
    static registry r;
    return r;
}

} // namespace

// ---------------------------------------------------------------------------
// registration
// ---------------------------------------------------------------------------

void ggml_moe_cache_set_enabled(int enabled) {
    reg().enabled = enabled != 0;
    if (reg().enabled) {
        static bool hooked = false;
        if (!hooked) {
            hooked = true;
            atexit(ggml_moe_cache_report);
        }
    }
}

int ggml_moe_cache_enabled(void) {
    return reg().enabled ? 1 : 0;
}

void ggml_moe_cache_clear_registry(void) {
    registry & r = reg();
    r.layers.clear();
    r.by_host.clear();
    r.eval = 0;
    r.clock = 0;
    r.stats = {};
}

void ggml_moe_cache_register_layer(int il,
                                   ggml_tensor * host_gate,
                                   ggml_tensor * host_up,
                                   ggml_tensor * host_down,
                                   ggml_tensor * pack_gate,
                                   ggml_tensor * pack_up,
                                   ggml_tensor * pack_down,
                                   ggml_tensor * map_hot,
                                   ggml_tensor * map_cold,
                                   ggml_tensor * map_ident,
                                   int n_slots, int n_expert,
                                   const int32_t * slot_owner_init) {
    registry & r = reg();

    layer_state st;
    st.il = il;
    st.host[0] = host_gate; st.host[1] = host_up; st.host[2] = host_down;
    st.pack[0] = pack_gate; st.pack[1] = pack_up; st.pack[2] = pack_down;
    st.map_hot   = map_hot;
    st.map_cold  = map_cold;
    st.map_ident = map_ident;
    st.n_slots  = n_slots;
    st.n_expert = n_expert;

    st.slot_owner.assign(n_slots, -1);
    st.expert_slot.assign(n_expert, -1);
    st.slot_used.assign(n_slots, 0);
    if (slot_owner_init) {
        for (int s = 0; s < n_slots; s++) {
            st.slot_owner[s] = slot_owner_init[s];
            if (slot_owner_init[s] >= 0 && slot_owner_init[s] < n_expert) {
                st.expert_slot[slot_owner_init[s]] = s;
            }
            // Seed recency in rank order, not all-equal: round 14's crosscheck found
            // that a warm cache seeded at one recency evicts its best-ranked entry
            // first, because min() returns the first minimum. Slot order is rank
            // order here, so a later slot must look less recently used.
            st.slot_used[s] = slot_owner_init[s] >= 0 ? (int64_t) (n_slots - s) : 0;
        }
    }

    r.layers.push_back(std::move(st));
    const int idx = (int) r.layers.size() - 1;
    if (host_gate) { r.by_host[host_gate] = { idx, 0 }; }
    if (host_up)   { r.by_host[host_up]   = { idx, 1 }; }
    if (host_down) { r.by_host[host_down] = { idx, 2 }; }
}

void ggml_moe_cache_begin_eval(void) {
    reg().eval++;
}

// ---------------------------------------------------------------------------
// the policy
// ---------------------------------------------------------------------------

int ggml_moe_cache_admit(ggml_backend_t backend,
                         const ggml_tensor * host_weight,
                         const ggml_tensor * node,
                         ggml_tensor * ids_tensor,
                         const int32_t * ids_host, int64_t n_ids) {
    registry & r = reg();
    if (!r.enabled || r.layers.empty()) {
        return 0;
    }
    // decode only: prefill routes far more experts than a layer has slots, and its
    // streaming path is deliberately left alone.
    if (node == nullptr || node->ne[2] != 1 || ids_tensor == nullptr) {
        return 0;
    }
    auto it = r.by_host.find(host_weight);
    if (it == r.by_host.end()) {
        return 0;
    }
    layer_state & st = r.layers[it->second.first];
    if (st.n_slots <= 0 || !st.map_hot || !st.map_cold || !st.map_ident) {
        return 0;
    }
    // Only act when the graph actually derived these ids from the identity map, i.e.
    // when they are this step's raw routing. A node can be batch-1 without being decode:
    // during prefill only the last token needs logits, so the final layer's MoE runs at
    // ne[2]==1 while the graph was built at n_tokens>1 and used the real cold map. Acting
    // on those mapped ids would admit the wrong experts and silence a cold chain that was
    // still carrying part of the layer. Checking provenance makes the two conditions the
    // same condition instead of two that merely usually agree.
    {
        const ggml_tensor * produced_by = ids_tensor->view_src ? ids_tensor->view_src : ids_tensor;
        if (produced_by->op != GGML_OP_GET_ROWS || produced_by->src[0] != st.map_ident) {
            return 0;
        }
    }
    // Every routed expert must be able to be resident at once, or "never evict an
    // expert routed this step" is unsatisfiable and the policy is unsound.
    if (n_ids <= 0 || n_ids > st.n_slots) {
        return 0;
    }
    // The ids are written back wholesale, so a strided view would be written wrong.
    if (!ggml_is_contiguous(ids_tensor) || ggml_nelements(ids_tensor) != n_ids) {
        return 0;
    }

    // The layer's three weight tensors are three separate split inputs sharing one
    // routing. Decide on the first one reached; the other two only need the staging
    // copy skipped, which is what returning 1 does.
    if (st.decided_eval == r.eval) {
        return 1;
    }
    st.decided_eval = r.eval;

    st.draw.assign(ids_host, ids_host + n_ids);

    const int64_t eval_clock_base = r.clock;
    int n_hits = 0, n_misses = 0, n_evict = 0;

    // --- pass 1: touch the hits -----------------------------------------------
    // Done before any victim is chosen, so a slot that is resident *and* routed this
    // step can never be selected as a victim below.
    for (int64_t i = 0; i < n_ids; i++) {
        const int32_t e = st.draw[i];
        if (e < 0 || e >= st.n_expert) {
            continue;
        }
        const int32_t slot = st.expert_slot[e];
        if (slot >= 0) {
            st.slot_used[slot] = ++r.clock;   // strict total order, never the step number
            n_hits++;
        }
    }

    // --- pass 2: admit the misses ---------------------------------------------
    for (int64_t i = 0; i < n_ids; i++) {
        const int32_t e = st.draw[i];
        if (e < 0 || e >= st.n_expert) {
            continue;   // nothing routed at this slot
        }
        if (st.expert_slot[e] >= 0) {
            continue;   // resident: a hit, or admitted a few iterations ago this step
        }

        // victim: strictly least recently used. Anything touched during this eval has a
        // stamp above eval_clock_base and is ineligible, which is the "never evict an
        // expert routed this step" invariant enforced rather than assumed.
        int     victim = -1;
        int64_t best   = 0;
        for (int sl = 0; sl < st.n_slots; sl++) {
            if (st.slot_used[sl] > eval_clock_base) {
                continue;                      // routed this step
            }
            if (st.slot_owner[sl] < 0) {       // never filled: free, take it first
                victim = sl;
                break;
            }
            if (victim < 0 || st.slot_used[sl] < best) {
                victim = sl;
                best   = st.slot_used[sl];
            }
        }
        if (victim < 0) {
            // unreachable while n_ids <= n_slots, but a wrong slot is silent
            // corruption, so fail loudly rather than write one
            GGML_ABORT("moe-cache: no evictable slot (layer %d, n_ids %lld, slots %d)",
                       st.il, (long long) n_ids, st.n_slots);
        }

        const int32_t evicted = st.slot_owner[victim];

        // --- the admission. THIS IS THE MISS'S OWN TRANSFER, REDIRECTED. ---
        // Source is the host mapping at the expert's natural offset; destination is the
        // persistent pack at the slot's offset. Exactly the bytes the staging copy would
        // have moved, to a destination that survives the eval. No padding_end: the pack
        // is sized S*nb[2] exactly, op_params[0]=1 excludes MMQ, and the load-time fill
        // already writes unpadded slots.
        for (int t = 0; t < 3; t++) {
            const ggml_tensor * src = st.host[t];
            ggml_tensor       * dst = st.pack[t];
            if (!src || !dst) {
                continue;
            }
            const size_t nb = src->nb[2];
            ggml_backend_tensor_set_async(backend, dst,
                                          (const uint8_t *) src->data + (size_t) e * nb,
                                          (size_t) victim * nb, nb);
            r.stats.bytes_h2d += (int64_t) nb;
        }

        // --- residency bookkeeping ---------------------------------------------
        // map_hot is the steering wheel: the graph's get_rows over it runs *after* this
        // point and turns these writes into the hot chain's slot ids. map_cold is kept
        // accurate in the same breath, because prefill still reads it and the two chains
        // are summed -- an expert present in both is added twice.
        const int32_t minus_one = -1;
        if (evicted >= 0 && evicted < st.n_expert) {
            ggml_backend_tensor_set_async(backend, st.map_hot, &minus_one,
                                          (size_t) evicted * sizeof(int32_t), sizeof(int32_t));
            ggml_backend_tensor_set_async(backend, st.map_cold, &evicted,
                                          (size_t) evicted * sizeof(int32_t), sizeof(int32_t));
            st.expert_slot[evicted] = -1;
            n_evict++;
        }
        {
            const int32_t slot_i32 = victim;
            ggml_backend_tensor_set_async(backend, st.map_hot, &slot_i32,
                                          (size_t) e * sizeof(int32_t), sizeof(int32_t));
            ggml_backend_tensor_set_async(backend, st.map_cold, &minus_one,
                                          (size_t) e * sizeof(int32_t), sizeof(int32_t));
        }

        st.slot_owner[victim] = e;
        st.expert_slot[e]     = victim;
        st.slot_used[victim]  = ++r.clock;
        n_misses++;
    }

    // --- invariant: every routed expert is now resident -----------------------
    // If a victim had been one of this step's own draws, the hot chain would read some
    // other expert's weights and the output would be quietly wrong rather than crash.
    // This is the cheap direct test of that (ten lookups), so it is always on.
    for (int64_t i = 0; i < n_ids; i++) {
        const int32_t e = st.draw[i];
        if (e < 0 || e >= st.n_expert) {
            continue;
        }
        const int32_t sl = st.expert_slot[e];
        if (sl < 0 || sl >= st.n_slots || st.slot_owner[sl] != e) {
            GGML_ABORT("moe-cache: draw %lld (expert %d) not resident after admission "
                       "(layer %d, slot %d, owner %d)",
                       (long long) i, e, st.il, sl,
                       sl >= 0 && sl < st.n_slots ? st.slot_owner[sl] : -2);
        }
    }

    // --- silence the cold chain for this step ---------------------------------
    // Every routed expert is now resident, so the hot chain alone reconstructs the
    // layer. The cold ids must go to -1 in the same step or the expert's contribution
    // is added twice: the chains are summed, and disjointness is what makes that a
    // reconstruction rather than a doubling. This write lands before the cold matmuls
    // consume it and before the hot get_rows overwrites the shared storage.
    if ((int64_t) st.minus1.size() != n_ids) {
        st.minus1.assign(n_ids, -1);
    }
    ggml_backend_tensor_set_async(backend, ids_tensor, st.minus1.data(), 0,
                                  n_ids * sizeof(int32_t));

    static int dbg_left = -1;
    if (dbg_left < 0) {
        const char * ev = getenv("GGML_MOE_CACHE_DEBUG");
        dbg_left = ev ? atoi(ev) : 0;
    }
    if (dbg_left > 0) {
        dbg_left--;
        fprintf(stderr, "moe-cache dbg: eval=%lld il=%d draws=%lld hits=%d misses=%d evict=%d [",
                (long long) r.eval, st.il, (long long) n_ids, n_hits, n_misses, n_evict);
        for (int64_t i = 0; i < n_ids; i++) {
            const int32_t e = st.draw[i];
            fprintf(stderr, " %d:%d", e,
                    (e >= 0 && e < st.n_expert) ? st.expert_slot[e] : -1);
        }
        fprintf(stderr, " ]\n");
    }

    r.stats.steps++;
    r.stats.draws          += n_ids;
    r.stats.hits           += n_hits;
    r.stats.misses         += n_misses;
    r.stats.evictions      += n_evict;
    r.stats.experts_copied += n_misses;
    if (n_misses == 0) {
        r.stats.empty_admits++;
    }
    return 1;
}

// ---------------------------------------------------------------------------
// accounting
// ---------------------------------------------------------------------------

void ggml_moe_cache_get_stats(ggml_moe_cache_stats * out) {
    if (out) {
        *out = reg().stats;
    }
}

void ggml_moe_cache_report(void) {
    const registry & r = reg();
    if (!r.enabled || r.stats.steps == 0) {
        return;
    }
    const ggml_moe_cache_stats & s = r.stats;
    const double hit = s.draws ? 100.0 * (double) s.hits / (double) s.draws : 0.0;
    const int n_layers = (int) r.layers.size();
    const double tokens = n_layers ? (double) s.steps / (double) n_layers : 0.0;
    const double mib_tok = tokens > 0.0 ? (double) s.bytes_h2d / tokens / (1024.0*1024.0) : 0.0;

    fprintf(stderr,
        "moe-cache: %lld layer-steps over %d layers (%.0f tokens), "
        "%lld draws, %lld hits, %lld misses, hit=%.1f%%\n",
        (long long) s.steps, n_layers, tokens,
        (long long) s.draws, (long long) s.hits, (long long) s.misses, hit);
    fprintf(stderr,
        "moe-cache: %lld evictions, %lld experts copied, %.2f GiB admitted H2D, "
        "%.1f MiB/token, %lld all-resident layer-steps\n",
        (long long) s.evictions, (long long) s.experts_copied,
        (double) s.bytes_h2d / (1024.0*1024.0*1024.0), mib_tok,
        (long long) s.empty_admits);
}
