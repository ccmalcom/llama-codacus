#include "ggml-moe-prefetch.h"

#if defined(__linux__)

#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

// chunk granularity; big enough to amortize syscall cost, small enough to
// spread one expert (~0.5-0.9 MiB here, runs of ~1-2 MiB) across workers
static constexpr size_t GGML_MOE_PREFETCH_CHUNK = 2u*1024u*1024u;

// MADV_POPULATE_READ brings pages into the page cache AND this mm's page
// tables, so the consumer takes no faults at all. Linux >= 5.14.
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif

namespace {

struct mapping_entry {
    uintptr_t base;
    size_t    size;
};

struct ticket {
    std::atomic<int> pending{0};
    uint64_t         epoch = 0;
};

struct job {
    uintptr_t addr;
    uint32_t  len;
    std::shared_ptr<ticket> tk;
};

struct prefetch_pool {
    std::mutex               mtx;
    std::condition_variable  cv_work;   // workers sleep here
    std::condition_variable  cv_done;   // waiters sleep here
    std::deque<job>          queue;
    std::vector<std::thread> workers;
    bool                     shutdown = false;

    std::unordered_map<const void *, std::shared_ptr<ticket>> tickets;

    // cumulative observability counters (printed at shutdown under GGML_MOE_PREFETCH_DEBUG)
    std::atomic<uint64_t> n_jobs{0};
    std::atomic<uint64_t> n_skipped{0};        // chunks already fully resident (mincore)
    std::atomic<uint64_t> n_failed{0};         // madvise returned an error
    std::atomic<uint64_t> bytes_enqueued{0};   // requested by the scheduler
    std::atomic<uint64_t> bytes_populated{0};  // actually passed to MADV_POPULATE_READ
    std::atomic<uint64_t> n_enqueues{0};       // ggml_moe_prefetch_mask calls that queued work
    std::atomic<uint64_t> n_ranges{0};         // coalesced expert runs enqueued
    std::atomic<uint64_t> n_epochs_used{0};    // scheduler passes that enqueued anything
    std::atomic<uint64_t> n_waits{0};          // waits that actually blocked
    std::atomic<uint64_t> wait_ns{0};          // time the calling thread spent blocked
    size_t                max_queue = 0;       // guarded by mtx
    uint64_t              last_epoch_seen = 0; // guarded by mtx

    ~prefetch_pool() { stop(); }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            shutdown = true;
            // complete outstanding tickets so no waiter blocks forever
            for (auto & j : queue) {
                if (j.tk) {
                    j.tk->pending.fetch_sub(1, std::memory_order_acq_rel);
                }
            }
            queue.clear();
            cv_work.notify_all();
            cv_done.notify_all();
        }
        for (auto & w : workers) {
            if (w.joinable()) w.join();
        }
        workers.clear();
        if (getenv("GGML_MOE_PREFETCH_DEBUG") && n_jobs.load() > 0) {
            const double gib = 1024.0*1024.0*1024.0;
            fprintf(stderr, "moe-host-prefetch: jobs=%llu skipped_resident=%llu failed=%llu enqueues=%llu ranges=%llu "
                    "bytes_enqueued=%.2f GiB bytes_populated=%.2f GiB passes=%llu max_queue=%zu waits_blocked=%llu wait_s=%.1f\n",
                    (unsigned long long) n_jobs.load(), (unsigned long long) n_skipped.load(),
                    (unsigned long long) n_failed.load(), (unsigned long long) n_enqueues.load(),
                    (unsigned long long) n_ranges.load(),
                    (double) bytes_enqueued.load()/gib, (double) bytes_populated.load()/gib,
                    (unsigned long long) n_epochs_used.load(), max_queue,
                    (unsigned long long) n_waits.load(), (double) wait_ns.load()/1e9);
        }
    }

    void start(int n_threads) {
        stop();
        std::lock_guard<std::mutex> lock(mtx);
        shutdown = false;
        workers.reserve(n_threads);
        for (int i = 0; i < n_threads; ++i) {
            workers.emplace_back([this, i] {
                char name[16];
                snprintf(name, sizeof(name), "moe-pf-%d", i);
                pthread_setname_np(pthread_self(), name);
                run();
            });
        }
    }

    static bool chunk_resident(uintptr_t addr, size_t len) {
        const long page = sysconf(_SC_PAGESIZE);
        const uintptr_t astart = addr & ~(uintptr_t)(page - 1);
        const size_t    alen   = (addr + len) - astart;
        const size_t    npages = (alen + page - 1)/page;
        if (npages > GGML_MOE_PREFETCH_CHUNK/4096 + 2) {
            return false;
        }
        unsigned char vec[GGML_MOE_PREFETCH_CHUNK/4096 + 2];
        if (mincore((void *)astart, alen, vec) != 0) {
            return false;
        }
        for (size_t i = 0; i < npages; ++i) {
            if (!(vec[i] & 1)) return false;
        }
        return true;
    }

    void run() {
        const long page = sysconf(_SC_PAGESIZE);
        for (;;) {
            job j;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv_work.wait(lock, [this] { return shutdown || !queue.empty(); });
                if (shutdown) return;
                j = std::move(queue.front());
                queue.pop_front();
            }
            n_jobs.fetch_add(1, std::memory_order_relaxed);
            // skip when every page is already resident; keeps the drain rate
            // high when the cache is warm and is what keeps this from re-reading
            // bytes the load-time WILLNEED or a previous micro-batch brought in
            if (chunk_resident(j.addr, j.len)) {
                n_skipped.fetch_add(1, std::memory_order_relaxed);
            } else {
                const uintptr_t astart = j.addr & ~(uintptr_t)(page - 1);
                const size_t    alen   = ((j.addr + j.len + page - 1) & ~(uintptr_t)(page - 1)) - astart;
                if (madvise((void *)astart, alen, MADV_POPULATE_READ) == 0) {
                    bytes_populated.fetch_add(alen, std::memory_order_relaxed);
                } else {
                    n_failed.fetch_add(1, std::memory_order_relaxed); // the fault path takes over
                }
            }
            if (j.tk) {
                const int left = j.tk->pending.fetch_sub(1, std::memory_order_acq_rel);
                if (left == 1) {
                    std::lock_guard<std::mutex> lock(mtx);
                    cv_done.notify_all();
                }
            }
        }
    }
};

struct prefetch_state {
    std::mutex                 reg_mtx;
    std::vector<mapping_entry> mappings;

    std::atomic<uint64_t> epoch{1};

    std::mutex                     pool_mtx;
    std::shared_ptr<prefetch_pool> pool;
};

static prefetch_state & state() {
    static prefetch_state s;
    return s;
}

// true when [p, p+len) lies inside a registered mmap
static bool is_mapped(const void * p, size_t len) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.reg_mtx);
    const uintptr_t a = (uintptr_t)p;
    for (const auto & m : s.mappings) {
        if (a >= m.base && a + len <= m.base + m.size) return true;
    }
    return false;
}

// coalesce the set bits of `seen` into [offset, offset+len) ranges of w
static void collect_mask_ranges(const ggml_tensor * w, const uint32_t * seen, int64_t n_as, size_t tail_pad,
        std::vector<std::pair<size_t, size_t>> & ranges) {
    const size_t stride = w->nb[2];
    const size_t wbytes = ggml_nbytes(w);

    int64_t id = 0;
    while (id < n_as) {
        while (id < n_as && !(seen[id >> 5] & (1u << (id & 31)))) ++id;
        if (id >= n_as) break;
        int64_t first = id;
        while (id < n_as && (seen[id >> 5] & (1u << (id & 31)))) ++id;
        const size_t off = (size_t)first*stride;
        const size_t len = std::min<size_t>((size_t)(id - first)*stride + tail_pad, wbytes - off);
        if (off < wbytes && len > 0) {
            ranges.emplace_back(off, len);
        }
    }
}

// enqueue ranges of tensor w at the queue head; returns 0 when the engine is off
// or w is not inside a registered mapping, 2 when w was already enqueued this
// epoch (nothing added), 1 when work was queued
static int enqueue_ranges(const ggml_tensor * w, const std::vector<std::pair<size_t, size_t>> & ranges) {
    auto & s = state();
    std::shared_ptr<prefetch_pool> pool_sp;
    {
        std::lock_guard<std::mutex> plock(s.pool_mtx);
        pool_sp = s.pool;
    }
    if (!pool_sp) return 0;

    if (!is_mapped(w->data, ggml_nbytes(w))) return 0;

    const uint64_t cur_epoch = s.epoch.load(std::memory_order_relaxed);

    prefetch_pool & pool = *pool_sp;
    std::lock_guard<std::mutex> lock(pool.mtx);
    if (pool.shutdown) return 0;

    auto & tk = pool.tickets[w];
    if (!tk) tk = std::make_shared<ticket>();
    if (tk->epoch == cur_epoch) {
        return 2; // already enqueued for this scheduler pass
    }
    tk->epoch = cur_epoch;

    std::vector<job> jobs;
    uint64_t bytes = 0;
    for (const auto & r : ranges) {
        bytes += r.second;
        for (size_t o = r.first; o < r.first + r.second; o += GGML_MOE_PREFETCH_CHUNK) {
            const size_t len = std::min(GGML_MOE_PREFETCH_CHUNK, r.first + r.second - o);
            jobs.push_back({(uintptr_t)w->data + o, (uint32_t)len, tk});
        }
    }
    if (jobs.empty()) return 2;

    pool.n_enqueues.fetch_add(1, std::memory_order_relaxed);
    pool.n_ranges.fetch_add(ranges.size(), std::memory_order_relaxed);
    pool.bytes_enqueued.fetch_add(bytes, std::memory_order_relaxed);
    if (pool.last_epoch_seen != cur_epoch) {
        pool.last_epoch_seen = cur_epoch;
        pool.n_epochs_used.fetch_add(1, std::memory_order_relaxed);
    }

    tk->pending.fetch_add((int)jobs.size(), std::memory_order_acq_rel);
    // urgent: the consumer waits on this tensor a few statements later
    pool.queue.insert(pool.queue.begin(), std::make_move_iterator(jobs.begin()), std::make_move_iterator(jobs.end()));
    pool.max_queue = std::max(pool.max_queue, pool.queue.size());
    pool.cv_work.notify_all();
    return 1;
}

static bool populate_read_supported() {
    const long page = sysconf(_SC_PAGESIZE);
    void * p = mmap(nullptr, (size_t)page, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return false;
    const bool ok = madvise(p, (size_t)page, MADV_POPULATE_READ) == 0;
    munmap(p, (size_t)page);
    return ok;
}

} // namespace

void ggml_moe_prefetch_register_mapping(const void * base, size_t size) {
    if (!base || size == 0) return;
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.reg_mtx);
    for (const auto & m : s.mappings) {
        if (m.base == (uintptr_t)base) return;
    }
    s.mappings.push_back({(uintptr_t)base, size});
}

void ggml_moe_prefetch_unregister_mapping(const void * base) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.reg_mtx);
    // a queued job may still point into this range; its madvise then fails
    // (ENOMEM once unmapped) and the worker counts it as failed
    s.mappings.erase(std::remove_if(s.mappings.begin(), s.mappings.end(),
                [base](const mapping_entry & m) { return m.base == (uintptr_t)base; }),
            s.mappings.end());
}

void ggml_moe_prefetch_set_n_threads(int n_threads) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.pool_mtx);
    if (n_threads <= 0) {
        s.pool.reset();
        return;
    }
    if (!populate_read_supported()) {
        fprintf(stderr, "%s: MADV_POPULATE_READ not supported; MoE host prefetch disabled\n", __func__);
        s.pool.reset();
        return;
    }
    if (s.pool && s.pool->workers.size() == (size_t)n_threads) {
        return;
    }
    s.pool.reset();
    s.pool = std::make_shared<prefetch_pool>();
    s.pool->start(n_threads);
    fprintf(stderr, "%s: MoE host prefetch pool started, %d worker threads, %zu KiB chunks\n",
            __func__, n_threads, GGML_MOE_PREFETCH_CHUNK/1024);
}

bool ggml_moe_prefetch_enabled(void) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.pool_mtx);
    return s.pool != nullptr;
}

void ggml_moe_prefetch_new_epoch(void) {
    state().epoch.fetch_add(1, std::memory_order_relaxed);
}

void ggml_moe_prefetch_mask(const struct ggml_tensor * w, const uint32_t * mask, int n_expert, size_t tail_pad) {
    if (!w || !w->data || !mask) return;
    const int64_t n_as = w->ne[2];
    if (n_as <= 1 || w->nb[2] == 0 || n_expert <= 0 || (int64_t) n_expert != n_as) return;

    std::vector<std::pair<size_t, size_t>> ranges;
    collect_mask_ranges(w, mask, n_as, tail_pad, ranges);
    const bool queued = enqueue_ranges(w, ranges);

    // GGML_MOE_PREFETCH_DEBUG=2: one line per enqueue that actually queued work
    // (epoch-deduplicated repeats are silent), for checking that what is populated
    // is exactly the routed cold set. Never on during a measurement.
    static const int debug = [] { const char * e = getenv("GGML_MOE_PREFETCH_DEBUG"); return e ? atoi(e) : 0; }();
    if (debug >= 2 && queued == 1) {
        int n_sel = 0;
        for (int e = 0; e < n_expert; ++e) {
            n_sel += (mask[e >> 5] >> (e & 31)) & 1u;
        }
        size_t bytes = 0;
        for (const auto & r : ranges) bytes += r.second;
        fprintf(stderr, "moe-host-prefetch: enqueue %s n_sel=%d/%d runs=%zu bytes=%.1f MiB\n",
                w->name, n_sel, n_expert, ranges.size(), (double) bytes/(1024.0*1024.0));
    }
}

void ggml_moe_prefetch_wait(const struct ggml_tensor * w) {
    auto & s = state();
    std::shared_ptr<prefetch_pool> pool;
    {
        std::lock_guard<std::mutex> plock(s.pool_mtx);
        pool = s.pool;
    }
    if (!pool) return;
    std::shared_ptr<ticket> tk;
    {
        std::lock_guard<std::mutex> lock(pool->mtx);
        auto it = pool->tickets.find(w);
        if (it == pool->tickets.end()) return;
        tk = it->second;
    }
    if (tk->pending.load(std::memory_order_acquire) <= 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    {
        // the shared_ptr keeps the pool alive; shutdown wakes cv_done
        std::unique_lock<std::mutex> lock(pool->mtx);
        pool->cv_done.wait(lock, [&] {
            return pool->shutdown || tk->pending.load(std::memory_order_acquire) <= 0;
        });
    }
    pool->n_waits.fetch_add(1, std::memory_order_relaxed);
    pool->wait_ns.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count(), std::memory_order_relaxed);
}

#else // !__linux__

void ggml_moe_prefetch_register_mapping(const void *, size_t) {}
void ggml_moe_prefetch_unregister_mapping(const void *) {}
void ggml_moe_prefetch_set_n_threads(int) {}
bool ggml_moe_prefetch_enabled(void) { return false; }
void ggml_moe_prefetch_new_epoch(void) {}
void ggml_moe_prefetch_mask(const struct ggml_tensor *, const uint32_t *, int, size_t) {}
void ggml_moe_prefetch_wait(const struct ggml_tensor *) {}

#endif
