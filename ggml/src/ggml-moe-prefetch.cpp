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
    const ggml_tensor * w;   // attribution only; never dereferenced off the enqueue path
};

// ---------------------------------------------------------------------------
// residency topology  (GGML_MOE_RESIDENCY_TOPOLOGY=1, off by default)
//
// A worker already pays one mincore() per chunk to decide whether to skip it.
// With this on, that same vector is walked to the end instead of short-circuiting
// on the first missing page, so the *shape* of the hole pattern -- how many
// contiguous missing runs and how long they are -- is recorded alongside how long
// MADV_POPULATE_READ then took on that exact chunk. No extra syscall is issued and
// nothing is populated that would not have been; the skip decision is bit-identical.
//
// Everything below is per worker and merged at shutdown, so no worker ever touches
// another worker's cache line and no atomic is added to the hot path.
// ---------------------------------------------------------------------------

static bool topo_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_MOE_RESIDENCY_TOPOLOGY");
        return e && atoi(e) != 0;
    }();
    return on;
}

static constexpr int TOPO_RUN_BUCKETS  = 6;    // missing-run length: 1, 2-4, 5-16, 17-64, 65-256, >256 pages
static constexpr int TOPO_MISS_BUCKETS = 8;    // missing pages in the chunk
static constexpr int TOPO_TL_SECONDS   = 1800; // 1 s timeline buckets; past that, folded into the last

static const char * const TOPO_RUN_NAMES[TOPO_RUN_BUCKETS] =
    { "1", "2-4", "5-16", "17-64", "65-256", "257+" };
static const char * const TOPO_MISS_NAMES[TOPO_MISS_BUCKETS] =
    { "1-8", "9-32", "33-64", "65-128", "129-192", "193-320", "321-448", "449+" };

static inline int topo_run_bucket(uint32_t len) {
    if (len <=   1) return 0;
    if (len <=   4) return 1;
    if (len <=  16) return 2;
    if (len <=  64) return 3;
    if (len <= 256) return 4;
    return 5;
}

static inline int topo_miss_bucket(uint32_t m) {
    if (m <=   8) return 0;
    if (m <=  32) return 1;
    if (m <=  64) return 2;
    if (m <= 128) return 3;
    if (m <= 192) return 4;
    if (m <= 320) return 5;
    if (m <= 448) return 6;
    return 7;
}

struct topo_scan {
    bool     ok       = false;  // mincore answered for this chunk
    uint32_t pages    = 0;
    uint32_t resident = 0;
    uint32_t missing  = 0;
    uint32_t runs     = 0;      // contiguous missing runs
    uint32_t transit  = 0;      // resident<->missing boundaries inside the chunk
    uint32_t max_run  = 0;
    uint32_t singles  = 0;      // missing runs of exactly one page
    uint32_t run_pages[TOPO_RUN_BUCKETS] = {0};
    uint32_t run_count[TOPO_RUN_BUCKETS] = {0};
};

struct topo_bucket {
    uint64_t chunks = 0, skipped = 0, failed = 0, unknown = 0;
    uint64_t pages = 0, resident = 0, missing = 0;
    uint64_t runs = 0, transit = 0, singles = 0, max_run_sum = 0;
    uint64_t pop_chunks = 0, pop_ns = 0, pop_bytes = 0, pop_missing = 0;
    uint64_t run_pages[TOPO_RUN_BUCKETS] = {0};
    uint64_t run_count[TOPO_RUN_BUCKETS] = {0};
};

struct topo_cell {   // 2D: chunk missing-page count x that chunk's mean missing-run length
    uint64_t n = 0, ns = 0, missing = 0;
};

struct topo_tensor {
    const char * name = nullptr;
    uint64_t chunks = 0, skipped = 0;
    uint64_t pages = 0, resident = 0, missing = 0;
    uint64_t runs = 0, singles = 0, max_run_sum = 0;
    uint64_t pop_chunks = 0, pop_ns = 0;
    double   t_first = 0.0, t_last = 0.0;
};

struct topo_rec {
    double       t;
    const char * name;
    uint64_t     off;
    uint32_t     pages, resident, missing, runs, max_run, singles, transit;
    uint32_t     pop_us;
    uint8_t      skipped;
};

static constexpr size_t TOPO_REC_CAP = 1024;   // per worker; adaptively decimated to cover the whole run

struct topo_worker {
    std::vector<topo_bucket>  tl;
    topo_cell                 cell[TOPO_MISS_BUCKETS][TOPO_RUN_BUCKETS];
    std::unordered_map<const void *, topo_tensor> tensors;
    std::vector<topo_rec>     recs;
    uint64_t seen = 0, stride = 1, overflow = 0;

    void init() {
        tl.assign(TOPO_TL_SECONDS, topo_bucket{});
        recs.reserve(TOPO_REC_CAP);
    }

    // deterministic decimation: keep every stride-th record, and when the buffer
    // fills, drop every other one and double the stride. The surviving sample is
    // spread over the whole run rather than being its tail.
    void sample(const topo_rec & r) {
        if ((seen++ % stride) != 0) return;
        recs.push_back(r);
        if (recs.size() < TOPO_REC_CAP) return;
        size_t k = 0;
        for (size_t i = 0; i < recs.size(); i += 2) recs[k++] = recs[i];
        recs.resize(k);
        stride *= 2;
    }
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

    // residency topology: empty unless GGML_MOE_RESIDENCY_TOPOLOGY is set
    std::vector<topo_worker>              topo;
    std::chrono::steady_clock::time_point topo_t0;
    double                                topo_t0_unix = 0.0;

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
        topo_dump();
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
        if (topo_enabled()) {
            topo.resize(n_threads);
            for (auto & t : topo) t.init();
            topo_t0      = std::chrono::steady_clock::now();
            topo_t0_unix = std::chrono::duration<double>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
        }
        workers.reserve(n_threads);
        for (int i = 0; i < n_threads; ++i) {
            workers.emplace_back([this, i] {
                char name[16];
                snprintf(name, sizeof(name), "moe-pf-%d", i);
                pthread_setname_np(pthread_self(), name);
                run(i);
            });
        }
    }

    // one mincore() either way. ts == nullptr is the original short-circuit: stop at
    // the first missing page, because all the caller wants is the skip decision. With
    // ts, the same vector is walked to the end and the hole pattern is described. The
    // returned skip decision is identical in both modes.
    static bool chunk_scan(uintptr_t addr, size_t len, topo_scan * ts) {
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
        if (!ts) {
            for (size_t i = 0; i < npages; ++i) {
                if (!(vec[i] & 1)) return false;
            }
            return true;
        }
        ts->ok    = true;
        ts->pages = (uint32_t) npages;
        uint32_t run  = 0;
        int      prev = -1;
        for (size_t i = 0; i < npages; ++i) {
            const int res = vec[i] & 1;
            if (res) { ts->resident++; } else { ts->missing++; }
            if (prev >= 0 && res != prev) ts->transit++;
            prev = res;
            if (!res) {
                run++;
            } else if (run) {
                ts->runs++;
                if (run > ts->max_run) ts->max_run = run;
                if (run == 1) ts->singles++;
                const int b = topo_run_bucket(run);
                ts->run_pages[b] += run;
                ts->run_count[b]++;
                run = 0;
            }
        }
        if (run) {
            ts->runs++;
            if (run > ts->max_run) ts->max_run = run;
            if (run == 1) ts->singles++;
            const int b = topo_run_bucket(run);
            ts->run_pages[b] += run;
            ts->run_count[b]++;
        }
        return ts->missing == 0;
    }

    static bool chunk_resident(uintptr_t addr, size_t len) {
        return chunk_scan(addr, len, nullptr);
    }

    void topo_record(topo_worker & tw, const job & j, const topo_scan & ts,
                     std::chrono::steady_clock::time_point tj,
                     uint64_t pop_ns, size_t pop_bytes, bool skipped, bool failed) {
        const double dt = std::chrono::duration<double>(tj - topo_t0).count();
        int idx = (int) dt;
        if (idx < 0) idx = 0;
        if (idx >= TOPO_TL_SECONDS) { idx = TOPO_TL_SECONDS - 1; tw.overflow++; }
        topo_bucket & b = tw.tl[idx];

        b.chunks++;
        if (skipped) b.skipped++;
        if (failed)  b.failed++;
        if (!ts.ok)  { b.unknown++; }
        else {
            b.pages       += ts.pages;
            b.resident    += ts.resident;
            b.missing     += ts.missing;
            b.runs        += ts.runs;
            b.transit     += ts.transit;
            b.singles     += ts.singles;
            b.max_run_sum += ts.max_run;
            for (int k = 0; k < TOPO_RUN_BUCKETS; k++) {
                b.run_pages[k] += ts.run_pages[k];
                b.run_count[k] += ts.run_count[k];
            }
        }
        if (!skipped) {
            b.pop_chunks++;
            b.pop_ns     += pop_ns;
            b.pop_bytes  += pop_bytes;
            b.pop_missing += ts.ok ? ts.missing : 0;
        }

        // 2D: how much was missing x how it was shaped, against what populate cost
        if (ts.ok && ts.missing > 0 && !skipped) {
            const uint32_t mean_run = ts.runs ? (ts.missing + ts.runs/2)/ts.runs : 0;
            topo_cell & c = tw.cell[topo_miss_bucket(ts.missing)][topo_run_bucket(mean_run)];
            c.n++;
            c.ns      += pop_ns;
            c.missing += ts.missing;
        }

        if (j.w && tw.tensors.size() < 4096) {
            topo_tensor & t = tw.tensors[j.w];
            if (!t.name) { t.name = j.w->name; t.t_first = dt; }
            t.t_last = dt;
            t.chunks++;
            if (skipped) t.skipped++;
            if (ts.ok) {
                t.pages += ts.pages; t.resident += ts.resident; t.missing += ts.missing;
                t.runs  += ts.runs;  t.singles  += ts.singles;  t.max_run_sum += ts.max_run;
            }
            if (!skipped) { t.pop_chunks++; t.pop_ns += pop_ns; }
        }

        topo_rec r;
        r.t        = topo_t0_unix + dt;
        r.name     = j.w ? j.w->name : "";
        r.off      = j.w ? (uint64_t)(j.addr - (uintptr_t) j.w->data) : 0;
        r.pages    = ts.pages;
        r.resident = ts.resident;
        r.missing  = ts.missing;
        r.runs     = ts.runs;
        r.max_run  = ts.max_run;
        r.singles  = ts.singles;
        r.transit  = ts.transit;
        r.pop_us   = (uint32_t) std::min<uint64_t>(pop_ns/1000, 0xffffffffull);
        r.skipped  = skipped ? 1 : 0;
        tw.sample(r);
    }

    // merge every worker's slots and write the whole thing out once, at shutdown.
    // GGML_MOE_RESIDENCY_TOPOLOGY_OUT=<path> gets the JSON; stderr always gets one
    // summary line so server.log carries the headline on its own.
    void topo_dump() {
        if (topo.empty()) return;

        topo_bucket total;
        std::vector<topo_bucket> tl(TOPO_TL_SECONDS);
        topo_cell cell[TOPO_MISS_BUCKETS][TOPO_RUN_BUCKETS];
        std::unordered_map<const void *, topo_tensor> tensors;
        std::vector<topo_rec> recs;
        uint64_t stride_max = 1, overflow = 0;

        auto acc = [](topo_bucket & d, const topo_bucket & s) {
            d.chunks += s.chunks; d.skipped += s.skipped; d.failed += s.failed;
            d.unknown += s.unknown; d.pages += s.pages; d.resident += s.resident;
            d.missing += s.missing; d.runs += s.runs; d.transit += s.transit;
            d.singles += s.singles; d.max_run_sum += s.max_run_sum;
            d.pop_chunks += s.pop_chunks; d.pop_ns += s.pop_ns;
            d.pop_bytes += s.pop_bytes; d.pop_missing += s.pop_missing;
            for (int k = 0; k < TOPO_RUN_BUCKETS; k++) {
                d.run_pages[k] += s.run_pages[k];
                d.run_count[k] += s.run_count[k];
            }
        };

        for (const auto & w : topo) {
            for (int i = 0; i < TOPO_TL_SECONDS; i++) { acc(tl[i], w.tl[i]); acc(total, w.tl[i]); }
            for (int a = 0; a < TOPO_MISS_BUCKETS; a++) {
                for (int b = 0; b < TOPO_RUN_BUCKETS; b++) {
                    cell[a][b].n       += w.cell[a][b].n;
                    cell[a][b].ns      += w.cell[a][b].ns;
                    cell[a][b].missing += w.cell[a][b].missing;
                }
            }
            for (const auto & kv : w.tensors) {
                topo_tensor & t = tensors[kv.first];
                const topo_tensor & s = kv.second;
                if (!t.name) { t.name = s.name; t.t_first = s.t_first; }
                t.t_first = std::min(t.t_first, s.t_first);
                t.t_last  = std::max(t.t_last,  s.t_last);
                t.chunks += s.chunks; t.skipped += s.skipped; t.pages += s.pages;
                t.resident += s.resident; t.missing += s.missing; t.runs += s.runs;
                t.singles += s.singles; t.max_run_sum += s.max_run_sum;
                t.pop_chunks += s.pop_chunks; t.pop_ns += s.pop_ns;
            }
            recs.insert(recs.end(), w.recs.begin(), w.recs.end());
            stride_max = std::max(stride_max, w.stride);
            overflow  += w.overflow;
        }
        std::sort(recs.begin(), recs.end(),
                  [](const topo_rec & a, const topo_rec & b) { return a.t < b.t; });

        const double mean_run = total.runs ? (double) total.missing/total.runs : 0.0;
        fprintf(stderr,
                "moe-residency-topology: chunks=%llu skipped=%llu scanned_pages=%llu resident=%.1f%% "
                "missing_runs=%llu mean_missing_run=%.1f pages (%.1f KiB) max_run_mean=%.1f "
                "single_page_runs=%.1f%% of runs (%.1f%% of missing pages) populate_mean=%.0f us "
                "samples=%zu stride=%llu\n",
                (unsigned long long) total.chunks, (unsigned long long) total.skipped,
                (unsigned long long) total.pages,
                total.pages ? 100.0*(double) total.resident/total.pages : 0.0,
                (unsigned long long) total.runs, mean_run, mean_run*4.0,
                total.chunks ? (double) total.max_run_sum/total.chunks : 0.0,
                total.runs ? 100.0*(double) total.run_count[0]/total.runs : 0.0,
                total.missing ? 100.0*(double) total.run_pages[0]/total.missing : 0.0,
                total.pop_chunks ? (double) total.pop_ns/total.pop_chunks/1000.0 : 0.0,
                recs.size(), (unsigned long long) stride_max);

        const char * out = getenv("GGML_MOE_RESIDENCY_TOPOLOGY_OUT");
        if (!out || !*out) return;
        FILE * f = fopen(out, "w");
        if (!f) {
            fprintf(stderr, "moe-residency-topology: cannot write %s\n", out);
            return;
        }
        auto bfields = [&](FILE * fp, const topo_bucket & b) {
            fprintf(fp, "\"chunks\":%llu,\"skipped\":%llu,\"failed\":%llu,\"unknown\":%llu,"
                    "\"pages\":%llu,\"resident\":%llu,\"missing\":%llu,\"runs\":%llu,"
                    "\"transitions\":%llu,\"singles\":%llu,\"max_run_sum\":%llu,"
                    "\"pop_chunks\":%llu,\"pop_ns\":%llu,\"pop_bytes\":%llu,\"pop_missing\":%llu,"
                    "\"run_pages\":[",
                    (unsigned long long) b.chunks, (unsigned long long) b.skipped,
                    (unsigned long long) b.failed, (unsigned long long) b.unknown,
                    (unsigned long long) b.pages, (unsigned long long) b.resident,
                    (unsigned long long) b.missing, (unsigned long long) b.runs,
                    (unsigned long long) b.transit, (unsigned long long) b.singles,
                    (unsigned long long) b.max_run_sum, (unsigned long long) b.pop_chunks,
                    (unsigned long long) b.pop_ns, (unsigned long long) b.pop_bytes,
                    (unsigned long long) b.pop_missing);
            for (int k = 0; k < TOPO_RUN_BUCKETS; k++)
                fprintf(fp, "%s%llu", k ? "," : "", (unsigned long long) b.run_pages[k]);
            fprintf(fp, "],\"run_count\":[");
            for (int k = 0; k < TOPO_RUN_BUCKETS; k++)
                fprintf(fp, "%s%llu", k ? "," : "", (unsigned long long) b.run_count[k]);
            fprintf(fp, "]");
        };

        fprintf(f, "{\n\"schema\":\"moe-residency-topology/1\",\n");
        fprintf(f, "\"t0_unix\":%.3f,\n\"page_bytes\":%ld,\n\"chunk_bytes\":%zu,\n",
                topo_t0_unix, sysconf(_SC_PAGESIZE), (size_t) GGML_MOE_PREFETCH_CHUNK);
        fprintf(f, "\"workers\":%zu,\n\"sample_stride\":%llu,\n\"timeline_overflow\":%llu,\n",
                topo.size(), (unsigned long long) stride_max, (unsigned long long) overflow);
        fprintf(f, "\"run_buckets\":[");
        for (int k = 0; k < TOPO_RUN_BUCKETS; k++) fprintf(f, "%s\"%s\"", k ? "," : "", TOPO_RUN_NAMES[k]);
        fprintf(f, "],\n\"miss_buckets\":[");
        for (int k = 0; k < TOPO_MISS_BUCKETS; k++) fprintf(f, "%s\"%s\"", k ? "," : "", TOPO_MISS_NAMES[k]);
        fprintf(f, "],\n\"totals\":{");
        bfields(f, total);
        fprintf(f, "},\n\"timeline\":[\n");
        bool first = true;
        for (int i = 0; i < TOPO_TL_SECONDS; i++) {
            if (tl[i].chunks == 0) continue;
            fprintf(f, "%s{\"t\":%.1f,", first ? "" : ",\n", topo_t0_unix + i + 0.5);
            bfields(f, tl[i]);
            fprintf(f, "}");
            first = false;
        }
        fprintf(f, "\n],\n\"cells\":[\n");
        first = true;
        for (int a = 0; a < TOPO_MISS_BUCKETS; a++) {
            for (int b = 0; b < TOPO_RUN_BUCKETS; b++) {
                if (cell[a][b].n == 0) continue;
                fprintf(f, "%s{\"miss\":\"%s\",\"run\":\"%s\",\"n\":%llu,\"ns\":%llu,\"missing\":%llu}",
                        first ? "" : ",\n", TOPO_MISS_NAMES[a], TOPO_RUN_NAMES[b],
                        (unsigned long long) cell[a][b].n, (unsigned long long) cell[a][b].ns,
                        (unsigned long long) cell[a][b].missing);
                first = false;
            }
        }
        fprintf(f, "\n],\n\"by_tensor\":[\n");
        first = true;
        for (const auto & kv : tensors) {
            const topo_tensor & t = kv.second;
            fprintf(f, "%s{\"name\":\"%s\",\"t_first\":%.1f,\"t_last\":%.1f,\"chunks\":%llu,"
                    "\"skipped\":%llu,\"pages\":%llu,\"resident\":%llu,\"missing\":%llu,"
                    "\"runs\":%llu,\"singles\":%llu,\"max_run_sum\":%llu,\"pop_chunks\":%llu,"
                    "\"pop_ns\":%llu}",
                    first ? "" : ",\n", t.name ? t.name : "", t.t_first, t.t_last,
                    (unsigned long long) t.chunks, (unsigned long long) t.skipped,
                    (unsigned long long) t.pages, (unsigned long long) t.resident,
                    (unsigned long long) t.missing, (unsigned long long) t.runs,
                    (unsigned long long) t.singles, (unsigned long long) t.max_run_sum,
                    (unsigned long long) t.pop_chunks, (unsigned long long) t.pop_ns);
            first = false;
        }
        fprintf(f, "\n],\n\"samples\":[\n");
        first = true;
        for (const auto & r : recs) {
            fprintf(f, "%s{\"t\":%.3f,\"name\":\"%s\",\"off\":%llu,\"pages\":%u,\"resident\":%u,"
                    "\"missing\":%u,\"runs\":%u,\"max_run\":%u,\"singles\":%u,\"transitions\":%u,"
                    "\"pop_us\":%u,\"skipped\":%u}",
                    first ? "" : ",\n", r.name ? r.name : "", (unsigned long long) r.off,
                    r.pages, r.resident, r.missing, r.runs, r.max_run, r.singles, r.transit,
                    r.pop_us, (unsigned) r.skipped);
            first = false;
        }
        fprintf(f, "\n]\n}\n");
        fclose(f);
        fprintf(stderr, "moe-residency-topology: wrote %s\n", out);
        topo.clear();   // one dump per pool lifetime
    }

    void run(int wid) {
        const long page = sysconf(_SC_PAGESIZE);
        topo_worker * tw = ((size_t) wid < topo.size()) ? &topo[wid] : nullptr;
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
            topo_scan ts;
            std::chrono::steady_clock::time_point tj;
            if (tw) tj = std::chrono::steady_clock::now();
            uint64_t pop_ns    = 0;
            size_t   pop_bytes = 0;
            bool     skipped   = false;
            bool     failed    = false;
            // skip when every page is already resident; keeps the drain rate
            // high when the cache is warm and is what keeps this from re-reading
            // bytes the load-time WILLNEED or a previous micro-batch brought in
            if (chunk_scan(j.addr, j.len, tw ? &ts : nullptr)) {
                n_skipped.fetch_add(1, std::memory_order_relaxed);
                skipped = true;
            } else {
                const uintptr_t astart = j.addr & ~(uintptr_t)(page - 1);
                const size_t    alen   = ((j.addr + j.len + page - 1) & ~(uintptr_t)(page - 1)) - astart;
                const auto t0 = tw ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
                const int rc = madvise((void *)astart, alen, MADV_POPULATE_READ);
                if (tw) {
                    pop_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t0).count();
                }
                if (rc == 0) {
                    bytes_populated.fetch_add(alen, std::memory_order_relaxed);
                    pop_bytes = alen;
                } else {
                    n_failed.fetch_add(1, std::memory_order_relaxed); // the fault path takes over
                    failed = true;
                }
            }
            if (tw) topo_record(*tw, j, ts, tj, pop_ns, pop_bytes, skipped, failed);
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
            jobs.push_back({(uintptr_t)w->data + o, (uint32_t)len, tk, w});
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
