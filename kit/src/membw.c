// membw: memory-bandwidth and NUMA-placement probe for the perseus lab kit.
//
//   membw topo [--shell]
//       NUMA nodes, their CPUs, and one-CPU-per-core lists. --shell prints
//       shell assignments for run.sh instead of JSON.
//   membw read [--cpus LIST | --node-cpus N] [--phys] [--threads N]
//              [--policy default|local|bind:N|interleave] [--mb TOTAL] [--secs S] [--reps R] [--label L]
//       Each thread pins itself, applies the memory policy, first-touches its own
//       slice, then streams reads over it. Reports aggregate GB/s and where the
//       pages actually landed (move_pages query), so a policy that silently did
//       nothing shows up in the output instead of in the numbers.
//   membw exec [--interleave | --membind N] [--cpunode N] [--phys] -- CMD [ARGS...]
//       Applies the policy/affinity to itself, then execs CMD (both are inherited),
//       so the kit doesn't depend on numactl being installed.
//
// Output is one JSON object per line. Failures print {"error": ...} and exit non-zero:
// a policy that can't be applied is an error, never a silent fallback.

#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/syscall.h>
#endif
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#define MAX_CPUS 1024
#define MAX_NODES 64
#define MPOL_DEFAULT_ 0
#define MPOL_BIND_ 2
#define MPOL_INTERLEAVE_ 3
#define PLACEMENT_SAMPLES 256

typedef struct {
    int n;
    int cpu[MAX_CPUS];
} cpulist_t;

static int g_nodes;
static int g_node_id[MAX_NODES];
static cpulist_t g_node_cpus[MAX_NODES];
static long g_node_mem_kb[MAX_NODES];
static int g_numa_sysfs;

static void die_json(const char *what, int err) {
    printf("{\"tool\":\"membw\",\"error\":\"%s\",\"errno\":%d,\"strerror\":\"%s\"}\n", what, err, strerror(err));
    fflush(stdout);
    exit(2);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Parses "0-5,12-17" into out. Returns 0 on success.
static int parse_cpulist(const char *s, cpulist_t *out) {
    out->n = 0;
    while (*s && *s != '\n') {
        char *end;
        long a = strtol(s, &end, 10);
        if (end == s) return -1;
        long b = a;
        s = end;
        if (*s == '-') {
            b = strtol(s + 1, &end, 10);
            if (end == s + 1) return -1;
            s = end;
        }
        for (long c = a; c <= b; c++) {
            if (out->n >= MAX_CPUS) return -1;
            out->cpu[out->n++] = (int)c;
        }
        if (*s == ',') s++;
    }
    return 0;
}

static int read_file(const char *path, char *buf, size_t cap) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    size_t n = fread(buf, 1, cap - 1, f);
    fclose(f);
    buf[n] = 0;
    return 0;
}

static void load_topology(void) {
    char path[256], buf[8192];
    g_nodes = 0;
    for (int id = 0; id < MAX_NODES; id++) {
        snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", id);
        if (read_file(path, buf, sizeof buf) != 0) continue;
        if (parse_cpulist(buf, &g_node_cpus[g_nodes]) != 0) die_json("bad cpulist in sysfs", EINVAL);
        g_node_id[g_nodes] = id;
        g_node_mem_kb[g_nodes] = -1;
        snprintf(path, sizeof path, "/sys/devices/system/node/node%d/meminfo", id);
        if (read_file(path, buf, sizeof buf) == 0) {
            char *p = strstr(buf, "MemTotal:");
            if (p) g_node_mem_kb[g_nodes] = strtol(p + 9, NULL, 10);
        }
        g_nodes++;
    }
    g_numa_sysfs = g_nodes > 0;
    if (!g_numa_sysfs) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        if (n < 1 || n > MAX_CPUS) die_json("cannot count CPUs", EINVAL);
        g_nodes = 1;
        g_node_id[0] = 0;
        g_node_mem_kb[0] = -1;
        g_node_cpus[0].n = (int)n;
        for (int i = 0; i < n; i++) g_node_cpus[0].cpu[i] = i;
    }
}

// Keeps only the first hardware thread of each physical core (SMT siblings dropped).
static void filter_phys(cpulist_t *l) {
    char path[256], buf[256];
    int w = 0;
    for (int i = 0; i < l->n; i++) {
        int cpu = l->cpu[i];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        cpulist_t sib;
        if (read_file(path, buf, sizeof buf) != 0 || parse_cpulist(buf, &sib) != 0 || sib.n == 0) {
            l->cpu[w++] = cpu;  // no topology info: keep, it's reported as-is
            continue;
        }
        int min = sib.cpu[0];
        for (int j = 1; j < sib.n; j++) if (sib.cpu[j] < min) min = sib.cpu[j];
        if (min == cpu) l->cpu[w++] = cpu;
    }
    l->n = w;
}

static int node_index(int node) {
    for (int i = 0; i < g_nodes; i++) if (g_node_id[i] == node) return i;
    return -1;
}

static void print_list(const cpulist_t *l, const char *sep) {
    for (int i = 0; i < l->n; i++) printf("%s%d", i ? sep : "", l->cpu[i]);
}

static int cmd_topo(int shell) {
    load_topology();
    cpulist_t all = {0}, all_phys = {0};
    if (shell) printf("NUMA_SYSFS=%d\nNODES=%d\nNODE_IDS=\"", g_numa_sysfs, g_nodes);
    for (int i = 0; shell && i < g_nodes; i++) printf("%s%d", i ? " " : "", g_node_id[i]);
    if (shell) printf("\"\n");
    if (!shell) printf("{\"tool\":\"membw\",\"mode\":\"topo\",\"numa_sysfs\":%s,\"nodes\":[", g_numa_sysfs ? "true" : "false");
    for (int i = 0; i < g_nodes; i++) {
        cpulist_t phys = g_node_cpus[i];
        filter_phys(&phys);
        for (int j = 0; j < g_node_cpus[i].n; j++) all.cpu[all.n++] = g_node_cpus[i].cpu[j];
        for (int j = 0; j < phys.n; j++) all_phys.cpu[all_phys.n++] = phys.cpu[j];
        if (shell) {
            printf("NODE%d_CPUS=\"", g_node_id[i]); print_list(&g_node_cpus[i], ","); printf("\"\n");
            printf("NODE%d_PHYS=\"", g_node_id[i]); print_list(&phys, ","); printf("\"\n");
            printf("NODE%d_NCPUS=%d\nNODE%d_NPHYS=%d\nNODE%d_MEM_KB=%ld\n", g_node_id[i], g_node_cpus[i].n,
                   g_node_id[i], phys.n, g_node_id[i], g_node_mem_kb[i]);
        } else {
            printf("%s{\"id\":%d,\"mem_kb\":%ld,\"cpus\":[", i ? "," : "", g_node_id[i], g_node_mem_kb[i]);
            print_list(&g_node_cpus[i], ",");
            printf("],\"phys_cpus\":[");
            print_list(&phys, ",");
            printf("]}");
        }
    }
    if (shell) {
        printf("NCPUS=%d\nNPHYS=%d\nALL_CPUS=\"", all.n, all_phys.n); print_list(&all, ","); printf("\"\n");
        printf("ALL_PHYS=\"");
        print_list(&all_phys, ",");
        printf("\"\n");
    } else {
        printf("],\"ncpus\":%d,\"nphys\":%d}\n", all.n, all_phys.n);
    }
    return 0;
}

#ifdef __linux__
static long set_policy(int mode, int node /* -1 = all nodes */) {
    unsigned long mask[MAX_NODES / (8 * sizeof(unsigned long)) + 1];
    memset(mask, 0, sizeof mask);
    if (mode == MPOL_DEFAULT_) return syscall(SYS_set_mempolicy, MPOL_DEFAULT_, NULL, 0);
    for (int i = 0; i < g_nodes; i++) {
        int id = g_node_id[i];
        if (node >= 0 && id != node) continue;
        mask[id / (8 * sizeof(unsigned long))] |= 1UL << (id % (8 * sizeof(unsigned long)));
    }
    return syscall(SYS_set_mempolicy, mode, mask, (unsigned long)(8 * sizeof mask));
}

static int pin_to(const cpulist_t *l) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < l->n; i++) CPU_SET(l->cpu[i], &set);
    return sched_setaffinity(0, sizeof set, &set);
}
#endif

// Minimal barrier: macOS has no pthread_barrier_t.
typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int count, waiting, gen;
} barrier_t;

static void barrier_init(barrier_t *b, int count) {
    pthread_mutex_init(&b->mu, NULL);
    pthread_cond_init(&b->cv, NULL);
    b->count = count;
    b->waiting = 0;
    b->gen = 0;
}

static void barrier_wait(barrier_t *b) {
    pthread_mutex_lock(&b->mu);
    int gen = b->gen;
    if (++b->waiting == b->count) {
        b->waiting = 0;
        b->gen++;
        pthread_cond_broadcast(&b->cv);
    } else {
        while (gen == b->gen) pthread_cond_wait(&b->cv, &b->mu);
    }
    pthread_mutex_unlock(&b->mu);
}

static uint64_t read_pass(const uint8_t *p, size_t bytes) {
#if defined(__SSE2__)
    __m128i a0 = _mm_setzero_si128(), a1 = a0, a2 = a0, a3 = a0;
    const __m128i *v = (const __m128i *)p;
    size_t n = bytes / 64;
    for (size_t i = 0; i < n; i++) {
        a0 = _mm_add_epi64(a0, _mm_load_si128(v + 4 * i));
        a1 = _mm_add_epi64(a1, _mm_load_si128(v + 4 * i + 1));
        a2 = _mm_add_epi64(a2, _mm_load_si128(v + 4 * i + 2));
        a3 = _mm_add_epi64(a3, _mm_load_si128(v + 4 * i + 3));
    }
    a0 = _mm_add_epi64(_mm_add_epi64(a0, a1), _mm_add_epi64(a2, a3));
    uint64_t out[2];
    _mm_storeu_si128((__m128i *)out, a0);
    return out[0] ^ out[1];
#else
    const uint64_t *v = (const uint64_t *)p;
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    size_t n = bytes / 32;
    for (size_t i = 0; i < n; i++) {
        a0 += v[4 * i];
        a1 += v[4 * i + 1];
        a2 += v[4 * i + 2];
        a3 += v[4 * i + 3];
    }
    return a0 ^ a1 ^ a2 ^ a3;
#endif
}

typedef struct {
    int cpu;  // -1 = unpinned
    size_t bytes;
    int policy_mode, policy_node;
    double secs;
    int reps;
    barrier_t *bar;
    int err;
    const char *err_what;
    double *rep_gbps;  // per rep, this thread's GB/s
    int placement[MAX_NODES + 1];  // last slot = unknown/failed
    volatile uint64_t sink;
} worker_t;

static void *worker(void *arg) {
    worker_t *w = (worker_t *)arg;
    uint8_t *buf = NULL;
#ifdef __linux__
    if (w->cpu >= 0) {
        cpulist_t one = {.n = 1};
        one.cpu[0] = w->cpu;
        if (pin_to(&one) != 0) { w->err = errno; w->err_what = "sched_setaffinity"; }
    }
    if (!w->err && w->policy_mode != -1 && set_policy(w->policy_mode, w->policy_node) != 0) {
        w->err = errno;
        w->err_what = "set_mempolicy";
    }
#endif
    if (!w->err) {
        void *p = mmap(NULL, w->bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { w->err = errno; w->err_what = "mmap"; }
        else { buf = (uint8_t *)p; memset(buf, 1, w->bytes); }  // first touch places the pages
    }
#ifdef __linux__
    if (!w->err) {
        void *pages[PLACEMENT_SAMPLES];
        int status[PLACEMENT_SAMPLES];
        long pg = sysconf(_SC_PAGESIZE);
        size_t npages = w->bytes / (size_t)pg;
        for (int i = 0; i < PLACEMENT_SAMPLES; i++) pages[i] = buf + (npages * (size_t)i / PLACEMENT_SAMPLES) * (size_t)pg;
        if (syscall(SYS_move_pages, 0, PLACEMENT_SAMPLES, pages, NULL, status, 0) == 0) {
            for (int i = 0; i < PLACEMENT_SAMPLES; i++) {
                int ni = status[i] >= 0 ? node_index(status[i]) : -1;
                w->placement[ni >= 0 ? ni : MAX_NODES]++;
            }
        } else {
            w->placement[MAX_NODES] = PLACEMENT_SAMPLES;
        }
    }
#endif
    for (int r = 0; r < w->reps; r++) {
        barrier_wait(w->bar);  // everyone starts together, so the aggregate is concurrent
        if (w->err) { barrier_wait(w->bar); continue; }
        double t0 = now_s(), t;
        size_t done = 0;
        do {
            w->sink ^= read_pass(buf, w->bytes);
            done += w->bytes;
            t = now_s();
        } while (t - t0 < w->secs);
        w->rep_gbps[r] = done / (t - t0) / 1e9;
        barrier_wait(w->bar);
    }
    if (buf) munmap(buf, w->bytes);
    return NULL;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int cmd_read(int argc, char **argv) {
    load_topology();
    cpulist_t cpus = {0};
    int have_cpus = 0, phys = 0, threads = 0, reps = 3, policy_mode = -1, policy_node = -1;
    double secs = 2.0, mb = 2048;
    const char *label = "", *policy_str = "default";
    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--cpus") && v) { if (parse_cpulist(v, &cpus) != 0) die_json("bad --cpus", EINVAL); have_cpus = 1; i++; }
        else if (!strcmp(a, "--node-cpus") && v) {
            int ni = node_index(atoi(v));
            if (ni < 0) die_json("unknown node in --node-cpus", EINVAL);
            cpus = g_node_cpus[ni]; have_cpus = 1; i++;
        }
        else if (!strcmp(a, "--phys")) phys = 1;
        else if (!strcmp(a, "--threads") && v) { threads = atoi(v); i++; }
        else if (!strcmp(a, "--mb") && v) { mb = atof(v); i++; }
        else if (!strcmp(a, "--secs") && v) { secs = atof(v); i++; }
        else if (!strcmp(a, "--reps") && v) { reps = atoi(v); i++; }
        else if (!strcmp(a, "--label") && v) { label = v; i++; }
        else if (!strcmp(a, "--policy") && v) {
            policy_str = v; i++;
            if (!strcmp(v, "default")) policy_mode = -1;
            else if (!strcmp(v, "local")) policy_mode = MPOL_DEFAULT_;
            else if (!strcmp(v, "interleave")) policy_mode = MPOL_INTERLEAVE_;
            else if (!strncmp(v, "bind:", 5)) {
                policy_mode = MPOL_BIND_; policy_node = atoi(v + 5);
                if (node_index(policy_node) < 0) die_json("unknown node in --policy bind:N", EINVAL);
            }
            else die_json("bad --policy", EINVAL);
        }
        else die_json("unknown read argument", EINVAL);
    }
#ifndef __linux__
    if (policy_mode != -1) die_json("memory policies need Linux", ENOSYS);
#endif
    if (!have_cpus) {
        for (int i = 0; i < g_nodes; i++)
            for (int j = 0; j < g_node_cpus[i].n; j++) cpus.cpu[cpus.n++] = g_node_cpus[i].cpu[j];
    }
    if (phys) filter_phys(&cpus);
    if (threads <= 0) threads = cpus.n;
    if (threads <= 0 || threads > MAX_CPUS || reps < 1 || reps > 100 || secs <= 0 || mb < 16) die_json("bad read parameters", EINVAL);

    size_t per = (size_t)(mb * 1024 * 1024 / threads);
    per -= per % 4096;
    if (per < 4096 * 64) die_json("buffer per thread too small; raise --mb", EINVAL);

    barrier_t bar;
    barrier_init(&bar, threads);
    worker_t *w = calloc((size_t)threads, sizeof *w);
    pthread_t *tid = calloc((size_t)threads, sizeof *tid);
    double *gbps = calloc((size_t)threads * (size_t)reps, sizeof *gbps);
    if (!w || !tid || !gbps) die_json("calloc", ENOMEM);
    for (int t = 0; t < threads; t++) {
#ifdef __linux__
        w[t].cpu = cpus.cpu[t % cpus.n];
#else
        w[t].cpu = -1;
#endif
        w[t].bytes = per;
        w[t].policy_mode = policy_mode;
        w[t].policy_node = policy_node;
        w[t].secs = secs;
        w[t].reps = reps;
        w[t].bar = &bar;
        w[t].rep_gbps = gbps + (size_t)t * (size_t)reps;
        if (pthread_create(&tid[t], NULL, worker, &w[t]) != 0) die_json("pthread_create", errno);
    }
    for (int t = 0; t < threads; t++) pthread_join(tid[t], NULL);
    for (int t = 0; t < threads; t++) if (w[t].err) die_json(w[t].err_what, w[t].err);

    double rep_total[100];
    for (int r = 0; r < reps; r++) {
        rep_total[r] = 0;
        for (int t = 0; t < threads; t++) rep_total[r] += gbps[(size_t)t * (size_t)reps + (size_t)r];
    }
    double sorted[100];
    memcpy(sorted, rep_total, sizeof(double) * (size_t)reps);
    qsort(sorted, (size_t)reps, sizeof(double), cmp_double);

    int placement[MAX_NODES + 1] = {0};
    for (int t = 0; t < threads; t++)
        for (int n = 0; n <= MAX_NODES; n++) placement[n] += w[t].placement[n];

    printf("{\"tool\":\"membw\",\"mode\":\"read\",\"label\":\"%s\",\"policy\":\"%s\",\"threads\":%d,\"phys_only\":%s,"
           "\"mb\":%.0f,\"secs\":%.2f,\"cpus\":[", label, policy_str, threads, phys ? "true" : "false", mb, secs);
    for (int t = 0; t < threads; t++) printf("%s%d", t ? "," : "", w[t].cpu);
    printf("],\"gbps_best\":%.3f,\"gbps_median\":%.3f,\"reps_gbps\":[", sorted[reps - 1], sorted[reps / 2]);
    for (int r = 0; r < reps; r++) printf("%s%.3f", r ? "," : "", rep_total[r]);
    printf("],\"placement_pages\":{");
    for (int n = 0; n < g_nodes; n++) printf("%s\"%d\":%d", n ? "," : "", g_node_id[n], placement[n]);
    printf("%s\"unknown\":%d},\"sse2\":%s}\n", g_nodes ? "," : "", placement[MAX_NODES],
#if defined(__SSE2__)
           "true"
#else
           "false"
#endif
    );
    free(w); free(tid); free(gbps);
    return 0;
}

static int cmd_exec(int argc, char **argv) {
#ifndef __linux__
    (void)argc; (void)argv;
    die_json("exec needs Linux", ENOSYS);
    return 2;
#else
    load_topology();
    int mode = -1, node = -1, cpunode = -1, phys = 0, i = 0;
    for (; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--")) { i++; break; }
        if (!strcmp(a, "--interleave")) mode = MPOL_INTERLEAVE_;
        else if (!strcmp(a, "--membind") && i + 1 < argc) { mode = MPOL_BIND_; node = atoi(argv[++i]); }
        else if (!strcmp(a, "--cpunode") && i + 1 < argc) cpunode = atoi(argv[++i]);
        else if (!strcmp(a, "--phys")) phys = 1;
        else die_json("unknown exec argument", EINVAL);
    }
    if (i >= argc) die_json("exec needs -- CMD", EINVAL);
    if (node >= 0 && node_index(node) < 0) die_json("unknown node in --membind", EINVAL);
    if (mode != -1 && set_policy(mode, node) != 0) die_json("set_mempolicy", errno);
    if (cpunode >= 0) {
        int ni = node_index(cpunode);
        if (ni < 0) die_json("unknown node in --cpunode", EINVAL);
        cpulist_t l = g_node_cpus[ni];
        if (phys) filter_phys(&l);
        if (pin_to(&l) != 0) die_json("sched_setaffinity", errno);
    }
    fflush(stdout);
    execvp(argv[i], &argv[i]);
    die_json("execvp", errno);
    return 2;
#endif
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "topo")) return cmd_topo(argc >= 3 && !strcmp(argv[2], "--shell"));
    if (argc >= 2 && !strcmp(argv[1], "read")) return cmd_read(argc - 2, argv + 2);
    if (argc >= 2 && !strcmp(argv[1], "exec")) return cmd_exec(argc - 2, argv + 2);
    fprintf(stderr, "usage: membw topo [--shell] | read [opts] | exec [opts] -- CMD...\n");
    return 2;
}
