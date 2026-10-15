// BACKLOG §19.25: a CPU neighbour streaming memory at a fixed rate beside decode. THREADS threads, each pinned to
// its own core (the first THREADS of the CPU list given), copy 64 MB blocks between two 512 MB buffers, so each copy
// reads and writes (bus traffic = 2 x the bytes copied); a thread sleeps to hold its share of RATE GB/s of traffic.
// Every 10 s it prints the traffic reached. Runs until killed.
//   bus_neighbour RATE_GBS THREADS FIRST_CPU
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double rate_per_thread;
static atomic_ullong moved;
static int first_cpu;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static void* run(void* arg) {
  const long id = (long)arg;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(first_cpu + id, &set);
  pthread_setaffinity_np(pthread_self(), sizeof set, &set);
  const size_t buf = 512ul << 20, blk = 64ul << 20;
  char *a = aligned_alloc(4096, buf), *b = aligned_alloc(4096, buf);
  memset(a, 1, buf), memset(b, 2, buf);
  const double t0 = now();
  double done = 0;  // bytes of traffic this thread has made
  for (size_t off = 0;; off = (off + blk) % buf) {
    memcpy(b + off, a + off, blk);
    done += 2.0 * blk;
    atomic_fetch_add(&moved, 2ull * blk);
    const double due = done / (rate_per_thread * 1e9) + t0 - now();
    if (due > 0) usleep((useconds_t)(due * 1e6));
  }
  return NULL;
}

int main(int argc, char** argv) {
  if (argc < 4) return fprintf(stderr, "usage: bus_neighbour RATE_GBS THREADS FIRST_CPU\n"), 2;
  const double rate = atof(argv[1]);
  const int threads = atoi(argv[2]);
  first_cpu = atoi(argv[3]);
  rate_per_thread = rate / threads;
  for (long i = 0; i < threads; ++i) {
    pthread_t t;
    pthread_create(&t, NULL, run, (void*)i);
  }
  unsigned long long last = 0;
  for (;;) {
    sleep(10);
    const unsigned long long m = atomic_load(&moved);
    printf("%.1f GB/s\n", (m - last) / 10e9);
    fflush(stdout);
    last = m;
  }
}
