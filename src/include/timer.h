/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See third_party/licenses/Apache-2.0.txt for license information
 ************************************************************************/

#ifndef NCCL_TIMER_H_
#define NCCL_TIMER_H_
#include <stdint.h>
#include <unistd.h>
#include <sys/time.h>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
static inline uint64_t timer_get_cycles(void) {
  return __rdtsc();
}
static inline double timer_arch_cycles_per_us(void) {
  return -1.0;
}
#elif defined(__aarch64__)
static inline uint64_t timer_get_cycles(void) {
  uint64_t value;
  asm volatile("mrs %0, cntvct_el0" : "=r"(value));
  return value;
}
static inline double timer_arch_cycles_per_us(void) {
  uint64_t freq_hz;
  asm volatile("mrs %0, cntfrq_el0" : "=r"(freq_hz));
  return (double)freq_hz / 1.0e6;
}
#else
#error "timer.h supports only x86 (__rdtsc) and aarch64 (cntvct_el0)"
#endif

static double freq = -1;
static void calibrate() {
  double archFreq = timer_arch_cycles_per_us();
  if (archFreq > 0.0) {
    freq = archFreq;
    return;
  }
  struct timeval tv;
  gettimeofday(&tv, NULL);
  uint64_t timeCycles = timer_get_cycles();
  double time = - tv.tv_sec*1E6 - tv.tv_usec;
  uint64_t total = 0ULL;
  for (int i=0; i<10000; i++) total += timer_get_cycles();
  (void)total;
  gettimeofday(&tv, NULL);
  timeCycles = timer_get_cycles() - timeCycles;
  time += tv.tv_sec*1E6 + tv.tv_usec;
  freq = timeCycles/time;
}
static inline double gettime() {
  if (freq == -1) calibrate();
  return timer_get_cycles()/freq;
}
static inline double timer_cycles_per_us(void) {
  if (freq == -1) calibrate();
  return freq; // timer ticks per microsecond
}
#if ENABLE_TIMER
static uint64_t counts[8];
static double times[8];
static double startTimes[8];
#define TIME_START(index) do { \
  counts[index]++; \
  startTimes[index] = gettime(); \
} while (0)

#define TIME_STOP(index) do { \
  times[index] += gettime() - startTimes[index]; \
} while (0)

#define TIME_CANCEL(index) do { \
  counts[index]--; \
} while (0)

#define TIME_PRINT(name) do { \
  printf("%s stats", name); \
  for (int i=0; i<8; i++) { \
    if (counts[i]) printf(" [%d] %g/%ld = %g", i, times[i], counts[i], times[i]/counts[i]); \
    counts[i] = 0; \
  } \
  printf("\n"); \
} while (0)
#else
#define TIME_START(index) do {} while(0)
#define TIME_STOP(index) do {} while(0)
#define TIME_CANCEL(index) do {} while(0)
#define TIME_PRINT(name)
#endif
#endif
