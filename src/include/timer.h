/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_TIMER_H_
#define NCCL_TIMER_H_
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#if defined(__i386__) || defined(__x86_64__)
#include <x86intrin.h>
#endif

static inline uint64_t timer_get_cycles(void) {
#if defined(__i386__) || defined(__x86_64__)
  return __rdtsc();
#elif defined(__aarch64__)
  uint64_t cycles;
  asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));
  return cycles;
#else
  struct timespec ts;
#ifdef CLOCK_MONOTONIC_RAW
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
  clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
  return (uint64_t)ts.tv_sec*1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

static double freq = -1;

static void calibrate() {
#if defined(__aarch64__)
  uint64_t freqHz;
  asm volatile("mrs %0, cntfrq_el0" : "=r"(freqHz));
  freq = (double)freqHz/1.0E6;
  return;
#elif !defined(__i386__) && !defined(__x86_64__)
  freq = 1000.0;
  return;
#endif

  struct timeval tv;
  gettimeofday(&tv, NULL);
  uint64_t timeCycles = timer_get_cycles();
  double time = - tv.tv_sec*1E6 - tv.tv_usec;
  for (int i=0; i<10000; i++) (void)timer_get_cycles();
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
  return freq;
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
