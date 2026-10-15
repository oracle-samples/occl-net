/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/


#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <sys/time.h>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>

namespace {

/** @brief Scripts deterministic x86 cycle and wall-clock readings. */
struct TimerBoundaryState {
  uint64_t firstCycle = 1000;
  uint64_t subsequentCycle = 11000;
  int cycleCalls = 0;
  int timeCalls = 0;
  std::array<timeval, 2> times = {{{2, 0}, {2, 100}}};
};

TimerBoundaryState timerBoundaryState;

/**
 * @brief Resets deterministic x86 cycle and wall-clock sources.
 * @note Side effects: Replaces timerBoundaryState.
 */
void ResetTimerBoundariesForTest() {
  timerBoundaryState = TimerBoundaryState{};
}

/**
 * @brief Returns the next deterministic architecture cycle sample.
 * @return firstCycle for the first sample and subsequentCycle thereafter.
 * @note Side effects: Increments cycleCalls.
 */
uint64_t ReadCyclesForTimerTest() {
  ++timerBoundaryState.cycleCalls;
  return timerBoundaryState.cycleCalls == 1
      ? timerBoundaryState.firstCycle
      : timerBoundaryState.subsequentCycle;
}

/**
 * @brief Returns the next deterministic wall-clock sample.
 * @param[out] value Valid non-NULL storage receiving one configured timeval.
 * @param[in] timezone Unused timezone argument, expected to be NULL.
 * @return Zero.
 * @note Side effects: Writes value and increments timerBoundaryState.timeCalls.
 */
int ReadTimeForTimerTest(timeval* value, void* timezone) {
  (void)timezone;
  const int index = timerBoundaryState.timeCalls < 2 ? timerBoundaryState.timeCalls : 1;
  *value = timerBoundaryState.times[index];
  ++timerBoundaryState.timeCalls;
  return 0;
}

}  // namespace

/**
 * @brief Instantiates timer.h on x86 with deterministic cycle and wall-clock sources for
 * calibration tests.
 */
#define __rdtsc ReadCyclesForTimerTest
#define gettimeofday ReadTimeForTimerTest
#endif
#include "timer.h"
#if defined(__x86_64__) || defined(__i386__)
#undef gettimeofday
#undef __rdtsc
#endif

/**
 * @brief Verifies that architecture cycle and frequency sources expose their platform contracts.
 */
TEST(TimerTest, ReadsArchitectureClock) {
#if defined(__x86_64__) || defined(__i386__)
  SCOPED_TRACE(
      "The x86 timer must read cycles directly and report that runtime calibration is required.");
  ResetTimerBoundariesForTest();
  EXPECT_EQ(timer_get_cycles(), 1000u)
      << "The cycle helper must return the architecture source value unchanged";
  EXPECT_EQ(timerBoundaryState.cycleCalls, 1)
      << "One direct cycle read must sample the architecture source exactly once";
  EXPECT_DOUBLE_EQ(timer_arch_cycles_per_us(), -1.0)
      << "x86 must request runtime calibration because the cycle frequency is not architectural";
#else
  SCOPED_TRACE("The AArch64 timer must expose a positive architectural frequency and readable counter.");
  const uint64_t first = timer_get_cycles();
  const uint64_t second = timer_get_cycles();
  EXPECT_GE(second, first) << "The architectural virtual counter must not move backwards";
  EXPECT_GT(timer_arch_cycles_per_us(), 0.0)
      << "AArch64 must convert its architectural counter frequency to cycles per microsecond";
#endif
}

/**
 * @brief Verifies that calibration uses the architecture frequency or elapsed wall time.
 */
TEST(TimerTest, CalibratesCycleFrequency) {
#if defined(__x86_64__) || defined(__i386__)
  SCOPED_TRACE("x86 calibration must divide the observed cycle delta by the observed microseconds.");
  ResetTimerBoundariesForTest();
  freq = -1.0;
  calibrate();
  EXPECT_DOUBLE_EQ(freq, 100.0)
      << "Calibration must compute cycles per microsecond from the two deterministic samples";
  EXPECT_EQ(timerBoundaryState.timeCalls, 2)
      << "Runtime calibration must read wall time before and after sampling";
  EXPECT_GE(timerBoundaryState.cycleCalls, 2)
      << "Calibration must sample cycles at both elapsed-time boundaries";
#else
  SCOPED_TRACE("AArch64 calibration must use the architectural frequency without wall-clock estimation.");
  freq = -1.0;
  const double expected = timer_arch_cycles_per_us();
  calibrate();
  EXPECT_DOUBLE_EQ(freq, expected)
      << "Calibration must retain the platform-provided cycles-per-microsecond value";
#endif
}

/**
 * @brief Verifies that converted time lazily calibrates and then reuses the cached frequency.
 */
TEST(TimerTest, ConvertsCyclesWithCachedFrequency) {
  SCOPED_TRACE("Time conversion and frequency access must lazily calibrate once and then reuse that result.");
#if defined(__x86_64__) || defined(__i386__)
  ResetTimerBoundariesForTest();
  timerBoundaryState.firstCycle = 120;
  timerBoundaryState.subsequentCycle = 120;
  freq = 4.0;
  EXPECT_DOUBLE_EQ(gettime(), 30.0)
      << "Converted time must divide the current cycle value by cycles per microsecond";
  EXPECT_EQ(timerBoundaryState.cycleCalls, 1)
      << "Converted time with cached calibration must read cycles exactly once";
  EXPECT_EQ(timerBoundaryState.timeCalls, 0)
      << "Converted time with cached calibration must not read wall time";
  EXPECT_DOUBLE_EQ(timer_cycles_per_us(), 4.0)
      << "The frequency accessor must return the cached calibration exactly";

  ResetTimerBoundariesForTest();
  freq = -1.0;
  EXPECT_DOUBLE_EQ(gettime(), 110.0)
      << "An uninitialized time conversion must calibrate before converting the next cycle sample";
  EXPECT_DOUBLE_EQ(freq, 100.0)
      << "Lazy time conversion must retain the calibrated cycles-per-microsecond value";
  EXPECT_EQ(timerBoundaryState.timeCalls, 2)
      << "Lazy time conversion must perform exactly one wall-clock calibration";
  const int calibratedCycleCalls = timerBoundaryState.cycleCalls;
  EXPECT_DOUBLE_EQ(timer_cycles_per_us(), 100.0)
      << "Frequency access after lazy conversion must return the cached calibration";
  EXPECT_EQ(timerBoundaryState.cycleCalls, calibratedCycleCalls)
      << "Reading a cached frequency must not resample the cycle source";

  ResetTimerBoundariesForTest();
  freq = -1.0;
  EXPECT_DOUBLE_EQ(timer_cycles_per_us(), 100.0)
      << "An uninitialized frequency access must trigger calibration";
  EXPECT_EQ(timerBoundaryState.timeCalls, 2)
      << "Lazy frequency access must perform exactly one wall-clock calibration";
#else
  freq = -1.0;
  EXPECT_GT(timer_cycles_per_us(), 0.0)
      << "An uninitialized AArch64 frequency access must use the architectural calibration";
  const double calibratedFrequency = freq;
  EXPECT_GE(gettime(), 0.0) << "Converted architectural time must be nonnegative";
  EXPECT_DOUBLE_EQ(timer_cycles_per_us(), calibratedFrequency)
      << "The frequency accessor must return the cached calibration exactly";
#endif
}
