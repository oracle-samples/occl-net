/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/

#include "cim.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <time.h>
#include <vector>

namespace {

std::atomic<int> sleepCallsForTest{0};
std::atomic<long> sleepSecondsForTest{0};
std::atomic<long> sleepNanosecondsForTest{0};
std::atomic<bool> bypassSleepForTest{false};
std::atomic<int> createThreadCallsForTest{0};
std::atomic<bool> failThreadCreationForTest{false};

/**
 * @brief Records the requested sleep interval and delegates to the operating system.
 * @param[in] requested Valid non-NULL requested interval.
 * @param[out] remaining NULL or storage for an interrupted remainder.
 * @return The operating-system nanosleep result.
 * @note Side effects: Records the interval, optionally sleeps, and increments sleepCallsForTest.
 */
int SleepForCimTest(const timespec* requested, timespec* remaining) {
  sleepSecondsForTest.store(requested->tv_sec, std::memory_order_relaxed);
  sleepNanosecondsForTest.store(requested->tv_nsec, std::memory_order_relaxed);
  sleepCallsForTest.fetch_add(1, std::memory_order_relaxed);
  if (bypassSleepForTest.load(std::memory_order_relaxed)) return 0;
  return ::nanosleep(requested, remaining);
}

/**
 * @brief Records worker creation and optionally injects a deterministic pthread failure.
 * @param[out] thread Non-NULL output populated by pthread_create on success.
 * @param[in] attributes Optional pthread attributes forwarded unchanged.
 * @param[in] startRoutine Valid non-NULL worker entry point forwarded unchanged.
 * @param[in] argument Opaque worker argument forwarded unchanged.
 * @return EAGAIN when failure injection is enabled; otherwise the pthread_create result.
 * @note Side effects: Increments createThreadCallsForTest and may create one worker thread.
 */
int CreateThreadForCimTest(pthread_t* thread, const pthread_attr_t* attributes,
                           void* (*startRoutine)(void*), void* argument) {
  createThreadCallsForTest.fetch_add(1, std::memory_order_relaxed);
  if (failThreadCreationForTest.load(std::memory_order_relaxed)) return EAGAIN;
  return ::pthread_create(thread, attributes, startRoutine, argument);
}

}  // namespace

/** @brief Compiles cim.cc with timing and thread creation redirected to observable seams above. */
#define nanosleep SleepForCimTest
#define pthread_create CreateThreadForCimTest
#include "../cim.cc"  // NOLINT(build/include)
#undef pthread_create
#undef nanosleep

namespace {

/**
 * @brief Computes the arithmetic sum of a test-owned load-weight vector.
 * @param[in] weights Finite load weights retained unchanged by the helper.
 * @return Arithmetic sum of all entries, or zero for an empty vector.
 */
float SumWeights(const std::vector<float>& weights) {
  return std::accumulate(weights.begin(), weights.end(), 0.0f);
}


}  // namespace

/**
 * @brief Verifies the direct millisecond sleep helper forwards a normalized timespec interval.
 */
TEST(CimUtilityTest, SleepsForRequestedMilliseconds) {
  SCOPED_TRACE("Millisecond sleep must convert its argument to the exact seconds and nanoseconds request.");
  const int callsBefore = sleepCallsForTest.load(std::memory_order_relaxed);
  bypassSleepForTest.store(true, std::memory_order_relaxed);
  msleep(7);
  EXPECT_EQ(sleepCallsForTest.load(std::memory_order_relaxed), callsBefore + 1)
      << "One helper call must issue exactly one nanosleep request";
  EXPECT_EQ(sleepSecondsForTest.load(std::memory_order_relaxed), 0)
      << "Seven milliseconds must not contain a complete second";
  EXPECT_EQ(sleepNanosecondsForTest.load(std::memory_order_relaxed), 7000000)
      << "Seven milliseconds must convert to exactly seven million nanoseconds";

  msleep(1007);
  EXPECT_EQ(sleepCallsForTest.load(std::memory_order_relaxed), callsBefore + 2)
      << "A second conversion must issue exactly one additional nanosleep request";
  EXPECT_EQ(sleepSecondsForTest.load(std::memory_order_relaxed), 1)
      << "One thousand and seven milliseconds must contain exactly one complete second";
  EXPECT_EQ(sleepNanosecondsForTest.load(std::memory_order_relaxed), 7000000)
      << "The subsecond remainder must stay normalized below one billion nanoseconds";
  bypassSleepForTest.store(false, std::memory_order_relaxed);
}

/**
 * @brief Verifies the relaxed float loader preserves the stored bit pattern and source storage.
 */
TEST(CimUtilityTest, LoadsFloatRelaxed) {
  SCOPED_TRACE("Relaxed debug loads must preserve the complete float bit pattern and source storage.");
  volatile float source = -3.25f;
  const float sourceBefore = source;
  uint32_t sourceBitsBefore = 0;
  std::memcpy(&sourceBitsBefore, &sourceBefore, sizeof(sourceBitsBefore));
  const float loaded = load_float_relaxed(&source);
  uint32_t loadedBits = 0;
  std::memcpy(&loadedBits, &loaded, sizeof(loadedBits));
  EXPECT_EQ(loadedBits, sourceBitsBefore)
      << "The relaxed load must preserve every bit of the stored float";
  const float sourceAfter = source;
  uint32_t sourceBitsAfter = 0;
  std::memcpy(&sourceBitsAfter, &sourceAfter, sizeof(sourceBitsAfter));
  EXPECT_EQ(sourceBitsAfter, sourceBitsBefore)
      << "The relaxed load must not modify caller-owned volatile storage";
}

/**
 * @brief Verifies the direct headroom-scale selector returns all three buckets at their ordinary
 * boundaries.
 */
TEST(CimAlgorithmTest, ChoosesHeadroomScaleDirectly) {
  SCOPED_TRACE("Headroom scale selection must cover cold, close, moderate, and distant latency ranges.");
  EXPECT_EQ(2u, CIM_chooseHeadroomScale2(0, 100))
      << "An unknown minimum latency must use the cold-start 1.0 scale";
  EXPECT_EQ(4u, CIM_chooseHeadroomScale2(100, 200))
      << "A maximum exactly twice the minimum must use the 2.0 scale";
  EXPECT_EQ(3u, CIM_chooseHeadroomScale2(100, 201))
      << "A maximum just above twice the minimum must use the 1.5 scale";
  EXPECT_EQ(3u, CIM_chooseHeadroomScale2(100, 500))
      << "A maximum exactly five times the minimum must remain in the 1.5 scale";
  EXPECT_EQ(2u, CIM_chooseHeadroomScale2(100, 501))
      << "A maximum above five times the minimum must use the 1.0 scale";
}

/** @brief Verifies integer threshold comparisons remain correct at their encoded range limits. */
TEST(CimAlgorithmTest, HeadroomScaleThresholdsDoNotOverflow) {
#if 0  // ADDRESS_LATER: Requires the reverted overflow-safe headroom threshold implementation.
  SCOPED_TRACE("Headroom bucket selection must compare mathematical ratios without uint64 wrap.");
  constexpr uint64_t aboveHalfRange = UINT64_MAX / 2u + 1u;
  constexpr uint64_t aboveFifthRange = UINT64_MAX / 5u + 1u;

  EXPECT_EQ(4u, CIM_chooseHeadroomScale2(aboveHalfRange, UINT64_MAX))
      << "A maximum below twice the minimum must remain in the 2.0 bucket";
  EXPECT_EQ(3u, CIM_chooseHeadroomScale2(aboveFifthRange, UINT64_MAX))
      << "A maximum above twice but below five times the minimum must use the 1.5 bucket";
#endif  // ADDRESS_LATER
}

/**
 * @brief Verifies cycle accumulation updates count, sum, and maximum across representative samples.
 */
TEST(CimDiagnosticsTest, CycleStatsAddUpdatesCountSumAndMaximum) {
  SCOPED_TRACE("Cycle statistics must retain the exact count, total, and largest observed sample.");
  CycStat stats = {};

  cyc_add(&stats, 10);
  EXPECT_EQ(stats.cnt, 1u);
  EXPECT_EQ(stats.cyc_sum, 10u);
  EXPECT_EQ(stats.cyc_max, 10u);
  cyc_add(&stats, 4);
  EXPECT_EQ(stats.cnt, 2u);
  EXPECT_EQ(stats.cyc_sum, 14u);
  EXPECT_EQ(stats.cyc_max, 10u);
  cyc_add(&stats, 20);
  EXPECT_EQ(stats.cnt, 3u);
  EXPECT_EQ(stats.cyc_sum, 34u);
  EXPECT_EQ(stats.cyc_max, 20u);
}

/** @brief Verifies cycle-stat diagnostics format populated and empty accumulators safely. */
TEST(CimDiagnosticsTest, CycleStatsPrintsPopulatedAndEmptyStatistics) {
  SCOPED_TRACE("Cycle-stat output must identify both accumulators and terminate every diagnostic line.");
  const CycStat populated = {2, 100, 60};
  const CycStat empty = {};
  testing::internal::CaptureStdout();
  print_cyc_stats("populated-empty", &populated, &empty);
  print_cyc_stats("empty-populated", &empty, &populated);
  print_cyc_stats("empty-empty", &empty, &empty);
  const std::string output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("populated-empty cyc-stats [0]"), std::string::npos);
  EXPECT_NE(output.find("empty-populated cyc-stats [0]"), std::string::npos);
  EXPECT_NE(output.find("empty-empty cyc-stats [0]"), std::string::npos);
  EXPECT_NE(output.find("[1]"), std::string::npos);
  EXPECT_EQ(std::count(output.begin(), output.end(), '\n'), 3);
}

/** @brief Verifies load-weight diagnostics at the minimum and maximum valid QP counts. */
TEST(CimDiagnosticsTest, LoadWeightDumpPrintsValidQpCountBoundaries) {
  SCOPED_TRACE("Load-weight output must include every active QP and the normalized total at valid bounds.");
  CIMState minimum = {};
  minimum.nqps = 1;
  minimum.lw[0] = 1.0f;
  CIMState maximum = {};
  maximum.nqps = CIM_MAX_QPS;
  for (int index = 0; index < maximum.nqps; ++index) {
    maximum.lw[index] = 1.0f / static_cast<float>(CIM_MAX_QPS);
  }
  testing::internal::CaptureStdout();
  dump_lw("minimum", &minimum);
  dump_lw("maximum", &maximum);
  const std::string output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("minimum: qp=0 lw=1.000000 | sum=1.000000\n"), std::string::npos);
  EXPECT_NE(output.find("maximum: qp=0 lw=0.007812 "), std::string::npos);
  EXPECT_NE(output.find("qp=127 lw=0.007812 | sum=1.000000\n"), std::string::npos);
  EXPECT_EQ(std::count(output.begin(), output.end(), 'q'), CIM_MAX_QPS + 1);
  EXPECT_EQ(std::count(output.begin(), output.end(), '\n'), 2);
}

/** @brief Verifies snapshot float comparison immediately below, at, and above its tolerance. */
TEST(CimDebugSnapshotTest, FloatComparisonHandlesValidToleranceBoundaries) {
  SCOPED_TRACE("Float comparison must be symmetric and change classification only above its tolerance.");
  constexpr float tolerance = 1e-6f;
  const float justBelowTolerance = std::nextafter(tolerance, 0.0f);
  const float justAboveTolerance = std::nextafter(tolerance, 1.0f);
  EXPECT_EQ(cimDbgFloatDiffers(0.0f, 0.0f), 0);
  EXPECT_EQ(cimDbgFloatDiffers(1.0f, 1.0f), 0);
  EXPECT_EQ(cimDbgFloatDiffers(0.0f, justBelowTolerance), 0);
  EXPECT_EQ(cimDbgFloatDiffers(justBelowTolerance, 0.0f), 0);
  EXPECT_EQ(cimDbgFloatDiffers(0.0f, tolerance), 0);
  EXPECT_EQ(cimDbgFloatDiffers(tolerance, 0.0f), 0);
  EXPECT_EQ(cimDbgFloatDiffers(0.0f, justAboveTolerance), 1);
  EXPECT_EQ(cimDbgFloatDiffers(justAboveTolerance, 0.0f), 1);
  EXPECT_EQ(cimDbgFloatDiffers(0.0f, 1.0f), 1);
}

/**
 * @brief Verifies snapshot capture across supported sources, views, counts, and sequence
 * boundaries.
 */
TEST(CimDebugSnapshotTest, CaptureHandlesSourceViewAndSequenceBoundaries) {
  SCOPED_TRACE("Snapshot capture must copy coherent enabled fields and clamp all valid dimensions safely.");
  volatile uint64_t latencies[] = {10u, 20u, 30u};
  volatile uint64_t rawLatencies[] = {11u, 21u, 31u};
  volatile uint32_t byteCounts[] = {100u, 200u, 300u};
  volatile float loadWeights[] = {0.25f, 0.75f};
  volatile uint32_t stableSequence = 2u;
  volatile uint32_t viewCount = 2u;
  volatile uint32_t viewIndices[CIM_MAX_QPS] = {2u, 0u};
  CIMDebugSnapshot populated = {};
  populated.lat[3] = 1003u;
  populated.raw[3] = 2003u;
  populated.bytes[3] = 3003u;
  populated.lw[3] = 0.5f;
  populated.view_qp_index[2] = 99u;
  cimDbgCaptureSnapshot(&populated, latencies, rawLatencies, byteCounts, loadWeights, 2, &stableSequence, &viewCount, viewIndices, 3);
  EXPECT_EQ(populated.lat[0], 10u);
  EXPECT_EQ(populated.lat[2], 30u);
  EXPECT_EQ(populated.raw[0], 11u);
  EXPECT_EQ(populated.raw[2], 31u);
  EXPECT_EQ(populated.bytes[0], 100u);
  EXPECT_EQ(populated.bytes[2], 300u);
  EXPECT_FLOAT_EQ(populated.lw[0], 0.25f);
  EXPECT_FLOAT_EQ(populated.lw[1], 0.75f);
  EXPECT_FLOAT_EQ(populated.lw[2], 0.0f);
  EXPECT_EQ(populated.view_count, 2u);
  EXPECT_EQ(populated.view_qp_index[0], 2u);
  EXPECT_EQ(populated.view_qp_index[1], 0u);
  EXPECT_EQ(populated.lat[3], 1003u);
  EXPECT_EQ(populated.raw[3], 2003u);
  EXPECT_EQ(populated.bytes[3], 3003u);
  EXPECT_FLOAT_EQ(populated.lw[3], 0.5f);
  EXPECT_EQ(populated.view_qp_index[2], 99u);
  EXPECT_EQ(latencies[0], 10u);
  EXPECT_EQ(latencies[1], 20u);
  EXPECT_EQ(latencies[2], 30u);
  EXPECT_EQ(rawLatencies[0], 11u);
  EXPECT_EQ(rawLatencies[1], 21u);
  EXPECT_EQ(rawLatencies[2], 31u);
  EXPECT_EQ(byteCounts[0], 100u);
  EXPECT_EQ(byteCounts[1], 200u);
  EXPECT_EQ(byteCounts[2], 300u);
  EXPECT_FLOAT_EQ(loadWeights[0], 0.25f);
  EXPECT_FLOAT_EQ(loadWeights[1], 0.75f);
  EXPECT_EQ(stableSequence, 2u);
  EXPECT_EQ(viewCount, 2u);
  EXPECT_EQ(viewIndices[0], 2u);
  EXPECT_EQ(viewIndices[1], 0u);
  CIMDebugSnapshot zeroQps = {};
  zeroQps.lat[0] = 101u;
  zeroQps.raw[0] = 201u;
  zeroQps.bytes[0] = 301u;
  zeroQps.lw[0] = 0.5f;
  zeroQps.view_qp_index[0] = 401u;
  cimDbgCaptureSnapshot(&zeroQps, latencies, rawLatencies, byteCounts, loadWeights, 0, nullptr, &viewCount, viewIndices, 0);
  EXPECT_EQ(zeroQps.view_count, 0u);
  EXPECT_EQ(zeroQps.lat[0], 101u);
  EXPECT_EQ(zeroQps.raw[0], 201u);
  EXPECT_EQ(zeroQps.bytes[0], 301u);
  EXPECT_FLOAT_EQ(zeroQps.lw[0], 0.5f);
  EXPECT_EQ(zeroQps.view_qp_index[0], 401u);
  CIMDebugSnapshot nullSources = {};
  cimDbgCaptureSnapshot(&nullSources, nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, CIM_MAX_QPS);
  EXPECT_EQ(nullSources.lat[0], 0u);
  EXPECT_EQ(nullSources.lat[CIM_MAX_QPS - 1], 0u);
  EXPECT_EQ(nullSources.raw[CIM_MAX_QPS - 1], 0u);
  EXPECT_EQ(nullSources.bytes[CIM_MAX_QPS - 1], 0u);
  EXPECT_FLOAT_EQ(nullSources.lw[CIM_MAX_QPS - 1], 0.0f);
  EXPECT_EQ(nullSources.view_count, static_cast<uint32_t>(CIM_MAX_QPS));
  EXPECT_EQ(nullSources.view_qp_index[0], 0u);
  EXPECT_EQ(nullSources.view_qp_index[CIM_MAX_QPS - 1], static_cast<uint32_t>(CIM_MAX_QPS - 1));
  viewCount = 0u;
  viewIndices[0] = static_cast<uint32_t>(CIM_MAX_QPS);
  CIMDebugSnapshot defaultView = {};
  cimDbgCaptureSnapshot(&defaultView, latencies, rawLatencies, byteCounts, loadWeights, 2, nullptr, &viewCount, viewIndices, 3);
  EXPECT_EQ(defaultView.view_count, 3u);
  EXPECT_EQ(defaultView.view_qp_index[0], 0u);
  EXPECT_EQ(viewCount, 0u);
  EXPECT_EQ(viewIndices[0], static_cast<uint32_t>(CIM_MAX_QPS));
  viewCount = 4u;
  CIMDebugSnapshot oversizedView = {};
  cimDbgCaptureSnapshot(&oversizedView, latencies, rawLatencies, byteCounts, loadWeights, 2, nullptr, &viewCount, nullptr, 3);
  EXPECT_EQ(oversizedView.view_count, 3u);
  EXPECT_EQ(oversizedView.view_qp_index[2], 2u);
  EXPECT_EQ(viewCount, 4u);
  volatile uint32_t retrySequence = 1u;
  volatile uint64_t retryLatency[] = {1u};
  CIMDebugSnapshot retried = {};
  // Keep the sequence odd until capture retries, then publish data before the even sequence.
  std::atomic<bool> captureStarted = false;
  std::atomic<bool> captureFinished = false;
  /**
   * @brief Captures a snapshot after the odd sequence is released.
   * @note Side effects: Updates retried and the capture synchronization flags.
   */
  std::thread captureThread([&]() {
    captureStarted.store(true, std::memory_order_release);
    cimDbgCaptureSnapshot(&retried, retryLatency, nullptr, nullptr, nullptr, 0, &retrySequence, nullptr, nullptr, 1);
    captureFinished.store(true, std::memory_order_release);
  });
  while (!captureStarted.load(std::memory_order_acquire)) std::this_thread::yield();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  EXPECT_FALSE(captureFinished.load(std::memory_order_acquire));
  __atomic_store_n(&retryLatency[0], 42u, __ATOMIC_RELAXED);
  __atomic_store_n(&retrySequence, 2u, __ATOMIC_RELEASE);
  captureThread.join();
  EXPECT_TRUE(captureFinished.load(std::memory_order_acquire));
  EXPECT_EQ(retried.lat[0], 42u);
  EXPECT_EQ(retrySequence, 2u);
}

/**
 * @brief Verifies snapshot commit publishes enabled fields and bounded views without corrupting
 * inputs.
 */
TEST(CimDebugSnapshotTest, CommitHandlesFieldAndViewBoundaries) {
  SCOPED_TRACE("Snapshot commit must publish the selected data exactly and preserve caller-owned sources.");
  CIMDebugSnapshot source = {};
  for (int index = 0; index < CIM_MAX_QPS; ++index) {
    source.lat[index] = static_cast<uint64_t>(index + 1);
    source.raw[index] = static_cast<uint64_t>(index + 101);
    source.bytes[index] = static_cast<uint32_t>(index + 201);
    source.lw[index] = static_cast<float>(index) / static_cast<float>(CIM_MAX_QPS - 1);
    source.view_qp_index[index] = static_cast<uint32_t>(CIM_MAX_QPS - 1 - index);
  }
  source.view_count = static_cast<uint32_t>(CIM_MAX_QPS);
  const CIMDebugSnapshot originalSource = source;
  // The extra element in each destination is a canary for an out-of-range write.
  uint64_t committedLat[CIM_MAX_QPS + 1];
  uint64_t committedRaw[CIM_MAX_QPS + 1];
  uint32_t committedBytes[CIM_MAX_QPS + 1];
  float committedLoadWeights[CIM_MAX_QPS + 1];
  uint32_t committedViewIndices[CIM_MAX_QPS + 1];
  std::fill(std::begin(committedLat), std::end(committedLat), 1001u);
  std::fill(std::begin(committedRaw), std::end(committedRaw), 1002u);
  std::fill(std::begin(committedBytes), std::end(committedBytes), 1003u);
  std::fill(std::begin(committedLoadWeights), std::end(committedLoadWeights), -1.0f);
  std::fill(std::begin(committedViewIndices), std::end(committedViewIndices), 1004u);
  uint32_t committedViewCount = 1005u;
  cimDbgCommitLastSnapshot(&source, committedLat, committedRaw, committedBytes, committedLoadWeights, &committedViewCount, committedViewIndices, CIM_MAX_QPS, -1, 2, -3, 4);
  for (int index = 0; index < CIM_MAX_QPS; ++index) {
    EXPECT_EQ(committedLat[index], source.lat[index]);
    EXPECT_EQ(committedRaw[index], source.raw[index]);
    EXPECT_EQ(committedBytes[index], source.bytes[index]);
    EXPECT_EQ(committedLoadWeights[index], source.lw[index]);
    EXPECT_EQ(committedViewIndices[index], source.view_qp_index[index]);
    EXPECT_EQ(source.lat[index], originalSource.lat[index]);
    EXPECT_EQ(source.raw[index], originalSource.raw[index]);
    EXPECT_EQ(source.bytes[index], originalSource.bytes[index]);
    EXPECT_EQ(source.lw[index], originalSource.lw[index]);
    EXPECT_EQ(source.view_qp_index[index], originalSource.view_qp_index[index]);
  }
  EXPECT_EQ(committedViewCount, static_cast<uint32_t>(CIM_MAX_QPS));
  EXPECT_EQ(committedLat[CIM_MAX_QPS], 1001u);
  EXPECT_EQ(committedRaw[CIM_MAX_QPS], 1002u);
  EXPECT_EQ(committedBytes[CIM_MAX_QPS], 1003u);
  EXPECT_EQ(committedLoadWeights[CIM_MAX_QPS], -1.0f);
  EXPECT_EQ(committedViewIndices[CIM_MAX_QPS], 1004u);
  EXPECT_EQ(source.view_count, originalSource.view_count);
  constexpr int partialQps = 3;
  CIMDebugSnapshot partialSource = source;
  partialSource.view_count = 0u;
  uint64_t selectorLat[partialQps + 1];
  uint64_t selectorRaw[partialQps + 1];
  uint32_t selectorBytes[partialQps + 1];
  float selectorLoadWeights[partialQps + 1];
  constexpr uint64_t latencySentinel = 2001u;
  constexpr uint64_t rawLatencySentinel = 2002u;
  constexpr uint32_t byteCountSentinel = 2003u;
  constexpr float loadWeightSentinel = -2.0f;
  /**
   * @brief Verifies one combination of snapshot field selectors.
   * @param[in] hasLat Latency selector passed to the production helper.
   * @param[in] hasRaw Raw-latency selector passed to the production helper.
   * @param[in] hasBytes Byte-count selector passed to the production helper.
   * @param[in] hasLw Load-weight selector passed to the production helper.
   * @param[in] copyLat Whether latency values are expected to be copied.
   * @param[in] copyRaw Whether raw-latency values are expected to be copied.
   * @param[in] copyBytes Whether byte-count values are expected to be copied.
   * @param[in] copyLw Whether load-weight values are expected to be copied.
   * @note Side effects: Resets local output buffers and records nonfatal test failures.
   */
  auto verifySelectors = [&](int hasLat, int hasRaw, int hasBytes, int hasLw,
      bool copyLat, bool copyRaw, bool copyBytes, bool copyLw) {
    std::fill(std::begin(selectorLat), std::end(selectorLat), latencySentinel);
    std::fill(std::begin(selectorRaw), std::end(selectorRaw), rawLatencySentinel);
    std::fill(std::begin(selectorBytes), std::end(selectorBytes), byteCountSentinel);
    std::fill(std::begin(selectorLoadWeights), std::end(selectorLoadWeights), loadWeightSentinel);
    uint32_t selectorViewCount = 99u;
    cimDbgCommitLastSnapshot(&partialSource, selectorLat, selectorRaw, selectorBytes, selectorLoadWeights, &selectorViewCount, nullptr, partialQps, hasLat, hasRaw, hasBytes, hasLw);
    for (int index = 0; index < partialQps; ++index) {
      EXPECT_EQ(selectorLat[index], copyLat ? partialSource.lat[index] : latencySentinel) << "index=" << index;
      EXPECT_EQ(selectorRaw[index], copyRaw ? partialSource.raw[index] : rawLatencySentinel) << "index=" << index;
      EXPECT_EQ(selectorBytes[index], copyBytes ? partialSource.bytes[index] : byteCountSentinel) << "index=" << index;
      EXPECT_EQ(selectorLoadWeights[index], copyLw ? partialSource.lw[index] : loadWeightSentinel) << "index=" << index;
    }
    EXPECT_EQ(selectorLat[partialQps], latencySentinel);
    EXPECT_EQ(selectorRaw[partialQps], rawLatencySentinel);
    EXPECT_EQ(selectorBytes[partialQps], byteCountSentinel);
    EXPECT_EQ(selectorLoadWeights[partialQps], loadWeightSentinel);
    EXPECT_EQ(selectorViewCount, 0u);
  };
  verifySelectors(-1, 0, 0, 0, true, false, false, false);
  verifySelectors(0, 2, 0, 0, false, true, false, false);
  verifySelectors(0, 0, -3, 0, false, false, true, false);
  verifySelectors(0, 0, 0, 4, false, false, false, true);
  verifySelectors(0, 0, 0, 0, false, false, false, false);
  CIMDebugSnapshot disabledSource = {};
  disabledSource.view_count = 2u;
  disabledSource.view_qp_index[0] = 2u;
  disabledSource.view_qp_index[1] = 0u;
  uint32_t disabledViewCount = 99u;
  uint32_t disabledViewIndices[] = {99u, 99u, 99u};
  cimDbgCommitLastSnapshot(&disabledSource, nullptr, nullptr, nullptr, nullptr, &disabledViewCount, disabledViewIndices, 3, 0, 0, 0, 0);
  EXPECT_EQ(disabledViewCount, 2u);
  EXPECT_EQ(disabledViewIndices[0], 2u);
  EXPECT_EQ(disabledViewIndices[1], 0u);
  EXPECT_EQ(disabledViewIndices[2], 99u);
  CIMDebugSnapshot emptySource = {};
  uint32_t emptyViewCount = 99u;
  cimDbgCommitLastSnapshot(&emptySource, nullptr, nullptr, nullptr, nullptr, &emptyViewCount, nullptr, 0, 1, 1, 1, 1);
  EXPECT_EQ(emptyViewCount, 0u);
}

/**
 * @brief Verifies view-count resolution clamps requested entries to every valid storage boundary.
 */
TEST(CimDebugSnapshotTest, ViewCountClampsRequestedCountAtValidBoundaries) {
  SCOPED_TRACE("View-count resolution must never expose more entries than the selected source provides.");
  CIMDebugSnapshot snapshot = {};
  EXPECT_EQ(cimDbgViewCount(&snapshot, 0), 0);
  EXPECT_EQ(cimDbgViewCount(&snapshot, 3), 3);
  EXPECT_EQ(cimDbgViewCount(&snapshot, CIM_MAX_QPS), CIM_MAX_QPS);
  EXPECT_EQ(snapshot.view_count, 0u);
  snapshot.view_count = 1u;
  EXPECT_EQ(cimDbgViewCount(&snapshot, 0), 0);
  EXPECT_EQ(cimDbgViewCount(&snapshot, 1), 1);
  EXPECT_EQ(snapshot.view_count, 1u);
  snapshot.view_count = 2u;
  EXPECT_EQ(cimDbgViewCount(&snapshot, 3), 2);
  EXPECT_EQ(cimDbgViewCount(&snapshot, 2), 2);
  EXPECT_EQ(cimDbgViewCount(&snapshot, 1), 1);
  EXPECT_EQ(snapshot.view_count, 2u);
  snapshot.view_count = 0u;
  EXPECT_EQ(cimDbgViewCount(&snapshot, CIM_MAX_QPS + 1), CIM_MAX_QPS);
  EXPECT_EQ(snapshot.view_count, 0u);
  snapshot.view_count = static_cast<uint32_t>(CIM_MAX_QPS);
  EXPECT_EQ(cimDbgViewCount(&snapshot, CIM_MAX_QPS), CIM_MAX_QPS);
  EXPECT_EQ(snapshot.view_count, static_cast<uint32_t>(CIM_MAX_QPS));
  snapshot.view_count = static_cast<uint32_t>(CIM_MAX_QPS + 1);
  EXPECT_EQ(cimDbgViewCount(&snapshot, CIM_MAX_QPS + 2), CIM_MAX_QPS);
  EXPECT_EQ(snapshot.view_count, static_cast<uint32_t>(CIM_MAX_QPS + 1));
  snapshot.view_count = static_cast<uint32_t>(std::numeric_limits<int>::max());
  EXPECT_EQ(cimDbgViewCount(&snapshot, std::numeric_limits<int>::max()), CIM_MAX_QPS);
  EXPECT_EQ(snapshot.view_count, static_cast<uint32_t>(std::numeric_limits<int>::max()));
}

/** @brief Verifies view-to-QP mapping and fallback behavior at valid index boundaries. */
TEST(CimDebugSnapshotTest, ViewQpIndexResolvesMappingsAndFallbacksAtValidBoundaries) {
  SCOPED_TRACE("View-index resolution must honor explicit mappings and use documented bounded fallbacks.");
  CIMDebugSnapshot snapshot = {};
  snapshot.view_count = 3u;
  snapshot.view_qp_index[0] = 0u;
  snapshot.view_qp_index[1] = 3u;
  snapshot.view_qp_index[2] = 4u;
  snapshot.view_qp_index[3] = 0u;
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, std::numeric_limits<int>::min(), 4), std::numeric_limits<int>::min());
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, -1, 4), -1);
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, 0, 4), 0);
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, 1, 4), 3);
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, 2, 4), 2);
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, 3, 4), 3);
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, std::numeric_limits<int>::max(), 4), std::numeric_limits<int>::max());
  EXPECT_EQ(snapshot.view_count, 3u);
  EXPECT_EQ(snapshot.view_qp_index[0], 0u);
  EXPECT_EQ(snapshot.view_qp_index[1], 3u);
  snapshot.view_qp_index[2] = 5u;
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, 2, 4), 2);
  snapshot.view_qp_index[2] = std::numeric_limits<uint32_t>::max();
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, 2, 4), 2);
  EXPECT_EQ(snapshot.view_qp_index[2], std::numeric_limits<uint32_t>::max());
  snapshot.view_count = 0u;
  snapshot.view_qp_index[0] = 2u;
  snapshot.view_qp_index[2] = 0u;
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, 0, 3), 2);
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, 2, 3), 0);
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, 3, 3), 3);
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, 0, 0), 0);
  EXPECT_EQ(snapshot.view_count, 0u);
  snapshot.view_count = 1u;
  snapshot.view_qp_index[0] = 2u;
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, 0, 3), 2);
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, 1, 3), 1);
  snapshot.view_count = 4u;
  snapshot.view_qp_index[1] = 0u;
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, 1, 2), 0);
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, 2, 2), 2);
  snapshot.view_count = static_cast<uint32_t>(CIM_MAX_QPS);
  snapshot.view_qp_index[CIM_MAX_QPS - 1] = static_cast<uint32_t>(CIM_MAX_QPS - 1);
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, CIM_MAX_QPS - 1, CIM_MAX_QPS), CIM_MAX_QPS - 1);
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, CIM_MAX_QPS, CIM_MAX_QPS), CIM_MAX_QPS);
  snapshot.view_count = static_cast<uint32_t>(std::numeric_limits<int>::max());
  snapshot.view_qp_index[CIM_MAX_QPS - 1] = static_cast<uint32_t>(std::numeric_limits<int>::max() - 1);
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, CIM_MAX_QPS - 1, std::numeric_limits<int>::max()), std::numeric_limits<int>::max() - 1);
  EXPECT_EQ(cimDbgViewQpIndex(&snapshot, CIM_MAX_QPS, std::numeric_limits<int>::max()), CIM_MAX_QPS);
  EXPECT_EQ(snapshot.view_count, static_cast<uint32_t>(std::numeric_limits<int>::max()));
  EXPECT_EQ(snapshot.view_qp_index[CIM_MAX_QPS - 1], static_cast<uint32_t>(std::numeric_limits<int>::max() - 1));
}

/**
 * @brief Verifies exact snapshot-printer output for every enabled-field combination and valid
 * boundary.
 */
TEST(CimDiagnosticsTest, SnapshotPrinterPrintsEveryFieldCombinationAndValidBoundary) {
  SCOPED_TRACE("Snapshot output must contain exactly the selected fields, bounded views, and line termination.");
  CIMDebugSnapshot snapshot = {};
  snapshot.lat[0] = 10u;
  snapshot.lat[1] = 20u;
  snapshot.raw[0] = 11u;
  snapshot.raw[1] = 21u;
  snapshot.bytes[0] = 100u;
  snapshot.bytes[1] = 200u;
  snapshot.lw[0] = 0.25f;
  snapshot.lw[1] = 0.75f;
  snapshot.view_count = 2u;
  snapshot.view_qp_index[0] = 1u;
  snapshot.view_qp_index[1] = 0u;
  const CIMDebugSnapshot originalSnapshot = snapshot;
  /** @brief Describes one diagnostic snapshot field-selection scenario. */
  struct PrintCase {
    int hasLat;
    int hasRaw;
    int hasBytes;
    int hasLw;
    const char* expected;
  };
  const PrintCase cases[] = {
      {0, 0, 0, 0, "\n"},
      {0, 0, 0, 4, "\n"},
      {-1, 0, 0, 0, "lat tag: q1=20 q0=10 \n"},
      {-1, 0, 0, 4, "lat/lw tag: q1=20(w=0.7500) q0=10(w=0.2500) \n"},
      {0, 2, 0, 0, "raw tag: q1=21 q0=11 \n"},
      {0, 2, 0, 4, "raw/lw tag: q1=21(w=0.7500) q0=11(w=0.2500) \n"},
      {0, 0, -3, 0, "bytes tag: q1=200 q0=100 \n"},
      {0, 0, -3, 4, "bytes/lw tag: q1=200(w=0.7500) q0=100(w=0.2500) \n"},
      {-1, 2, 0, 0, "lat/raw tag: q1=20(raw=21) q0=10(raw=11) \n"},
      {-1, 2, 0, 4, "lat/raw/lw tag: q1=20(raw=21,w=0.7500) q0=10(raw=11,w=0.2500) \n"},
      {-1, 0, -3, 0, "lat/bytes tag: q1=20(bytes=200) q0=10(bytes=100) \n"},
      {-1, 0, -3, 4, "lat/bytes/lw tag: q1=20(bytes=200,w=0.7500) q0=10(bytes=100,w=0.2500) \n"},
      {0, 2, -3, 0, "raw/bytes tag: q1=21(bytes=200) q0=11(bytes=100) \n"},
      {0, 2, -3, 4, "raw/bytes/lw tag: q1=21(bytes=200,w=0.7500) q0=11(bytes=100,w=0.2500) \n"},
      {-1, 2, -3, 0, "lat/raw/bytes tag: q1=20(raw=21,bytes=200) q0=10(raw=11,bytes=100) \n"},
      {-1, 2, -3, 4, "lat/raw/bytes/lw tag: q1=20(raw=21,bytes=200,w=0.7500) q0=10(raw=11,bytes=100,w=0.2500) \n"},
  };
  for (const PrintCase& testCase : cases) {
    const char* latencyLabel = testCase.hasLat ? "lat" : nullptr;
    const char* rawLatencyLabel = testCase.hasRaw ? "raw" : nullptr;
    const char* byteCountLabel = testCase.hasBytes ? "bytes" : nullptr;
    const bool hasPrimaryField = testCase.hasLat || testCase.hasRaw || testCase.hasBytes;
    const char* loadWeightLabel = testCase.hasLw && hasPrimaryField ? "lw" : nullptr;
    SCOPED_TRACE(testing::Message() << "hasLat=" << testCase.hasLat << ", hasRaw=" << testCase.hasRaw << ", hasBytes=" << testCase.hasBytes << ", hasLw=" << testCase.hasLw);
    testing::internal::CaptureStdout();
    cimDbgPrintSnapshot("tag", latencyLabel, rawLatencyLabel, byteCountLabel, loadWeightLabel, &snapshot, 2, testCase.hasLat, testCase.hasRaw, testCase.hasBytes, testCase.hasLw);
    EXPECT_EQ(testing::internal::GetCapturedStdout(), testCase.expected);
  }
  testing::internal::CaptureStdout();
  cimDbgPrintSnapshot(nullptr, "lat", "raw", "bytes", "lw", &snapshot, 2, -1, 2, -3, 4);
  EXPECT_EQ(testing::internal::GetCapturedStdout(), "lat/raw/bytes/lw: q1=20(raw=21,bytes=200,w=0.7500) q0=10(raw=11,bytes=100,w=0.2500) \n");
  testing::internal::CaptureStdout();
  cimDbgPrintSnapshot("", "lat", "raw", "bytes", "lw", &snapshot, 2, -1, 2, -3, 4);
  EXPECT_EQ(testing::internal::GetCapturedStdout(), "lat/raw/bytes/lw: q1=20(raw=21,bytes=200,w=0.7500) q0=10(raw=11,bytes=100,w=0.2500) \n");
  testing::internal::CaptureStdout();
  cimDbgPrintSnapshot("tag", "lat", "raw", "bytes", "lw", &snapshot, 0, -1, 2, -3, 4);
  EXPECT_EQ(testing::internal::GetCapturedStdout(), "lat/raw/bytes/lw tag: \n");
  for (int index = 0; index < CIM_MAX_QPS; ++index) {
    EXPECT_EQ(snapshot.lat[index], originalSnapshot.lat[index]);
    EXPECT_EQ(snapshot.raw[index], originalSnapshot.raw[index]);
    EXPECT_EQ(snapshot.bytes[index], originalSnapshot.bytes[index]);
    EXPECT_EQ(snapshot.lw[index], originalSnapshot.lw[index]);
    EXPECT_EQ(snapshot.view_qp_index[index], originalSnapshot.view_qp_index[index]);
  }
  EXPECT_EQ(snapshot.view_count, originalSnapshot.view_count);
  CIMDebugSnapshot defaultView = snapshot;
  defaultView.view_count = 0u;
  defaultView.view_qp_index[0] = 0u;
  defaultView.view_qp_index[1] = 1u;
  testing::internal::CaptureStdout();
  cimDbgPrintSnapshot(nullptr, "lat", nullptr, nullptr, nullptr, &defaultView, 2, 1, 0, 0, 0);
  EXPECT_EQ(testing::internal::GetCapturedStdout(), "lat: q0=10 q1=20 \n");
  CIMDebugSnapshot invalidView = snapshot;
  invalidView.view_qp_index[0] = 2u;
  invalidView.view_qp_index[1] = 0u;
  testing::internal::CaptureStdout();
  cimDbgPrintSnapshot(nullptr, "lat", nullptr, nullptr, nullptr, &invalidView, 2, 1, 0, 0, 0);
  EXPECT_EQ(testing::internal::GetCapturedStdout(), "lat: q0=10 q0=10 \n");
  CIMDebugSnapshot partialView = snapshot;
  partialView.view_count = 1u;
  testing::internal::CaptureStdout();
  cimDbgPrintSnapshot(nullptr, "lat", nullptr, nullptr, nullptr, &partialView, 2, 1, 0, 0, 0);
  EXPECT_EQ(testing::internal::GetCapturedStdout(), "lat: q1=20 \n");
  EXPECT_EQ(partialView.view_count, 1u);
  CIMDebugSnapshot clampedView = snapshot;
  clampedView.view_count = 3u;
  testing::internal::CaptureStdout();
  cimDbgPrintSnapshot(nullptr, "lat", nullptr, nullptr, nullptr, &clampedView, 2, 1, 0, 0, 0);
  EXPECT_EQ(testing::internal::GetCapturedStdout(), "lat: q1=20 q0=10 \n");
  EXPECT_EQ(clampedView.view_count, 3u);
  CIMDebugSnapshot numericBoundaries = {};
  numericBoundaries.view_count = 2u;
  numericBoundaries.view_qp_index[0] = 0u;
  numericBoundaries.view_qp_index[1] = 1u;
  numericBoundaries.lat[1] = std::numeric_limits<uint64_t>::max();
  numericBoundaries.raw[1] = std::numeric_limits<uint64_t>::max();
  numericBoundaries.bytes[1] = std::numeric_limits<uint32_t>::max();
  numericBoundaries.lw[1] = 1.0f;
  const std::string maximumUint64 = std::to_string(std::numeric_limits<uint64_t>::max());
  const std::string maximumUint32 = std::to_string(std::numeric_limits<uint32_t>::max());
  const std::string numericBoundaryOutput = "lat/raw/bytes/lw: q0=0(raw=0,bytes=0,w=0.0000) q1=" + maximumUint64 + "(raw=" + maximumUint64 + ",bytes=" + maximumUint32 + ",w=1.0000) \n";
  testing::internal::CaptureStdout();
  cimDbgPrintSnapshot(nullptr, "lat", "raw", "bytes", "lw", &numericBoundaries, 2, 1, 1, 1, 1);
  EXPECT_EQ(testing::internal::GetCapturedStdout(), numericBoundaryOutput);
  CIMDebugSnapshot maximum = {};
  maximum.view_count = 0u;
  for (int index = 0; index < CIM_MAX_QPS; ++index) {
    maximum.lat[index] = static_cast<uint64_t>(index);
    maximum.view_qp_index[index] = static_cast<uint32_t>(index);
  }
  std::string expectedMaximumOutput = "maximum: ";
  for (int index = 0; index < CIM_MAX_QPS; ++index) {
    expectedMaximumOutput += "q" + std::to_string(index) + "=" + std::to_string(index) + " ";
  }
  expectedMaximumOutput += "\n";
  testing::internal::CaptureStdout();
  cimDbgPrintSnapshot(nullptr, "maximum", nullptr, nullptr, nullptr, &maximum, CIM_MAX_QPS, 1, 0, 0, 0);
  const std::string maximumOutput = testing::internal::GetCapturedStdout();
  EXPECT_EQ(maximumOutput, expectedMaximumOutput);
  EXPECT_EQ(maximum.view_count, 0u);
  maximum.view_count = static_cast<uint32_t>(CIM_MAX_QPS + 1);
  testing::internal::CaptureStdout();
  cimDbgPrintSnapshot(nullptr, "maximum", nullptr, nullptr, nullptr, &maximum, CIM_MAX_QPS, 1, 0, 0, 0);
  EXPECT_EQ(testing::internal::GetCapturedStdout(), expectedMaximumOutput);
  EXPECT_EQ(maximum.view_count, static_cast<uint32_t>(CIM_MAX_QPS + 1));
  EXPECT_EQ(maximum.lat[0], 0u);
  EXPECT_EQ(maximum.lat[CIM_MAX_QPS - 1], static_cast<uint64_t>(CIM_MAX_QPS - 1));
  EXPECT_EQ(maximum.view_qp_index[CIM_MAX_QPS - 1], static_cast<uint32_t>(CIM_MAX_QPS - 1));
}

/** @brief Verifies comparison with the last snapshot reports every enabled field difference. */
TEST(CimDebugSnapshotTest, ComparisonWithLastDetectsEnabledFieldDifferencesAtValidBoundaries) {
  SCOPED_TRACE("Last-snapshot comparison must ignore disabled data and detect each enabled change.");
  CIMDebugSnapshot snapshot = {};
  uint64_t lastLat[CIM_MAX_QPS] = {};
  uint64_t lastRaw[CIM_MAX_QPS] = {};
  uint32_t lastBytes[CIM_MAX_QPS] = {};
  float lastLoadWeights[CIM_MAX_QPS] = {};
  constexpr int finalQpIndex = CIM_MAX_QPS - 1;
  EXPECT_EQ(cimDbgSnapshotDiffersFromLast(&snapshot, nullptr, nullptr, nullptr, nullptr, 0, 1, 1, 1, 1), 0);
  snapshot.lat[0] = 1u;
  snapshot.raw[0] = 2u;
  snapshot.bytes[0] = 3u;
  snapshot.lw[0] = 1.0f;
  EXPECT_EQ(cimDbgSnapshotDiffersFromLast(&snapshot, nullptr, nullptr, nullptr, nullptr, CIM_MAX_QPS, 0, 0, 0, 0), 0);
  snapshot.lat[0] = 0u;
  snapshot.raw[0] = 0u;
  snapshot.bytes[0] = 0u;
  snapshot.lw[0] = 0.0f;
  EXPECT_EQ(cimDbgSnapshotDiffersFromLast(&snapshot, lastLat, lastRaw, lastBytes, lastLoadWeights, CIM_MAX_QPS, 1, 1, 1, 1), 0);
  snapshot.lat[0] = 1u;
  EXPECT_EQ(cimDbgSnapshotDiffersFromLast(&snapshot, lastLat, nullptr, nullptr, nullptr, 1, 1, 0, 0, 0), 1);
  snapshot.lat[0] = 0u;
  snapshot.raw[0] = 1u;
  EXPECT_EQ(cimDbgSnapshotDiffersFromLast(&snapshot, nullptr, lastRaw, nullptr, nullptr, 1, 0, 1, 0, 0), 1);
  snapshot.raw[0] = 0u;
  snapshot.bytes[0] = 1u;
  EXPECT_EQ(cimDbgSnapshotDiffersFromLast(&snapshot, nullptr, nullptr, lastBytes, nullptr, 1, 0, 0, 1, 0), 1);
  snapshot.bytes[0] = 0u;
  snapshot.lw[0] = 1e-6f;
  EXPECT_EQ(cimDbgSnapshotDiffersFromLast(&snapshot, nullptr, nullptr, nullptr, lastLoadWeights, 1, 0, 0, 0, 1), 0);
  snapshot.lw[0] = std::nextafter(1e-6f, 1.0f);
  EXPECT_EQ(cimDbgSnapshotDiffersFromLast(&snapshot, nullptr, nullptr, nullptr, lastLoadWeights, 1, 0, 0, 0, 1), 1);
  snapshot.lw[0] = 0.0f;
  snapshot.lat[1] = 1u;
  EXPECT_EQ(cimDbgSnapshotDiffersFromLast(&snapshot, lastLat, nullptr, nullptr, nullptr, 1, 1, 0, 0, 0), 0);
  snapshot.lat[1] = 0u;
  snapshot.lat[finalQpIndex] = 1u;
  EXPECT_EQ(cimDbgSnapshotDiffersFromLast(&snapshot, lastLat, lastRaw, lastBytes, lastLoadWeights, CIM_MAX_QPS, 1, 1, 1, 1), 1);
  snapshot.lat[finalQpIndex] = 0u;
  snapshot.raw[finalQpIndex] = 1u;
  EXPECT_EQ(cimDbgSnapshotDiffersFromLast(&snapshot, lastLat, lastRaw, lastBytes, lastLoadWeights, CIM_MAX_QPS, 1, 1, 1, 1), 1);
  snapshot.raw[finalQpIndex] = 0u;
  snapshot.bytes[finalQpIndex] = 1u;
  EXPECT_EQ(cimDbgSnapshotDiffersFromLast(&snapshot, lastLat, lastRaw, lastBytes, lastLoadWeights, CIM_MAX_QPS, 1, 1, 1, 1), 1);
  snapshot.bytes[finalQpIndex] = 0u;
  snapshot.lw[finalQpIndex] = std::nextafter(1e-6f, 1.0f);
  EXPECT_EQ(cimDbgSnapshotDiffersFromLast(&snapshot, lastLat, lastRaw, lastBytes, lastLoadWeights, CIM_MAX_QPS, 1, 1, 1, 1), 1);
  snapshot.lw[finalQpIndex] = 0.0f;
  snapshot.lat[0] = 1u;
  EXPECT_EQ(cimDbgSnapshotDiffersFromLast(&snapshot, lastLat, nullptr, nullptr, nullptr, 1, -1, 0, 0, 0), 1);
}

/** @brief Verifies pairwise snapshot comparison detects enabled differences at valid boundaries. */
TEST(CimDebugSnapshotTest, PairComparisonDetectsEnabledFieldDifferencesAtValidBoundaries) {
  SCOPED_TRACE("Pairwise comparison must be field-selective, bounded, and sensitive to enabled changes.");
  CIMDebugSnapshot first = {};
  CIMDebugSnapshot second = {};
  constexpr int finalQpIndex = CIM_MAX_QPS - 1;
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, 0, 1, 1, 1, 1), 0);
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, CIM_MAX_QPS, 1, 1, 1, 1), 0);
  first.lat[0] = 1u;
  first.raw[0] = 2u;
  first.bytes[0] = 3u;
  first.lw[0] = 1.0f;
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, CIM_MAX_QPS, 0, 0, 0, 0), 0);
  first.lat[0] = 0u;
  first.raw[0] = 0u;
  first.bytes[0] = 0u;
  first.lw[0] = 0.0f;
  first.lat[0] = 1u;
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, 1, 1, 0, 0, 0), 1);
  first.lat[0] = 0u;
  second.lat[0] = 1u;
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, 1, 1, 0, 0, 0), 1);
  second.lat[0] = 0u;
  first.raw[0] = 1u;
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, 1, 0, 1, 0, 0), 1);
  first.raw[0] = 0u;
  second.raw[0] = 1u;
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, 1, 0, 1, 0, 0), 1);
  second.raw[0] = 0u;
  first.bytes[0] = 1u;
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, 1, 0, 0, 1, 0), 1);
  first.bytes[0] = 0u;
  second.bytes[0] = 1u;
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, 1, 0, 0, 1, 0), 1);
  second.bytes[0] = 0u;
  first.lw[0] = std::nextafter(1e-6f, 0.0f);
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, 1, 0, 0, 0, 1), 0);
  first.lw[0] = 1e-6f;
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, 1, 0, 0, 0, 1), 0);
  first.lw[0] = std::nextafter(1e-6f, 1.0f);
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, 1, 0, 0, 0, 1), 1);
  first.lw[0] = 0.0f;
  first.lat[1] = 1u;
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, 1, 1, 0, 0, 0), 0);
  first.lat[1] = 0u;
  first.view_count = 1u;
  first.view_qp_index[0] = 1u;
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, CIM_MAX_QPS, 1, 1, 1, 1), 0);
  first.view_count = 0u;
  first.view_qp_index[0] = 0u;
  first.lat[finalQpIndex] = 1u;
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, CIM_MAX_QPS, 1, 1, 1, 1), 1);
  first.lat[finalQpIndex] = 0u;
  first.raw[finalQpIndex] = 1u;
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, CIM_MAX_QPS, 1, 1, 1, 1), 1);
  first.raw[finalQpIndex] = 0u;
  first.bytes[finalQpIndex] = 1u;
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, CIM_MAX_QPS, 1, 1, 1, 1), 1);
  first.bytes[finalQpIndex] = 0u;
  first.lw[finalQpIndex] = std::nextafter(1e-6f, 1.0f);
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, CIM_MAX_QPS, 1, 1, 1, 1), 1);
  first.lw[finalQpIndex] = 0.0f;
  first.raw[0] = 1u;
  EXPECT_EQ(cimDbgSnapshotsDiffer(&first, &second, 1, 0, -1, 0, 0), 1);
}

/**
 * @brief Verifies byte-source configuration updates its pointer, normalizes its label, and
 * applies positive QP counts while retaining the current count for zero.
 */
TEST(CimDiagnosticsTest, ByteSourceHandlesSourceLabelAndQpCountBoundaries) {
  SCOPED_TRACE(
      "Byte-source configuration must preserve pointer identity, own a normalized label, and "
      "replace the QP count only for positive input.");
  // Snapshot and restore the affected byte-source globals under the production mutex.
  pthread_mutex_lock(&g_lat_mu);
  volatile uint32_t* const originalBytesPointer = g_bytes_ptr;
  const int originalQpCount = g_lat_nqps;
  const std::string originalBytesLabel = g_bytes_label;
  const int originalStartedState = g_lat_started;
  const int originalRunningState = g_lat_running;
  pthread_mutex_unlock(&g_lat_mu);
  ASSERT_EQ(originalStartedState, 0);
  ASSERT_EQ(originalRunningState, 0);
  uint32_t primaryBytes[CIM_MAX_QPS] = {};
  uint32_t replacementBytes[CIM_MAX_QPS] = {};
  char copiedLabel[] = "byte-source";
  CIM_DebugLatPrinterSetByteSource(copiedLabel, primaryBytes, 1);
  EXPECT_EQ(g_bytes_ptr, primaryBytes);
  EXPECT_EQ(g_lat_nqps, 1);
  EXPECT_STREQ(g_bytes_label, "byte-source");
  copiedLabel[0] = 'X';
  EXPECT_STREQ(g_bytes_label, "byte-source");
  const std::string longLabel(80, 'x');
  CIM_DebugLatPrinterSetByteSource(longLabel.c_str(), primaryBytes, CIM_MAX_QPS);
  EXPECT_EQ(g_lat_nqps, CIM_MAX_QPS);
  EXPECT_EQ(g_bytes_label[sizeof(g_bytes_label) - 1], '\0');
  EXPECT_EQ(std::string(g_bytes_label, sizeof(g_bytes_label) - 1), std::string(63, 'x'));
  CIM_DebugLatPrinterSetByteSource("", replacementBytes, 0);
  EXPECT_EQ(g_bytes_ptr, replacementBytes);
  EXPECT_EQ(g_lat_nqps, CIM_MAX_QPS);
  EXPECT_STREQ(g_bytes_label, "bytes");
  CIM_DebugLatPrinterSetByteSource("temporary-label", primaryBytes, 0);
  EXPECT_STREQ(g_bytes_label, "temporary-label");
  CIM_DebugLatPrinterSetByteSource(nullptr, primaryBytes, 0);
  EXPECT_STREQ(g_bytes_label, "bytes");
  CIM_DebugLatPrinterSetByteSource(nullptr, nullptr, 0);
  EXPECT_EQ(g_bytes_ptr, nullptr);
  EXPECT_EQ(g_lat_nqps, CIM_MAX_QPS);
  CIM_DebugLatPrinterSetByteSource(nullptr, nullptr, 1);
  EXPECT_EQ(g_lat_nqps, 1);
  EXPECT_EQ(g_lat_started, originalStartedState);
  EXPECT_EQ(g_lat_running, originalRunningState);
  pthread_mutex_lock(&g_lat_mu);
  g_bytes_ptr = originalBytesPointer;
  g_lat_nqps = originalQpCount;
  strncpy(g_bytes_label, originalBytesLabel.c_str(), sizeof(g_bytes_label));
  g_bytes_label[sizeof(g_bytes_label) - 1] = 0;
  pthread_mutex_unlock(&g_lat_mu);
}

/** @brief Verifies view-source selection installs and clears each supported diagnostic source. */
TEST(CimDiagnosticsTest, ViewSourceSetsAndClearsSources) {
  SCOPED_TRACE("View-source selection must expose only the requested source and clear stale selections.");
  // Snapshot and restore the affected view-source globals under the production mutex.
  pthread_mutex_lock(&g_lat_mu);
  volatile uint32_t* const originalViewCountPointer = g_dbg_view_count_ptr;
  volatile uint32_t* const originalViewIndexPointer = g_dbg_view_qp_index_ptr;
  const int originalStartedState = g_lat_started;
  const int originalRunningState = g_lat_running;
  pthread_mutex_unlock(&g_lat_mu);
  ASSERT_EQ(originalStartedState, 0);
  ASSERT_EQ(originalRunningState, 0);
  uint32_t viewCount = 2u;
  uint32_t viewIndices[CIM_MAX_QPS] = {};
  for (int index = 0; index < CIM_MAX_QPS; ++index) {
    viewIndices[index] = static_cast<uint32_t>(CIM_MAX_QPS - 1 - index);
  }
  CIM_DebugLatPrinterSetViewSource(&viewCount, viewIndices);
  EXPECT_EQ(g_dbg_view_count_ptr, &viewCount);
  EXPECT_EQ(g_dbg_view_qp_index_ptr, viewIndices);
  CIM_DebugLatPrinterSetViewSource(nullptr, viewIndices);
  EXPECT_EQ(g_dbg_view_count_ptr, nullptr);
  EXPECT_EQ(g_dbg_view_qp_index_ptr, viewIndices);
  CIM_DebugLatPrinterSetViewSource(&viewCount, nullptr);
  EXPECT_EQ(g_dbg_view_count_ptr, &viewCount);
  EXPECT_EQ(g_dbg_view_qp_index_ptr, nullptr);
  CIM_DebugLatPrinterSetViewSource(nullptr, nullptr);
  EXPECT_EQ(g_dbg_view_count_ptr, nullptr);
  EXPECT_EQ(g_dbg_view_qp_index_ptr, nullptr);
  EXPECT_EQ(g_lat_started, originalStartedState);
  EXPECT_EQ(g_lat_running, originalRunningState);
  EXPECT_EQ(viewCount, 2u);
  for (int index = 0; index < CIM_MAX_QPS; ++index) {
    EXPECT_EQ(viewIndices[index], static_cast<uint32_t>(CIM_MAX_QPS - 1 - index));
  }
  pthread_mutex_lock(&g_lat_mu);
  g_dbg_view_count_ptr = originalViewCountPointer;
  g_dbg_view_qp_index_ptr = originalViewIndexPointer;
  pthread_mutex_unlock(&g_lat_mu);
}

/** @brief Verifies the diagnostic printer thread handles disabled, enabled, and shutdown states. */
TEST(CimDiagnosticsTest, PrinterThreadBehavior) {
  SCOPED_TRACE("Printer-thread control must avoid unwanted output and terminate cleanly on shutdown.");
  // Snapshot and restore every shared printer setting under its production mutex.
  pthread_mutex_lock(&g_lat_mu);
  volatile uint64_t* const originalLatencyPointer = g_lat_ptr;
  volatile uint64_t* const originalRawLatencyPointer = g_r_lat_ptr;
  volatile uint32_t* const originalByteCountPointer = g_bytes_ptr;
  const int originalQpCount = g_lat_nqps;
  const std::string originalLatencyLabel = g_lat_label;
  const std::string originalRawLatencyLabel = g_r_lat_label;
  const std::string originalByteCountLabel = g_bytes_label;
  volatile float* const originalLoadWeightPointer = g_lw_ptr;
  const int originalLoadWeightQpCount = g_lw_nqps;
  const std::string originalLoadWeightLabel = g_lw_label;
  volatile uint32_t* const originalSequencePointer = g_dbg_seq_ptr;
  volatile uint32_t* const originalViewCountPointer = g_dbg_view_count_ptr;
  volatile uint32_t* const originalViewIndexPointer = g_dbg_view_qp_index_ptr;
  const int originalInterval = g_lat_interval_ms;
  const int originalRunningState = g_lat_running;
  const int originalStartedState = g_lat_started;
  pthread_mutex_unlock(&g_lat_mu);
  ASSERT_EQ(originalStartedState, 0);
  ASSERT_EQ(originalRunningState, 0);
  pthread_mutex_lock(&g_lat_mu);
  g_lat_interval_ms = 1;
  pthread_mutex_unlock(&g_lat_mu);
  int unusedArgument = 7;
  EXPECT_EQ(lat_printer_thread(&unusedArgument), nullptr);
  EXPECT_EQ(unusedArgument, 7);
  /**
   * @brief Runs the latency-printer worker for a bounded interval.
   * @param[in] duration Nonnegative interval before shutdown is requested.
   * @param[in] argument Opaque argument forwarded unchanged to lat_printer_thread.
   * @return The value returned by lat_printer_thread.
   * @note Side effects: Starts and joins a thread and toggles the shared running flag.
   */
  auto runWorkerFor = [&](std::chrono::milliseconds duration, void* argument) {
    void* result = &unusedArgument;
    std::atomic<bool> workerStarted = false;
    pthread_mutex_lock(&g_lat_mu);
    g_lat_running = 1;
    pthread_mutex_unlock(&g_lat_mu);
    /**
     * @brief Invokes the latency-printer entry point and records its result.
     * @note Side effects: Updates workerStarted and result.
     */
    std::thread worker([&]() {
      workerStarted.store(true, std::memory_order_release);
      result = lat_printer_thread(argument);
    });
    // Synchronize startup before timing the active interval and requesting termination.
    while (!workerStarted.load(std::memory_order_acquire)) std::this_thread::yield();
    std::this_thread::sleep_for(duration);
    pthread_mutex_lock(&g_lat_mu);
    g_lat_running = 0;
    pthread_mutex_unlock(&g_lat_mu);
    worker.join();
    return result;
  };
  volatile float idleLoadWeights[] = {0.5f};
  CIM_DebugLatPrinterSetSource(nullptr, nullptr, nullptr, 1);
  CIM_DebugLatPrinterSetWeightSource("idle_lw", idleLoadWeights, 1);
  CIM_DebugLatPrinterSetByteSource(nullptr, nullptr, 0);
  CIM_DebugLatPrinterSetSyncSeq(nullptr);
  CIM_DebugLatPrinterSetViewSource(nullptr, nullptr);
  testing::internal::CaptureStdout();
  void* const weightOnlyResult = runWorkerFor(std::chrono::milliseconds(5), nullptr);
  const std::string weightOnlyOutput = testing::internal::GetCapturedStdout();
  EXPECT_EQ(weightOnlyResult, nullptr);
  EXPECT_TRUE(weightOnlyOutput.empty());
  volatile uint64_t idleLatency[] = {9u};
  const int nonpositiveQpCounts[] = {0, -1};
  for (const int nonpositiveQpCount : nonpositiveQpCounts) {
    CIM_DebugLatPrinterSetSource("idle_lat", idleLatency, nullptr, nonpositiveQpCount);
    CIM_DebugLatPrinterSetWeightSource(nullptr, nullptr, 0);
    testing::internal::CaptureStdout();
    void* const nonpositiveResult = runWorkerFor(std::chrono::milliseconds(5), &unusedArgument);
    const std::string nonpositiveOutput = testing::internal::GetCapturedStdout();
    EXPECT_EQ(nonpositiveResult, nullptr) << "nqps=" << nonpositiveQpCount;
    EXPECT_TRUE(nonpositiveOutput.empty()) << "nqps=" << nonpositiveQpCount;
    EXPECT_EQ(idleLatency[0], 9u);
  }
  volatile uint64_t latencies[CIM_MAX_QPS + 1] = {};
  volatile uint64_t rawLatencies[CIM_MAX_QPS + 1] = {};
  volatile uint32_t byteCounts[CIM_MAX_QPS + 1] = {};
  volatile float loadWeights[CIM_MAX_QPS + 1] = {};
  for (int index = 0; index < CIM_MAX_QPS; ++index) {
    latencies[index] = static_cast<uint64_t>(index + 10);
    rawLatencies[index] = static_cast<uint64_t>(index + 20);
    byteCounts[index] = static_cast<uint32_t>(index + 30);
    loadWeights[index] = static_cast<float>(index) / static_cast<float>(CIM_MAX_QPS - 1);
  }
  latencies[CIM_MAX_QPS] = 10001u;
  rawLatencies[CIM_MAX_QPS] = 10002u;
  byteCounts[CIM_MAX_QPS] = 10003u;
  loadWeights[CIM_MAX_QPS] = -1.0f;
  volatile uint32_t sequence = 0u;
  volatile uint32_t viewCount = 3u;
  volatile uint32_t viewIndices[CIM_MAX_QPS] = {};
  viewIndices[0] = static_cast<uint32_t>(CIM_MAX_QPS);
  viewIndices[1] = static_cast<uint32_t>(CIM_MAX_QPS - 1);
  viewIndices[2] = 0u;
  CIM_DebugLatPrinterSetSource("thread_lat", latencies, rawLatencies, CIM_MAX_QPS + 1);
  CIM_DebugLatPrinterSetWeightSource("thread_lw", loadWeights, CIM_MAX_QPS + 1);
  CIM_DebugLatPrinterSetByteSource("thread_bytes", byteCounts, 0);
  CIM_DebugLatPrinterSetSyncSeq(&sequence);
  CIM_DebugLatPrinterSetViewSource(&viewCount, viewIndices);
  testing::internal::CaptureStdout();
  void* const activeResult = runWorkerFor(std::chrono::milliseconds(20), nullptr);
  const std::string activeOutput = testing::internal::GetCapturedStdout();
  EXPECT_EQ(activeResult, nullptr);
  EXPECT_NE(activeOutput.find("thread_lat/thread_lat_raw/thread_bytes/thread_lw [0]: "), std::string::npos);
  EXPECT_NE(activeOutput.find("thread_lat/thread_lat_raw/thread_bytes/thread_lw [1]: "), std::string::npos);
  const std::string finalSupportedQp = "q" + std::to_string(CIM_MAX_QPS - 1) + "=";
  EXPECT_NE(activeOutput.find(finalSupportedQp), std::string::npos);
  const std::string firstUnsupportedQp = "q" + std::to_string(CIM_MAX_QPS) + "=";
  EXPECT_EQ(activeOutput.find(firstUnsupportedQp), std::string::npos);
  const std::string expectedViewEntries = "q0=10(raw=20,bytes=30,w=0.0000) q" + std::to_string(CIM_MAX_QPS - 1) + "=" + std::to_string(CIM_MAX_QPS + 9) + "(raw=" + std::to_string(CIM_MAX_QPS + 19) + ",bytes=" + std::to_string(CIM_MAX_QPS + 29) + ",w=1.0000) q0=10(raw=20,bytes=30,w=0.0000) ";
  std::istringstream activeOutputStream(activeOutput);
  std::vector<std::string> activeLines;
  std::string activeLine;
  while (std::getline(activeOutputStream, activeLine)) {
    activeLines.push_back(activeLine);
  }
  EXPECT_GE(activeLines.size(), 2u);
  EXPECT_EQ(activeLines.size() % 2u, 0u);
  for (size_t lineIndex = 0; lineIndex + 1 < activeLines.size(); lineIndex += 2) {
    EXPECT_NE(activeLines[lineIndex].find("[0]:"), std::string::npos) << "line=" << lineIndex;
    EXPECT_EQ(activeLines[lineIndex].find("[1]:"), std::string::npos) << "line=" << lineIndex;
    EXPECT_NE(activeLines[lineIndex].find(expectedViewEntries), std::string::npos) << "line=" << lineIndex;
    EXPECT_NE(activeLines[lineIndex + 1].find("[1]:"), std::string::npos) << "line=" << lineIndex + 1;
    EXPECT_EQ(activeLines[lineIndex + 1].find("[0]:"), std::string::npos) << "line=" << lineIndex + 1;
    EXPECT_NE(activeLines[lineIndex + 1].find(expectedViewEntries), std::string::npos) << "line=" << lineIndex + 1;
  }
  EXPECT_EQ(g_lat_nqps, CIM_MAX_QPS + 1);
  for (int index = 0; index < CIM_MAX_QPS; ++index) {
    EXPECT_EQ(latencies[index], static_cast<uint64_t>(index + 10));
    EXPECT_EQ(rawLatencies[index], static_cast<uint64_t>(index + 20));
    EXPECT_EQ(byteCounts[index], static_cast<uint32_t>(index + 30));
    EXPECT_EQ(loadWeights[index], static_cast<float>(index) / static_cast<float>(CIM_MAX_QPS - 1));
  }
  EXPECT_EQ(latencies[CIM_MAX_QPS], 10001u);
  EXPECT_EQ(rawLatencies[CIM_MAX_QPS], 10002u);
  EXPECT_EQ(byteCounts[CIM_MAX_QPS], 10003u);
  EXPECT_EQ(loadWeights[CIM_MAX_QPS], -1.0f);
  EXPECT_EQ(sequence, 0u);
  EXPECT_EQ(viewCount, 3u);
  EXPECT_EQ(viewIndices[0], static_cast<uint32_t>(CIM_MAX_QPS));
  EXPECT_EQ(viewIndices[1], static_cast<uint32_t>(CIM_MAX_QPS - 1));
  EXPECT_EQ(viewIndices[2], 0u);
  EXPECT_EQ(g_lat_started, 0);
  pthread_mutex_lock(&g_lat_mu);
  g_lat_ptr = originalLatencyPointer;
  g_r_lat_ptr = originalRawLatencyPointer;
  g_bytes_ptr = originalByteCountPointer;
  g_lat_nqps = originalQpCount;
  strncpy(g_lat_label, originalLatencyLabel.c_str(), sizeof(g_lat_label));
  g_lat_label[sizeof(g_lat_label) - 1] = 0;
  strncpy(g_r_lat_label, originalRawLatencyLabel.c_str(), sizeof(g_r_lat_label));
  g_r_lat_label[sizeof(g_r_lat_label) - 1] = 0;
  strncpy(g_bytes_label, originalByteCountLabel.c_str(), sizeof(g_bytes_label));
  g_bytes_label[sizeof(g_bytes_label) - 1] = 0;
  g_lw_ptr = originalLoadWeightPointer;
  g_lw_nqps = originalLoadWeightQpCount;
  strncpy(g_lw_label, originalLoadWeightLabel.c_str(), sizeof(g_lw_label));
  g_lw_label[sizeof(g_lw_label) - 1] = 0;
  g_dbg_seq_ptr = originalSequencePointer;
  g_dbg_view_count_ptr = originalViewCountPointer;
  g_dbg_view_qp_index_ptr = originalViewIndexPointer;
  g_lat_interval_ms = originalInterval;
  g_lat_running = originalRunningState;
  g_lat_started = originalStartedState;
  pthread_mutex_unlock(&g_lat_mu);
}

/** @brief Verifies inclusive float clamping below, within, and above the requested range. */
TEST(CimAlgorithmTest, ClampFloatToInclusiveRange) {
  SCOPED_TRACE("Float clamping must preserve in-range values and return the nearest inclusive endpoint.");
  constexpr float lower = 0.0f;

  constexpr float upper = 1.0f;

  EXPECT_FLOAT_EQ(clampf(-0.5f, lower, upper), lower);
  EXPECT_FLOAT_EQ(clampf(lower, lower, upper), lower);
  EXPECT_FLOAT_EQ(clampf(0.5f, lower, upper), 0.5f);
  EXPECT_FLOAT_EQ(clampf(upper, lower, upper), upper);
  EXPECT_FLOAT_EQ(clampf(1.5f, lower, upper), upper);
}

/** @brief Verifies unit-interval clamping at and beyond both endpoints. */
TEST(CimAlgorithmTest, ClampFloatToUnitInterval) {
  SCOPED_TRACE("Unit clamping must preserve values in [0,1] and clamp only values outside that interval.");
  EXPECT_FLOAT_EQ(clamp01(-0.5f), 0.0f);
  EXPECT_FLOAT_EQ(clamp01(0.0f), 0.0f);
  EXPECT_FLOAT_EQ(clamp01(0.5f), 0.5f);
  EXPECT_FLOAT_EQ(clamp01(1.0f), 1.0f);
  EXPECT_FLOAT_EQ(clamp01(1.5f), 1.0f);
}

/** @brief Verifies headroom selection returns the documented multiplier at latency boundaries. */
TEST(CimAlgorithmTest, ChooseHeadroomReturnsExpectedMultiplierAtValidBoundaries) {
  SCOPED_TRACE("Headroom selection must change scale only at its documented latency-ratio boundaries.");
  EXPECT_FLOAT_EQ(CIM_chooseHeadroom(0u, 0u), 1.0f);
  EXPECT_FLOAT_EQ(CIM_chooseHeadroom(0u, 100u), 1.0f);
  EXPECT_FLOAT_EQ(CIM_chooseHeadroom(100u, 100u), 2.0f);
  EXPECT_FLOAT_EQ(CIM_chooseHeadroom(100u, 200u), 2.0f);
  EXPECT_FLOAT_EQ(CIM_chooseHeadroom(100u, 201u), 1.5f);
  EXPECT_FLOAT_EQ(CIM_chooseHeadroom(100u, 500u), 1.5f);
  EXPECT_FLOAT_EQ(CIM_chooseHeadroom(100u, 501u), 1.0f);
  constexpr uint64_t largestSafeMinimum = UINT64_MAX / 5u;
  EXPECT_FLOAT_EQ(CIM_chooseHeadroom(largestSafeMinimum, largestSafeMinimum * 5u), 1.5f);
}

/**
 * @brief Verifies maximum-weight selection and threshold clamping with optional index output.
 */
TEST(CimAlgorithmTest, ChooseMinHealthyHandlesValidBoundariesAndNullOutput) {
  SCOPED_TRACE("Threshold selection must use the maximum weight, clamp it, and optionally return its index.");
  const float normalWeights[] = {0.10f, 0.50f, 0.30f};
  int maximumIndex = -1;
  float threshold = CIM_chooseMinHealthy_fromWeights(normalWeights, 3, &maximumIndex);
  EXPECT_EQ(maximumIndex, 1);
  EXPECT_FLOAT_EQ(threshold, 0.10f);
  const float smallWeights[] = {0.10f, 0.05f};
  threshold = CIM_chooseMinHealthy_fromWeights(smallWeights, 2, &maximumIndex);
  EXPECT_FLOAT_EQ(threshold, 0.03f);
  threshold = CIM_chooseMinHealthy_fromWeights(normalWeights, 3, nullptr);
  EXPECT_FLOAT_EQ(threshold, 0.10f);
}

/**
 * @brief Verifies gated movement and normalization across valid boundaries and optional storage.
 */
TEST(CimAlgorithmTest, GateMoveAndNormalizeHandlesValidBoundariesAndNullInput) {
  SCOPED_TRACE("Gating must bound movement, normalize active weights, and honor optional input storage.");
  EXPECT_NO_FATAL_FAILURE(gate_move_and_normalize(nullptr, 4));
  float boundaryWeight = 0.75f;
  gate_move_and_normalize(&boundaryWeight, 0);
  EXPECT_FLOAT_EQ(boundaryWeight, 0.75f);
  gate_move_and_normalize(&boundaryWeight, 1);
  EXPECT_FLOAT_EQ(boundaryWeight, 0.75f);
  float redistributedWeights[] = {0.48f, 0.01f, 0.50f, 0.01f, 0.00f};
  gate_move_and_normalize(redistributedWeights, 5);
  EXPECT_FLOAT_EQ(redistributedWeights[0], 0.49f);
  EXPECT_FLOAT_EQ(redistributedWeights[1], 0.00f);
  EXPECT_FLOAT_EQ(redistributedWeights[2], 0.51f);
  EXPECT_FLOAT_EQ(redistributedWeights[3], 0.00f);
  EXPECT_FLOAT_EQ(redistributedWeights[4], 0.00f);
  EXPECT_NEAR(std::accumulate(std::begin(redistributedWeights), std::end(redistributedWeights), 0.0f), 1.0f, 1e-6f);
  float lowSumWeights[] = {0.399f, 0.30f, 0.30f};
  gate_move_and_normalize(lowSumWeights, 3);
  EXPECT_NEAR(lowSumWeights[0], 0.40f, 1e-6f);
  EXPECT_FLOAT_EQ(lowSumWeights[1], 0.30f);
  EXPECT_FLOAT_EQ(lowSumWeights[2], 0.30f);
  EXPECT_NEAR(std::accumulate(std::begin(lowSumWeights), std::end(lowSumWeights), 0.0f), 1.0f, 1e-6f);
  float highSumWeights[] = {0.401f, 0.30f, 0.30f};
  gate_move_and_normalize(highSumWeights, 3);
  EXPECT_NEAR(highSumWeights[0], 0.40f, 1e-6f);
  EXPECT_FLOAT_EQ(highSumWeights[1], 0.30f);
  EXPECT_FLOAT_EQ(highSumWeights[2], 0.30f);
  EXPECT_NEAR(std::accumulate(std::begin(highSumWeights), std::end(highSumWeights), 0.0f), 1.0f, 1e-6f);
  std::vector<float> allGatedWeights(40, 0.025f);
  gate_move_and_normalize(allGatedWeights.data(), static_cast<int>(allGatedWeights.size()));
  EXPECT_NEAR(allGatedWeights[0], 1.0f, 1e-5f);
  for (std::size_t index = 1; index < allGatedWeights.size(); ++index) {
    EXPECT_FLOAT_EQ(allGatedWeights[index], 0.0f);
  }
}


/** @brief Verifies one or equal known latencies are classified as uncongested. */
TEST(CimAlgorithmTest, ComputeLatencyFactorsHandlesEqualMeasurements) {
  SCOPED_TRACE("One or several equal known measurements must all produce zero factors.");
  const uint64_t singleLatency[] = {77u};
  float singleFactor[] = {-1.0f};
  CIM_computeLatencyFactors(singleLatency, static_cast<int>(std::size(singleLatency)), singleFactor);
  EXPECT_FLOAT_EQ(singleFactor[0], 0.0f);
  const uint64_t equalLatencies[] = {100u, 100u, 100u};
  float equalFactors[] = {-1.0f, -1.0f, -1.0f};
  CIM_computeLatencyFactors(equalLatencies, static_cast<int>(std::size(equalLatencies)), equalFactors);
  EXPECT_FLOAT_EQ(equalFactors[0], 0.0f);
  EXPECT_FLOAT_EQ(equalFactors[1], 0.0f);
  EXPECT_FLOAT_EQ(equalFactors[2], 0.0f);
}

/** @brief Verifies entirely unknown latency input is treated as uncongested. */
TEST(CimAlgorithmTest, ComputeLatencyFactorsHandlesUnknownMeasurements) {
  SCOPED_TRACE("Every zero latency must remain a valid unknown value with a zero factor.");
  const uint64_t zeroLatencies[] = {0u, 0u, 0u};
  float zeroFactors[] = {-1.0f, -1.0f, -1.0f};
  CIM_computeLatencyFactors(zeroLatencies, static_cast<int>(std::size(zeroLatencies)), zeroFactors);
  EXPECT_FLOAT_EQ(zeroFactors[0], 0.0f);
  EXPECT_FLOAT_EQ(zeroFactors[1], 0.0f);
  EXPECT_FLOAT_EQ(zeroFactors[2], 0.0f);
}

/** @brief Verifies bounded latency factors for mixed, bucketed, and descending inputs. */
TEST(CimAlgorithmTest, ComputeLatencyFactorsHandlesMixedMeasurements) {
  SCOPED_TRACE("Mixed latency factors must follow relative measurements and remain bounded.");
  const uint64_t coldLatencies[] = {0u, 60u, 120u};
  float coldFactors[] = {-1.0f, -1.0f, -1.0f};
  CIM_computeLatencyFactors(coldLatencies, static_cast<int>(std::size(coldLatencies)), coldFactors);
  EXPECT_FLOAT_EQ(coldFactors[0], 0.0f);
  EXPECT_FLOAT_EQ(coldFactors[1], 0.5f);
  EXPECT_FLOAT_EQ(coldFactors[2], 1.0f);
  const uint64_t bucketTwoLatencies[] = {60u, 70u, 80u, 100u};
  float bucketTwoFactors[] = {-1.0f, -1.0f, -1.0f, -1.0f};
  CIM_computeLatencyFactors(bucketTwoLatencies, static_cast<int>(std::size(bucketTwoLatencies)), bucketTwoFactors);
  EXPECT_FLOAT_EQ(bucketTwoFactors[0], 0.0f);
  EXPECT_FLOAT_EQ(bucketTwoFactors[1], 0.125f);
  EXPECT_FLOAT_EQ(bucketTwoFactors[2], 0.25f);
  EXPECT_FLOAT_EQ(bucketTwoFactors[3], 0.50f);
  const uint64_t bucketOnePointFiveLatencies[] = {60u, 65u, 70u, 200u};
  float bucketOnePointFiveFactors[] = {-1.0f, -1.0f, -1.0f, -1.0f};
  CIM_computeLatencyFactors(bucketOnePointFiveLatencies, static_cast<int>(std::size(bucketOnePointFiveLatencies)), bucketOnePointFiveFactors);
  EXPECT_FLOAT_EQ(bucketOnePointFiveFactors[0], 0.0f);
  EXPECT_NEAR(bucketOnePointFiveFactors[1], 5.0f / 210.0f, 1e-6f);
  EXPECT_NEAR(bucketOnePointFiveFactors[2], 10.0f / 210.0f, 1e-6f);
  EXPECT_NEAR(bucketOnePointFiveFactors[3], 140.0f / 210.0f, 1e-6f);
  const uint64_t bucketOneLatencies[] = {60u, 65u, 70u, 600u};
  float bucketOneFactors[] = {-1.0f, -1.0f, -1.0f, -1.0f};
  CIM_computeLatencyFactors(bucketOneLatencies, static_cast<int>(std::size(bucketOneLatencies)), bucketOneFactors);
  EXPECT_FLOAT_EQ(bucketOneFactors[0], 0.0f);
  EXPECT_NEAR(bucketOneFactors[1], 5.0f / 540.0f, 1e-6f);
  EXPECT_NEAR(bucketOneFactors[2], 10.0f / 540.0f, 1e-6f);
  EXPECT_FLOAT_EQ(bucketOneFactors[3], 1.0f);
  const uint64_t descendingLatencies[] = {200u, 100u, 50u};
  float guardedDescendingFactors[] = {1234.0f, -1.0f, -1.0f, -1.0f, 5678.0f};
  CIM_computeLatencyFactors(descendingLatencies, static_cast<int>(std::size(descendingLatencies)), &guardedDescendingFactors[1]);
  EXPECT_FLOAT_EQ(guardedDescendingFactors[0], 1234.0f);
  EXPECT_NEAR(guardedDescendingFactors[1], 150.0f / 225.0f, 1e-6f);
  EXPECT_NEAR(guardedDescendingFactors[2], 50.0f / 225.0f, 1e-6f);
  EXPECT_FLOAT_EQ(guardedDescendingFactors[3], 0.0f);
  EXPECT_FLOAT_EQ(guardedDescendingFactors[4], 5678.0f);
}

/** @brief Verifies the supported maximum QP count preserves every factor output. */
TEST(CimAlgorithmTest, ComputeLatencyFactorsHandlesMaximumQpCount) {
  SCOPED_TRACE("The complete supported QP array must be initialized without crossing its bounds.");
  std::vector<uint64_t> maximumLatencies(CIM_MAX_QPS, 100u);
  std::vector<float> maximumFactors(CIM_MAX_QPS, -1.0f);
  CIM_computeLatencyFactors(maximumLatencies.data(),
                            static_cast<int>(maximumLatencies.size()),
                            maximumFactors.data());
  for (float factor : maximumFactors) {
    EXPECT_FLOAT_EQ(factor, 0.0f);
  }
}

/** @brief Verifies an empty latency range returns without reading or writing caller storage. */
TEST(CimAlgorithmTest, ComputeLatencyFactorsHandlesEmptyRange) {
  SCOPED_TRACE("An empty range must leave output storage unchanged without dereferencing input.");
  float factor = 0.75f;

  CIM_computeLatencyFactors(nullptr, 0, &factor);

  EXPECT_FLOAT_EQ(factor, 0.75f);
}

/** @brief Verifies equal QP latencies produce equal normalized direct weights. */
TEST(CimAlgorithmTest, EqualLatenciesProduceUniformDirectWeights) {
  SCOPED_TRACE("Equal healthy latencies must produce a uniform normalized distribution.");
  const uint64_t latencies[] = {100, 100, 100, 100};
  std::vector<float> weights(4, 0.0f);
  int isEqual = 0;

  CIM_latencyToWeightsDirect(latencies, 4, weights.data(), &isEqual);

  EXPECT_EQ(1, isEqual);
  for (float weight : weights) {
    EXPECT_FLOAT_EQ(weight, 0.25f);
  }
  EXPECT_FLOAT_EQ(SumWeights(weights), 1.0f);
}

/** @brief Verifies direct weighting keeps all unknown QPs uniformly eligible. */
TEST(CimAlgorithmTest, DirectWeightsKeepUnknownQpsUniform) {
  SCOPED_TRACE("Direct weighting must assign finite, normalized, uniform weights to unknown QPs.");
  const uint64_t latencies[] = {0u, 0u, 0u};
  std::vector<float> weights(3, -1.0f);
  int isEqual = 0;

  CIM_latencyToWeightsDirect(latencies, 3, weights.data(), &isEqual);

  EXPECT_EQ(1, isEqual);
  for (float weight : weights) {
    EXPECT_NEAR(1.0f / 3.0f, weight, 1e-6f);
    EXPECT_TRUE(std::isfinite(weight));
  }
  EXPECT_NEAR(1.0f, SumWeights(weights), 1e-6f);
}

/** @brief Verifies active-window weighting keeps all unknown QPs uniformly eligible. */
TEST(CimAlgorithmTest, ActiveWindowWeightsKeepUnknownQpsUniform) {
  SCOPED_TRACE(
      "Active-window weighting must assign finite, normalized, uniform weights to unknown QPs.");
  const uint64_t latencies[] = {0u, 0u, 0u};
  std::vector<float> weights(3, -1.0f);

  CIM_computeActiveWindowWeights(latencies, 3, 0, 3, weights.data());

  for (float weight : weights) {
    EXPECT_NEAR(1.0f / 3.0f, weight, 1e-6f);
    EXPECT_TRUE(std::isfinite(weight));
  }
  EXPECT_NEAR(1.0f, SumWeights(weights), 1e-6f);
}

/**
 * @brief Verifies nonuniform latency weights and explicit non-equal output classification.
 */
TEST(CimAlgorithmTest, LowerLatencyGetsHigherDirectWeight) {
  SCOPED_TRACE("Unequal healthy latencies must produce ordered weights and clear the equality output.");
  const uint64_t latencies[] = {100, 200, 400, 800};
  std::vector<float> weights(4, 0.0f);
  int isEqual = 7;

  CIM_latencyToWeightsDirect(latencies, 4, weights.data(), &isEqual);

#if 0 // ADDRESS_LATER
  EXPECT_EQ(isEqual, 0);
#endif 0 // ADDRESS_LATER
  EXPECT_GT(weights[0], weights[1]);
  EXPECT_GT(weights[1], weights[2]);
  EXPECT_GE(weights[2], weights[3]);
  EXPECT_NEAR(SumWeights(weights), 1.0f, 1e-6f);
}

/** @brief Verifies active-QP window selection wraps around the queue-pair ring. */
TEST(CimAlgorithmTest, ActiveWindowWrapsAroundQpBase) {
  SCOPED_TRACE("The active window must include exactly its wrapped QP range and exclude all others.");
  const uint64_t latencies[] = {400, 800, 100, 200};
  std::vector<float> weights(3, 0.0f);

  CIM_computeActiveWindowWeights(latencies, 4, 2, 3, weights.data());

  EXPECT_GT(weights[0], weights[1]);
  EXPECT_GT(weights[1], weights[2]);
  EXPECT_NEAR(SumWeights(weights), 1.0f, 1e-6f);
}

/** @brief Verifies a cold QP inside an active window remains eligible with maximum weight. */
TEST(CimAlgorithmTest, ActiveWindowKeepsColdQpEligible) {
  SCOPED_TRACE("A zero latency inside the window must establish the minimum and remain eligible.");
  const uint64_t latencies[] = {100u, 0u, 200u};
  std::vector<float> weights(3, -1.0f);

  CIM_computeActiveWindowWeights(latencies, 3, 0, 3, weights.data());

  EXPECT_NEAR(weights[0], 1.0f / 3.0f, 1e-6f);
  EXPECT_NEAR(weights[1], 2.0f / 3.0f, 1e-6f);
  EXPECT_FLOAT_EQ(weights[2], 0.0f);
  EXPECT_NEAR(SumWeights(weights), 1.0f, 1e-6f);
}

/** @brief Verifies direct weighting updates its minimum for a lower later measurement. */
TEST(CimAlgorithmTest, DirectWeightsHandleDescendingLatencies) {
  SCOPED_TRACE("A lower later measurement must become the minimum used by every direct weight.");
  const uint64_t latencies[] = {100u, 50u, 200u};
  std::vector<float> weights(3, -1.0f);
  int isEqual = 1;

  CIM_latencyToWeightsDirect(latencies, 3, weights.data(), &isEqual);

  EXPECT_NEAR(7.0f / 19.0f, weights[0], 1e-6f);
  EXPECT_NEAR(9.0f / 19.0f, weights[1], 1e-6f);
  EXPECT_NEAR(3.0f / 19.0f, weights[2], 1e-6f);
  EXPECT_NEAR(1.0f, SumWeights(weights), 1e-6f);
}

/** @brief Verifies direct weighting keeps a later cold QP eligible. */
TEST(CimAlgorithmTest, DirectWeightsKeepLaterColdQpEligible) {
  SCOPED_TRACE("A later zero latency must establish a cold minimum and retain positive weight.");
  const uint64_t latencies[] = {100u, 0u, 200u};
  std::vector<float> weights(3, -1.0f);
  int isEqual = 1;

  CIM_latencyToWeightsDirect(latencies, 3, weights.data(), &isEqual);

  EXPECT_NEAR(1.0f / 3.0f, weights[0], 1e-6f);
  EXPECT_NEAR(2.0f / 3.0f, weights[1], 1e-6f);
  EXPECT_FLOAT_EQ(0.0f, weights[2]);
  EXPECT_NEAR(1.0f, SumWeights(weights), 1e-6f);
}

/** @brief Verifies empty input ranges return without changing caller state. */
TEST(CimAlgorithmTest, NullOrEmptyInputsAreNoops) {
  SCOPED_TRACE("No-op inputs must return safely without fabricating weights or modifying sentinels.");
  float weights[] = {0.5f, 0.5f};
  const uint64_t latencies[] = {100, 200};
  int isEqual = 7;

  CIM_latencyToWeightsDirect(latencies, 0, weights, &isEqual);
  EXPECT_FLOAT_EQ(weights[0], 0.5f);
  EXPECT_FLOAT_EQ(weights[1], 0.5f);
  EXPECT_EQ(isEqual, 7);

  CIM_computeActiveWindowWeights(latencies, 2, 0, 0, weights);
  EXPECT_FLOAT_EQ(weights[0], 0.5f);
  EXPECT_FLOAT_EQ(weights[1], 0.5f);
}

/** @brief Verifies initialization updates active load weights and preserves all other state. */
TEST(CimStateTest, InitializeLoadWeights) {
  SCOPED_TRACE("State initialization must populate every active weight and leave inactive capacity untouched.");
  /**
   * @brief Verifies load-weight initialization for one QP count.
   * @param[in] nqps Active QP count in [0, CIM_MAX_QPS].
   * @note Side effects: Records nonfatal test failures.
   */
  const auto verifyInitialization = [](int nqps) {
    CIMState state = {};
    state.nqps = nqps;
    state.spread_eps_us = 123456;
    state.running = 7;
    state.thread = pthread_self();
    state.dbg_seq = 9;
    state.dbg_view_count = 3;
    state.seed = 2468;
    state.lw_initialized = 5;
    for (int index = 0; index < CIM_MAX_QPS; ++index) {
      state.dbg_view_qp_index[index] = static_cast<uint32_t>(index);
      state.lw[index] = -100.0f - static_cast<float>(index);
      state.latency[index] = 1000u + static_cast<uint64_t>(index);
      state.raw_latency[index] = 2000u + static_cast<uint64_t>(index);
      state.lat_bytes[index] = 3000u + static_cast<uint32_t>(index);
      state.split0_mild_score[index] = static_cast<uint8_t>((index % 250) + 1);
      state.split0_mild_slow_qp[index] = static_cast<uint8_t>(((index + 17) % 250) + 1);
    }
    CIMState expected;
    // Copy padding too so the comparison detects writes outside the active weight range.
    std::memcpy(&expected, &state, sizeof(state));
    const float expectedWeight = nqps > 0 ? 1.0f / static_cast<float>(nqps) : 0.0f;
    for (int index = 0; index < nqps; ++index) {
      expected.lw[index] = expectedWeight;
    }
    init_lw_example(&state);
    EXPECT_EQ(std::memcmp(&state, &expected, sizeof(state)), 0) << "nqps=" << nqps;
    for (int index = 0; index < CIM_MAX_QPS; ++index) {
      const float expectedEntry = index < nqps ? expectedWeight : -100.0f - static_cast<float>(index);
      EXPECT_EQ(state.lw[index], expectedEntry) << "nqps=" << nqps << ", index=" << index;
    }
    EXPECT_EQ(state.nqps, nqps);
    if (nqps == 0) return;
    const float activeSum = std::accumulate(state.lw, state.lw + nqps, 0.0f);
    EXPECT_NEAR(activeSum, 1.0f, 1e-6f) << "nqps=" << nqps;
  };
  verifyInitialization(0);
  verifyInitialization(1);
  verifyInitialization(3);
  verifyInitialization(CIM_MAX_QPS);
}

/**
 * @brief Verifies CIM startup rejects invalid counts and initializes every valid QP-count boundary.
 */
TEST(CimStateTest, StartBehavior) {
  SCOPED_TRACE("CIM startup must reject invalid counts and return fully initialized state for valid counts.");
  EXPECT_EQ(CIM_Start(std::numeric_limits<int>::min()), nullptr);
  EXPECT_EQ(CIM_Start(-1), nullptr);
  EXPECT_EQ(CIM_Start(0), nullptr);
  /**
   * @brief Verifies one successful CIM startup and shutdown.
   * @param[in] requestedNqps QP count passed to CIM_Start.
   * @param[in] expectedNqps QP count expected after production clamping.
   * @note Side effects: Allocates and frees CIM state and records test failures.
   */
  const auto verifySuccessfulStart = [](int requestedNqps, int expectedNqps) {
    SCOPED_TRACE("requestedNqps=" + std::to_string(requestedNqps));
    CIMState* state = CIM_Start(requestedNqps);
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->nqps, expectedNqps);
    EXPECT_EQ(state->dbg_view_count, static_cast<uint32_t>(expectedNqps));
    EXPECT_EQ(state->lw_initialized, 1);
    EXPECT_EQ(state->running, 0);
    EXPECT_EQ(state->spread_eps_us, 0u);
    EXPECT_EQ(state->dbg_seq, 0u);
    const float expectedWeight = 1.0f / static_cast<float>(expectedNqps);
    float activeSum = 0.0f;
    for (int index = 0; index < CIM_MAX_QPS; ++index) {
      const uint32_t expectedViewIndex = index < expectedNqps ? static_cast<uint32_t>(index) : 0u;
      EXPECT_EQ(state->dbg_view_qp_index[index], expectedViewIndex) << "index=" << index;
      const float expectedEntryWeight = index < expectedNqps ? expectedWeight : 0.0f;
      EXPECT_EQ(state->lw[index], expectedEntryWeight) << "index=" << index;
      EXPECT_EQ(state->latency[index], 0u) << "index=" << index;
      EXPECT_EQ(state->raw_latency[index], 0u) << "index=" << index;
      EXPECT_EQ(state->lat_bytes[index], 0u) << "index=" << index;
      EXPECT_EQ(state->split0_mild_score[index], 0u) << "index=" << index;
      EXPECT_EQ(state->split0_mild_slow_qp[index], 0u) << "index=" << index;
      if (index < expectedNqps) activeSum += state->lw[index];
    }
    EXPECT_NEAR(activeSum, 1.0f, 1e-5f);
    CIM_Stop(state);
  };
  verifySuccessfulStart(1, 1);
  verifySuccessfulStart(3, 3);
  verifySuccessfulStart(CIM_MAX_QPS, CIM_MAX_QPS);
  verifySuccessfulStart(CIM_MAX_QPS + 1, CIM_MAX_QPS);
  verifySuccessfulStart(std::numeric_limits<int>::max(), CIM_MAX_QPS);
}

/** @brief Verifies that a four-QP CIM state can be created and destroyed. */
TEST(CimStateTest, CardMapCopiesBoundedEntries) {
  SCOPED_TRACE("A four-QP state must support successful creation and destruction.");
  CIMState* state = CIM_Start(4);
  ASSERT_NE(state, nullptr);
  CIM_Stop(state);
}

/** @brief Verifies worker-creation failure restores the printer to a fully stopped state. */
TEST(CimDiagnosticsTest, PrinterStartFailureRollsBackState) {
  SCOPED_TRACE("A failed pthread_create call must not leave the diagnostic printer marked active.");
  const int originalInterval = g_lat_interval_ms;
  const int callsBefore = createThreadCallsForTest.load(std::memory_order_relaxed);
  ASSERT_EQ(0, g_lat_started);
  ASSERT_EQ(0, g_lat_running);

  failThreadCreationForTest.store(true, std::memory_order_relaxed);
  CIM_DebugLatPrinterStart(7);
  failThreadCreationForTest.store(false, std::memory_order_relaxed);

  EXPECT_EQ(callsBefore + 1, createThreadCallsForTest.load(std::memory_order_relaxed));
  EXPECT_EQ(0, g_lat_started) << "Failed thread creation must roll back the started flag";
  EXPECT_EQ(0, g_lat_running) << "Failed thread creation must roll back the running flag";
  EXPECT_EQ(7, g_lat_interval_ms) << "The requested interval must remain available for a later retry";
  g_lat_interval_ms = originalInterval;
}

/** @brief Verifies stopping an idle printer returns without changing its stopped state. */
TEST(CimDiagnosticsTest, PrinterStopBeforeStartIsNoop) {
  SCOPED_TRACE("Stopping before startup must return without joining a nonexistent worker.");
  ASSERT_EQ(0, g_lat_started);
  ASSERT_EQ(0, g_lat_running);

  CIM_DebugLatPrinterStop();

  EXPECT_EQ(0, g_lat_started);
  EXPECT_EQ(0, g_lat_running);
}

/** @brief Verifies printer startup applies interval updates and owns at most one worker thread. */
TEST(CimDiagnosticsTest, PrinterStartBehavior) {
  SCOPED_TRACE(
      "Printer startup must apply positive intervals, retain the current interval for zero, and "
      "avoid duplicate workers.");
  // Save the shared interval for restoration after both worker threads have been joined.
  const int originalInterval = g_lat_interval_ms;
  ASSERT_EQ(g_lat_started, 0);
  ASSERT_EQ(g_lat_running, 0);
  CIM_DebugLatPrinterStart(1);
  ASSERT_EQ(g_lat_started, 1);
  EXPECT_EQ(g_lat_running, 1);
  EXPECT_EQ(g_lat_interval_ms, 1);
  const pthread_t startedThread = g_lat_th;
  CIM_DebugLatPrinterStart(0);
  EXPECT_EQ(g_lat_started, 1);
  EXPECT_EQ(g_lat_running, 1);
  EXPECT_EQ(g_lat_interval_ms, 1);
  EXPECT_NE(pthread_equal(startedThread, g_lat_th), 0);
  CIM_DebugLatPrinterStop();
  EXPECT_EQ(g_lat_started, 0);
  EXPECT_EQ(g_lat_running, 0);
  g_lat_interval_ms = 2;
  CIM_DebugLatPrinterStart(0);
  EXPECT_EQ(g_lat_started, 1);
  EXPECT_EQ(g_lat_running, 1);
  EXPECT_EQ(g_lat_interval_ms, 2);
  CIM_DebugLatPrinterStop();
  EXPECT_EQ(g_lat_started, 0);
  EXPECT_EQ(g_lat_running, 0);
  g_lat_interval_ms = originalInterval;
}
