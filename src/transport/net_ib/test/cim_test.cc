/*
Copyright (c) 2026 OCCL-net.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "cim.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <thread>
#include <vector>

#include "../cim.cc"  // NOLINT(build/include)

namespace {

float SumWeights(const std::vector<float>& weights) {
  return std::accumulate(weights.begin(), weights.end(), 0.0f);
}

}  // namespace

TEST(CimWeightsTest, EqualLatenciesProduceUniformDirectWeights) {
  const uint64_t latencies[] = {100, 100, 100, 100};
  std::vector<float> weights(4, 0.0f);
  int isEqual = 0;

  CIM_latencyToWeightsDirect(latencies, 4, weights.data(), &isEqual);

  EXPECT_EQ(isEqual, 1);
  for (float weight : weights) {
    EXPECT_FLOAT_EQ(weight, 0.25f);
  }
  EXPECT_FLOAT_EQ(SumWeights(weights), 1.0f);
}

TEST(CimWeightsTest, LowerLatencyGetsHigherDirectWeight) {
  const uint64_t latencies[] = {100, 200, 400, 800};
  std::vector<float> weights(4, 0.0f);
  int isEqual = 1;

  CIM_latencyToWeightsDirect(latencies, 4, weights.data(), &isEqual);

  EXPECT_EQ(isEqual, 0);
  EXPECT_GT(weights[0], weights[1]);
  EXPECT_GT(weights[1], weights[2]);
  EXPECT_GE(weights[2], weights[3]);
  EXPECT_NEAR(SumWeights(weights), 1.0f, 1e-6f);
}

TEST(CimWeightsTest, ActiveWindowWrapsAroundQpBase) {
  const uint64_t latencies[] = {400, 800, 100, 200};
  std::vector<float> weights(3, 0.0f);

  CIM_computeActiveWindowWeights(latencies, 4, 2, 3, weights.data());

  EXPECT_GT(weights[0], weights[1]);
  EXPECT_GT(weights[1], weights[2]);
  EXPECT_NEAR(SumWeights(weights), 1.0f, 1e-6f);
}

TEST(CimWeightsTest, NullOrEmptyInputsAreNoops) {
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

TEST(CimStateTest, StartClampsQpCountAndInitializesUniformWeights) {
  CIMState* state = CIM_Start(CIM_MAX_QPS + 100);
  ASSERT_NE(state, nullptr);

  EXPECT_EQ(state->nqps, CIM_MAX_QPS);
  EXPECT_EQ(state->lw_initialized, 1);
  EXPECT_EQ(state->card_map_valid, 0);
  EXPECT_EQ(state->n_cards, 0);

  float sum = 0.0f;
  for (int i = 0; i < state->nqps; i++) {
    EXPECT_FLOAT_EQ(state->lw[i], 1.0f / static_cast<float>(CIM_MAX_QPS));
    EXPECT_EQ(state->qp_card[i], 0);
    sum += state->lw[i];
  }
  EXPECT_NEAR(sum, 1.0f, 1e-5f);

  CIM_Stop(state);
}

TEST(CimStateTest, CardMapCopiesBoundedEntries) {
  CIMState* state = CIM_Start(4);
  ASSERT_NE(state, nullptr);

  const uint16_t cardMap[] = {3, 2, 1, 0, 9, 9};
  CIM_SetQpCardMap(state, cardMap, 6, 4);

  EXPECT_EQ(state->card_map_valid, 1);
  EXPECT_EQ(state->n_cards, 4);
  EXPECT_EQ(state->qp_card[0], 3);
  EXPECT_EQ(state->qp_card[1], 2);
  EXPECT_EQ(state->qp_card[2], 1);
  EXPECT_EQ(state->qp_card[3], 0);

  CIM_Stop(state);
}

TEST(CimStateTest, InvalidStartReturnsNull) {
  EXPECT_EQ(CIM_Start(0), nullptr);
  EXPECT_EQ(CIM_Start(-1), nullptr);
}

TEST(CimDebugPrinterTest, SourceSettersAndStartStopAreRepeatable) {
  volatile uint64_t latencies[] = {0, 0};
  volatile uint64_t rawLatencies[] = {0, 0};
  volatile float weights[] = {0.25f, 0.75f};
  volatile uint32_t seq = 0;

  CIM_DebugLatPrinterSetSource("test_lat", latencies, rawLatencies, 2);
  CIM_DebugLatPrinterSetWeightSource("test_lw", weights, 2);
  CIM_DebugLatPrinterSetSyncSeq(&seq);
  CIM_DebugLatPrinterStart(1);
  CIM_DebugLatPrinterStart(2);

  latencies[0] = 10;
  rawLatencies[0] = 12;
  weights[0] = 0.5f;
  std::this_thread::sleep_for(std::chrono::milliseconds(5));

  CIM_DebugLatPrinterStop();
  CIM_DebugLatPrinterStop();

  CIM_DebugLatPrinterSetSource(nullptr, nullptr, nullptr, 0);
  CIM_DebugLatPrinterSetWeightSource(nullptr, nullptr, 0);
  CIM_DebugLatPrinterSetSyncSeq(nullptr);
}
