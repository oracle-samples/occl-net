/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 OCCL-net. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_CIM_H_
#define NET_IB_CIM_H_

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#define CIM_MAX_QPS 128
#define CIM_MAX_CARDS 16

typedef struct CIMState CIMState;

struct CIMState {
  int nqps;
  int running;
  pthread_t thread;
  volatile uint32_t dbg_seq;
  float lw[CIM_MAX_QPS];
  uint64_t latency[CIM_MAX_QPS];
  uint64_t raw_latency[CIM_MAX_QPS];
  uint32_t lat_bytes[CIM_MAX_QPS];
  unsigned int seed;
  int lw_initialized;
  uint16_t* qp_card;
  uint16_t n_cards;
  int card_map_valid;
};

CIMState* CIM_Start(int nqps);
void CIM_Stop(CIMState* s);

void CIM_SetQpCardMap(CIMState* s, const uint16_t* qp_card, int n, uint16_t n_cards);

void CIM_DebugLatPrinterStart(int interval_ms);
void CIM_DebugLatPrinterStop(void);
void CIM_DebugLatPrinterSetSource(const char* label, volatile uint64_t* lat, volatile uint64_t* r_lat, int nqps);
void CIM_DebugLatPrinterSetWeightSource(const char* label, volatile float* lw, int nqps);
void CIM_DebugLatPrinterSetByteSource(const char* label, volatile uint32_t* bytes, int nqps);
void CIM_DebugLatPrinterSetSyncSeq(volatile uint32_t* seq);

typedef struct {
  uint64_t cnt;
  uint64_t cyc_sum;
  uint64_t cyc_max;
} CycStat;

static inline void cyc_add(CycStat* st, uint64_t delta) {
  st->cnt++;
  st->cyc_sum += delta;
  if (delta > st->cyc_max) st->cyc_max = delta;
}

static inline float clampf(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

static inline float clamp01(float x) {
  return clampf(x, 0.0f, 1.0f);
}

static inline float CIM_chooseHeadroom(uint64_t minLat, uint64_t maxLat) {
  if (minLat == 0) return 1.0f;

  float ratio = (float)maxLat / (float)minLat;
  if (ratio <= 2.0f) return 2.0f;
  if (ratio <= 5.0f) return 1.5f;
  return 1.0f;
}

static inline float CIM_chooseMinHealthy_fromWeights(const float* lw, int n) {
  float wmax = 0.0f;
  for (int i = 0; i < n; i++) {
    if (lw[i] > wmax) wmax = lw[i];
  }

  const float gateFrac = 0.20f;
  const float absMin = 0.03f;
  const float absMax = 0.30f;

  float mh = gateFrac * wmax;
  return clampf(mh, absMin, absMax);
}

static inline void CIM_computeActiveWindowWeights(const uint64_t* lat, int nqps, int qpIndexBase, int nDataQps, float* weights) {
  if (weights == NULL || nqps <= 0 || nDataQps <= 0) return;
  if (nDataQps > nqps) nDataQps = nqps;

  uint64_t minLat = UINT64_MAX;
  uint64_t maxLat = 0;
  for (int i = 0; i < nDataQps; i++) {
    int q = (qpIndexBase + i) % nqps;
    uint64_t v = lat[q];
    if (v == 0) minLat = 0;
    if (v < minLat) minLat = v;
    if (v > maxLat) maxLat = v;
  }

  if (minLat == UINT64_MAX || maxLat <= minLat) {
    float w = 1.0f / (float)nDataQps;
    for (int i = 0; i < nDataQps; i++) weights[i] = w;
    return;
  }

  float headroomMult = CIM_chooseHeadroom(minLat, maxLat);
  if (headroomMult < 1.0f) headroomMult = 1.0f;

  double range = (double)(maxLat - minLat);
  double effMax = (double)minLat + headroomMult * range;
  double denom = effMax - (double)minLat;

  float sum = 0.0f;
  for (int i = 0; i < nDataQps; i++) {
    int q = (qpIndexBase + i) % nqps;
    uint64_t v = lat[q];
    float g = 1.0f;
    if (v != 0) {
      double x = ((double)v - (double)minLat) / denom;
      float lf = clamp01((float)x);
      g = 1.0f - lf;
      if (g < 0.0f) g = 0.0f;
    }
    weights[i] = g;
    sum += g;
  }

  if (sum > 0.0f) {
    float inv = 1.0f / sum;
    for (int i = 0; i < nDataQps; i++) weights[i] *= inv;
  } else {
    float w = 1.0f / (float)nDataQps;
    for (int i = 0; i < nDataQps; i++) weights[i] = w;
  }
}

static inline void CIM_latencyToWeightsDirect(const uint64_t* lat, int nqps, float* weights, int* is_equal) {
  if (weights == NULL || nqps <= 0) return;
  if (is_equal != NULL) *is_equal = 0;

  uint64_t minLat = UINT64_MAX;
  uint64_t maxLat = 0;
  for (int i = 0; i < nqps; i++) {
    uint64_t v = lat[i];
    if (v == 0) minLat = 0;
    if (v < minLat) minLat = v;
    if (v > maxLat) maxLat = v;
  }

  if (minLat == UINT64_MAX || maxLat <= minLat || (maxLat - minLat) <= 2) {
    float w = 1.0f / (float)nqps;
    for (int i = 0; i < nqps; i++) weights[i] = w;
    if (is_equal != NULL) *is_equal = 1;
    return;
  }

  float headroomMult = CIM_chooseHeadroom(minLat, maxLat);
  if (headroomMult < 1.0f) headroomMult = 1.0f;

  float denom = headroomMult * (float)(maxLat - minLat);
  if (denom <= 0.0f) {
    float w = 1.0f / (float)nqps;
    for (int i = 0; i < nqps; i++) weights[i] = w;
    if (is_equal != NULL) *is_equal = 1;
    return;
  }

  float sum = 0.0f;
  for (int i = 0; i < nqps; i++) {
    uint64_t v = lat[i];
    float g = 0.0f;
    if (v == 0) {
      g = denom;
    } else {
      float delta = (float)(v - minLat);
      g = denom - delta;
      if (g < 0.0f) g = 0.0f;
    }
    weights[i] = g;
    sum += g;
  }

  if (sum > 0.0f) {
    float inv = 1.0f / sum;
    for (int i = 0; i < nqps; i++) weights[i] *= inv;
  } else {
    float w = 1.0f / (float)nqps;
    for (int i = 0; i < nqps; i++) weights[i] = w;
    if (is_equal != NULL) *is_equal = 1;
  }
}

static inline void gate_move_and_normalize(CIMState* s, float* lw, int n) {
  if (lw == NULL || n <= 1) return;

  float min_healthy = CIM_chooseMinHealthy_fromWeights(lw, n);
  const int haveCardMap = (s != NULL && s->qp_card != NULL && s->card_map_valid && s->n_cards > 1);
  int nCards = haveCardMap ? (int)s->n_cards : 0;
  if (nCards > CIM_MAX_CARDS) nCards = CIM_MAX_CARDS;

  float sum_in = 0.0f;
  int maxIdx = 0;
  float maxV = lw[0];

  int healthy_count = 0;
  int has_unhealthy_pos = 0;

  int firstHealthyGlobal = -1;
  int lastGlobal = -1;
  float pendingGlobal = 0.0f;

  if (!haveCardMap || nCards <= 0 || s->qp_card == NULL) {
    for (int i = n - 1; i >= 0; i--) {
      float w = lw[i];
      if (w < 0.0f) w = 0.0f;
      sum_in += w;

      if (w > maxV) {
        maxV = w;
        maxIdx = i;
      }

      if (w >= min_healthy) {
        healthy_count++;
        firstHealthyGlobal = i;
        lastGlobal = i;
        continue;
      }

      if (w > 0.0f) {
        has_unhealthy_pos = 1;
        if (w < min_healthy) {
          lw[i] = 0.0f;
          if (lastGlobal >= 0) lw[lastGlobal] += w;
          else pendingGlobal += w;
        }
      }
    }

    if (!has_unhealthy_pos) {
      const float EPS = 1e-4f;
      float diff = 1.0f - sum_in;
      if (diff > EPS || diff < -EPS) lw[maxIdx] += diff;
      return;
    }

    if (healthy_count == 0) {
      lw[maxIdx] += pendingGlobal;
      const float EPS = 1e-4f;
      float diff = 1.0f - sum_in;
      if (diff > EPS || diff < -EPS) lw[maxIdx] += diff;
      return;
    }

    if (pendingGlobal > 0.0f) lw[firstHealthyGlobal] += pendingGlobal;

    {
      const float EPS = 1e-4f;
      float diff = 1.0f - sum_in;
      if (diff > EPS || diff < -EPS) lw[firstHealthyGlobal] += diff;
    }
    return;
  }

  const uint16_t* qp_card = s->qp_card;
  int firstHealthyCard[CIM_MAX_CARDS];
  int lastSeenCard[CIM_MAX_CARDS];
  float pendingCard[CIM_MAX_CARDS];

  for (int c = 0; c < nCards; c++) {
    firstHealthyCard[c] = -1;
    lastSeenCard[c] = -1;
    pendingCard[c] = 0.0f;
  }

  for (int i = n - 1; i >= 0; i--) {
    float w = lw[i];
    if (w < 0.0f) w = 0.0f;
    sum_in += w;

    if (w > maxV) {
      maxV = w;
      maxIdx = i;
    }

    unsigned c = (unsigned)qp_card[i];
    const int c_ok = (c < (unsigned)nCards);

    if (w >= min_healthy) {
      healthy_count++;
      firstHealthyGlobal = i;
      lastGlobal = i;

      if (c_ok) {
        firstHealthyCard[c] = i;
        lastSeenCard[c] = i;
      }
      continue;
    }

    if (w > 0.0f) {
      has_unhealthy_pos = 1;
      if (w < min_healthy) {
        lw[i] = 0.0f;

        int t = -1;
        if (c_ok && lastSeenCard[c] >= 0) t = lastSeenCard[c];
        else if (lastGlobal >= 0) t = lastGlobal;

        if (t >= 0) {
          lw[t] += w;
        } else {
          if (c_ok) pendingCard[c] += w;
          else pendingGlobal += w;
        }
      }
    }
  }

  if (!has_unhealthy_pos) {
    const float EPS = 1e-4f;
    float diff = 1.0f - sum_in;
    if (diff > EPS || diff < -EPS) lw[maxIdx] += diff;
    return;
  }

  if (healthy_count == 0) {
    float pending = pendingGlobal;
    for (int c = 0; c < nCards; c++) pending += pendingCard[c];
    lw[maxIdx] += pending;
    {
      const float EPS = 1e-4f;
      float diff = 1.0f - sum_in;
      if (diff > EPS || diff < -EPS) lw[maxIdx] += diff;
    }
    return;
  }

  for (int c = 0; c < nCards; c++) {
    if (pendingCard[c] == 0.0f) continue;
    if (firstHealthyCard[c] >= 0) lw[firstHealthyCard[c]] += pendingCard[c];
    else if (firstHealthyGlobal >= 0) lw[firstHealthyGlobal] += pendingCard[c];
  }
  if (pendingGlobal > 0.0f) lw[firstHealthyGlobal] += pendingGlobal;

  {
    const float EPS = 1e-4f;
    float diff = 1.0f - sum_in;
    if (diff > EPS || diff < -EPS) lw[firstHealthyGlobal] += diff;
  }
}

#endif
