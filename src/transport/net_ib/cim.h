#ifndef _CIM_H_
#define _CIM_H_

#include <stdint.h>
#include <stdio.h>
#include <pthread.h>
#include "timer.h"               // NVIDIA provided timer.h

#define CIM_MAX_QPS 128   // reasonable upper bound for now
//keeping it same as  NCCL_IB_MAX_QPS 128

// Opaque state for one communicator's CIM instance
typedef struct CIMState CIMState;

struct CIMState {
    int      nqps;                  // number of QPs for this comm
    uint64_t spread_eps_us;         // cached FRM equal-spread deadband for this comm
    int      running;               // thread loop flag
    pthread_t thread;
    volatile uint32_t dbg_seq;      // seqlock for coherent debug snapshots
    uint32_t dbg_view_count;        // number of debug entries to render
    uint32_t dbg_view_qp_index[CIM_MAX_QPS]; // debug slot -> global QP index
    float    lw[CIM_MAX_QPS];       // per-QP load weight (0..1)
    uint64_t latency[CIM_MAX_QPS];  // Cache latency in cim
    uint64_t raw_latency[CIM_MAX_QPS]; // Cache raw latency before norm
    uint32_t lat_bytes[CIM_MAX_QPS]; // Bytes used for the most recent published stamp
    unsigned int seed; // RNG seed per thread
    int lw_initialized;   // 0 = not initialized, 1 = initialized
    uint8_t split0_mild_score[CIM_MAX_QPS];   // split=0 repeated weak-spread score per active window base
    uint8_t split0_mild_slow_qp[CIM_MAX_QPS]; // split=0 last repeated slower global QP per active window base
};

/**
 * Start CIM congestion-score / load-weight updater for a communicator.
 *
 * nqps: number of QPs for this communicator (clamped to CIM_MAX_QPS)
 *
 * Returns: pointer to CIMState, or NULL if allocation / start failed.
 */
CIMState* CIM_Start(int nqps);
void CIM_Stop(CIMState* s);

void CIM_DebugLatPrinterStart(int interval_ms);
void CIM_DebugLatPrinterStop(void);
void CIM_DebugLatPrinterSetSource(const char* label,
                                 volatile uint64_t* lat,
                                 volatile uint64_t* r_lat,
                                 int nqps);
void CIM_DebugLatPrinterSetWeightSource(const char* label,
                                        volatile float* lw,
                                        int nqps);
void CIM_DebugLatPrinterSetByteSource(const char* label,
                                      volatile uint32_t* bytes,
                                      int nqps);
void CIM_DebugLatPrinterSetSyncSeq(volatile uint32_t* seq);
void CIM_DebugLatPrinterSetViewSource(volatile uint32_t* viewCount,
                                      volatile uint32_t* viewQpIndex);

/* -------------------- Option B timer: cycle accumulation -------------------- */

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

static inline float CIM_chooseMinHealthy_fromWeights(const float* lw, int n, int* maxIdxOut) {
  float wmax = lw[0];
  int maxIdx = 0;
  for (int i = 1; i < n; i++) {
    if (lw[i] > wmax) {
      wmax = lw[i];
      maxIdx = i;
    }
  }
  if (maxIdxOut) *maxIdxOut = maxIdx;

  // Gate anything below ~20% of best path
  const float gateFrac = 0.20f;

  const float absMin = 0.03f;   // never allow very tiny weights
  const float absMax = 0.30f;   // cap threshold

  float mh = gateFrac * wmax;
  return clampf(mh, absMin, absMax);
}

// gate_move_and_normalize()
// Purpose:
//   Gate (zero out) "unhealthy" positive weights < min_healthy, and move their mass
//   forward in ring order onto a later healthy QP, preserving total sum ~= 1.
//
// Inputs:
//   lw          : in/out array of length n, load-weights per QP (assumed >=0, sum~1).
//   n           : number of QPs in this communicator.
//   min_healthy : threshold. Any 0 < lw[i] < min_healthy is gated (set to 0) and its
//                 weight is moved to a later healthy QP.
//
// Notes on behavior:
//   - "Healthy" means lw[i] >= min_healthy.
//   - If we encounter gated mass before we've seen any healthy target (because we're
//     scanning backwards), we store it in "pending" buckets and add it to the first
//     healthy QP once discovered.
//   - If there are *no* healthy QPs at all, we dump all mass into maxIdx.
//
// EPS sum-fix:
//   Floating arithmetic + clamping can cause sum(lw) to drift slightly from 1.0.
//   We compute sum_in during the scan and apply a tiny correction (diff = 1 - sum_in)
//   to one stable target index if |diff| > EPS. This keeps invariants stable without
//   an extra normalization pass.
//

static inline void gate_move_and_normalize(float* __restrict lw, int n) {
  if (!lw || n <= 1) return;

  int maxIdx = 0;
  const float min_healthy = CIM_chooseMinHealthy_fromWeights(lw, n, &maxIdx);

  // Track the original sum in one backward pass so the final sum-fix stays O(1).
  float sum_in = 0.0f;
  int firstHealthy = -1;   // smallest healthy index after backward scan
  int lastHealthy = -1;    // next healthy target in ring order while scanning backward
  int gatedAny = 0;
  float pending = 0.0f;

  for (int i = n - 1; i >= 0; i--) {
    const float w = lw[i];
    sum_in += w;

    if (w >= min_healthy) {
      firstHealthy = i;
      lastHealthy = i;
    } else if (w > 0.0f) {
      gatedAny = 1;
      lw[i] = 0.0f;
      if (lastHealthy >= 0) {
        lw[lastHealthy] += w;
      } else {
        pending += w;
      }
    }
  }

  int fixIdx = maxIdx;
  if (gatedAny) {
    if (firstHealthy >= 0) {
      if (pending > 0.0f) lw[firstHealthy] += pending;
      fixIdx = firstHealthy;
    } else {
      lw[maxIdx] += pending;
    }
  }

  const float diff = 1.0f - sum_in;
  if (diff > 1e-4f || diff < -1e-4f) lw[fixIdx] += diff;
}

static inline float clamp01(float x) {
  return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

/*
 * ============================================================
 * Bucket A: ratio <= 2  → headroomMult = 2.0
 * Latencies: [60, 70, 80, 100]
 *
 * min = 60, max = 100
 * ratio = 100 / 60 = 1.67  → headroom = 2.0
 *
 * range  = max - min = 40
 * effMax = min + 2.0 * range = 60 + 80 = 140
 *
 * Lf = (lat - min) / (effMax - min)
 *
 *   60  → 0 / 80  = 0.000
 *   70  → 10/80  = 0.125
 *   80  → 20/80  = 0.250
 *   100 → 40/80  = 0.500
 *
 * cf = 1 - Lf
 *
 *   [1.000, 0.875, 0.750, 0.500]
 *   sum = 3.125
 *
 * Final load weights:
 *
 *   QP0 (60)  → 1.000 / 3.125 = 0.320
 *   QP1 (70)  → 0.875 / 3.125 = 0.280
 *   QP2 (80)  → 0.750 / 3.125 = 0.240
 *   QP3 (100) → 0.500 / 3.125 = 0.160
 *
 * Interpretation:
 *   - All QPs get traffic
 *   - Differences are smooth
 *   - No QP is gated
 * ============================================================
 */

/*
 * ============================================================
 * Bucket B: 2 < ratio <= 5  → headroomMult = 1.5
 * Latencies: [60, 65, 70, 200]
 *
 * min = 60, max = 200
 * ratio = 200 / 60 = 3.33  → headroom = 1.5
 *
 * range  = max - min = 140
 * effMax = min + 1.5 * range = 60 + 210 = 270
 *
 * Lf = (lat - min) / (effMax - min)
 *
 *   60  → 0 / 210  = 0.000
 *   65  → 5 / 210  = 0.024
 *   70  → 10/210  = 0.048
 *   200 → 140/210 = 0.667
 *
 * cf = 1 - Lf
 *
 *   [1.000, 0.976, 0.952, 0.333]
 *   sum = 3.261
 *
 * Final load weights:
 *
 *   QP0 (60)  → 1.000 / 3.261 = 0.307
 *   QP1 (65)  → 0.976 / 3.261 = 0.299
 *   QP2 (70)  → 0.952 / 3.261 = 0.292
 *   QP3 (200) → 0.333 / 3.261 = 0.102
 *
 * Interpretation:
 *   - Slow QP is penalized but not killed
 *   - Still receives ~10% traffic
 *   - Allows recovery and measurement
 * ============================================================
 */

/*
 * ============================================================
 * Bucket C: ratio > 5  → headroomMult = 1.0
 * Latencies: [60, 65, 70, 600]
 *
 * min = 60, max = 600
 * ratio = 600 / 60 = 10  → headroom = 1.0
 *
 * range  = max - min = 540
 * effMax = min + 1.0 * range = 600
 *
 * Lf = (lat - min) / (effMax - min)
 *
 *   60  → 0 / 540  = 0.000
 *   65  → 5 / 540  = 0.009
 *   70  → 10/540  = 0.019
 *   600 → 540/540 = 1.000
 *
 * cf = 1 - Lf
 *
 *   [1.000, 0.991, 0.981, 0.000]
 *   sum = 2.972
 *
 * Final load weights:
 *
 *   QP0 (60)  → 1.000 / 2.972 = 0.336
 *   QP1 (65)  → 0.991 / 2.972 = 0.333
 *   QP2 (70)  → 0.981 / 2.972 = 0.330
 *   QP3 (600) → 0.000 / 2.972 = 0.000
 *
 * Interpretation:
 *   - Extremely slow QP is gated
 *   - Fast QPs evenly share traffic
 *   - Prevents tail latency
 * ============================================================
 */

/*
 * Mix example: [60, 70, 120, 200, 980, 1050]
 * min=60, max=1050, range=990, ratio=17.5 -> headroomMult=1.0
 * effMax = 60 + 1.0*990 = 1050, denom=990
 *
 * Lf = (lat-60)/990:
 *   60:   0/990=0.0000  -> cf=1.0000
 *   70:  10/990=0.0101  -> cf=0.9899
 *  120:  60/990=0.0606  -> cf=0.9394
 *  200: 140/990=0.1414  -> cf=0.8586
 *  980: 920/990=0.9293  -> cf=0.0707
 * 1050: 990/990=1.0000  -> cf=0.0000
 *
 * sum(cf)=3.8586
 *
 * weights:
 *   60  -> 0.2592
 *   70  -> 0.2565
 *   120 -> 0.2435
 *   200 -> 0.2225
 *   980 -> 0.0183
 *   1050-> 0.0000
 */

// Headroom uses only three values {1.0, 1.5, 2.0}. Keep the hot-path choice in
// integer space so we avoid a floating-point divide just to select a bucket.
// Returned value is scaled by 2 to represent 1.0/1.5/2.0 as 2/3/4.
static inline uint32_t CIM_chooseHeadroomScale2(uint64_t minLat, uint64_t maxLat) {
  if (minLat == 0) {
    return 2u;  // cold start / safety
  } else if (maxLat <= minLat * 2u) {
    return 4u;
  } else if (maxLat <= minLat * 5u) {
    return 3u;
  } else {
    return 2u;
  }
}

static inline float CIM_chooseHeadroom(uint64_t minLat, uint64_t maxLat) {
  return 0.5f * (float)CIM_chooseHeadroomScale2(minLat, maxLat);
}

// lat[] is per-QP latency (0 means "unknown/cold")
static inline void CIM_computeLatencyFactors(const uint64_t* lat, int nqps, float* Lf) {

  uint64_t minLat = UINT64_MAX;
  uint64_t maxLat = 0;
  // Find min/max among valid entries
  for (int i = 0; i < nqps; i++) {
    const uint64_t v = lat[i];
    if (v == 0) {
      minLat = 0;
    } else {
      if (v < minLat) minLat = v;
      if (v > maxLat) maxLat = v;
    }
  }

  // No valid data
  if (minLat == UINT64_MAX) {
    for (int i = 0; i < nqps; i++) Lf[i] = 0.f;
    return;
  }

  // All equal (or only one valid)
  if (maxLat <= minLat) {
    for (int i = 0; i < nqps; i++) Lf[i] = 0.f;
    return;
  }

  const float headroomMult = 0.5f * (float)CIM_chooseHeadroomScale2(minLat, maxLat);
#if CIM_DEBUG
  fprintf(stdout,"Headroom chosen is %0.2f", headroomMult);
  fflush(stdout);
#endif

  const float denom = headroomMult * (float)(maxLat - minLat);
  if (denom <= 0.f) {
    for (int i = 0; i < nqps; i++) Lf[i] = 0.f;
    return;
  }

  for (int i = 0; i < nqps; i++) {
    uint64_t v = lat[i];
    if (v == 0) {
      Lf[i] = 0.f; // treat unknown as "not congested" for now
    } else {
      const float delta = (float)(v - minLat);
      Lf[i] = clamp01(delta / denom);
#if CIM_DEBUG
      fprintf(stdout,"latency factor for qpIndex=%d is =%0.4f \n", i, Lf[i]);
      fflush(stdout);
#endif
    }
  }
}

static inline void CIM_computeActiveWindowWeights(const uint64_t* lat,
                                                  int nqps,
                                                  int qpIndexBase,
                                                  int nDataQps,
                                                  float* weights) {
  if (!weights || nqps <= 0 || nDataQps <= 0) return;
  if (nDataQps > nqps) nDataQps = nqps;

  // In split=0, weights[] is LOCAL to the active window.
  // weights[0] always corresponds to global qpIndexBase.
  const int firstQ = qpIndexBase % nqps;
  uint64_t minLat = lat[firstQ];
  uint64_t maxLat = lat[firstQ];
  int q = firstQ;
  for (int i = 1; i < nDataQps; i++) {
    if (++q == nqps) q = 0;
    const uint64_t v = lat[q];
    if (v == 0) {
      minLat = 0;
    } else if (v < minLat) {
      minLat = v;
    } else if (v > maxLat) {
      maxLat = v;
    }
  }

  if (maxLat <= minLat) {
    const float w = 1.f / (float)nDataQps;
    for (int i = 0; i < nDataQps; i++) weights[i] = w;
    return;
  }

  const float headroomMult = 0.5f * (float)CIM_chooseHeadroomScale2(minLat, maxLat);
#if CIM_DEBUG
  fprintf(stdout, "Headroom chosen is %0.2f", headroomMult);
  fflush(stdout);
#endif
  const float denom = headroomMult * (float)(maxLat - minLat);

  float sum = 0.f;
  q = firstQ;
  for (int i = 0; i < nDataQps; i++) {
    const uint64_t v = lat[q];
    // Use the scaled direct form g' = denom - delta to avoid a per-QP divide.
    // Normalization removes the common denom scale afterward.
    float g = denom;
    if (v != 0) {
      const float delta = (float)(v - minLat);
#if CIM_DEBUG
      const float lf = clamp01(delta / denom);
      fprintf(stdout, "latency factor for active qp=%d global_qp=%d is =%0.4f \n", i, q, lf);
      fflush(stdout);
#endif
      g -= delta;
      if (g < 0.f) g = 0.f;
    }
    weights[i] = g;
    sum += g;
    if (++q == nqps) q = 0;
  }

  const float inv = 1.f / sum;
  for (int i = 0; i < nDataQps; i++) weights[i] *= inv;
}

// Convert per-QP latency samples into normalized weights for the full QP set.
//
// Worked examples:
//
// 1) Weak spread -> equal weights
//    lat = [60, 61, 60, 61]
//    minLat = 60, maxLat = 61, spread = 1 <= 2
//    => signal is treated as too weak to steer
//    => weights = [0.25, 0.25, 0.25, 0.25], *is_equal = 1
//
// 2) One slower QP, but still active
//    lat = [60, 65, 70, 200]
//    minLat = 60, maxLat = 200
//    ratio = 200 / 60 ~= 3.33 => headroomMult = 1.5
//    denom = 1.5 * (200 - 60) = 210
//    g = [210, 205, 200, 70]
//    sum(g) = 685
//    weights = g / sum(g)
//            ~= [0.30657, 0.29927, 0.29197, 0.10219]
//    => the worst QP is penalized, but not drained to zero
//
// 3) Cold/unknown QP remains eligible
//    lat = [0, 80, 100, 120]
//    minLat = 0, maxLat = 120
//    minLat == 0 => headroomMult = 1.0
//    denom = 1.0 * (120 - 0) = 120
//    v == 0 is treated as "cold but eligible", so g = denom
//    g = [120, 40, 20, 0]
//    sum(g) = 180
//    weights ~= [0.66667, 0.22222, 0.11111, 0.0]
//    => this function does not starve a cold QP before it gets measured
//
// Note: this function only produces the raw normalized weights. Later policy
// stages may still gate tiny positive weights and redistribute them.
static inline void CIM_latencyToWeightsDirect(const uint64_t* lat, int nqps,
                                              float* weights, int *is_equal) {
  if (nqps <= 0) return;

  uint64_t minLat = lat[0];
  uint64_t maxLat = lat[0];

  // Find min/max among valid entries
  for (int i = 1; i < nqps; i++) {
    const uint64_t v = lat[i];
    if (v == 0) {
      minLat = 0;
    } else if (v < minLat) {
      minLat = v;
    } else if (v > maxLat) {
      maxLat = v;
    }
  }

  if (maxLat <= minLat) {
    float w = 1.0f / (float)nqps;
    for (int i = 0; i < nqps; i++) weights[i] = w;
    *is_equal = 1;
    return;
  }

  const float headroomMult = 0.5f * (float)CIM_chooseHeadroomScale2(minLat, maxLat);

#if CIM_DEBUG
  fprintf(stdout, "Headroom chosen is %0.2f\n", headroomMult);
  fflush(stdout);
#endif

  // denom = headroomMult * (maxLat - minLat)
  // Original code:
  //   Lf = (lat - minLat) / denom
  //   g  = 1 - Lf
  //   w  = g / sum(g)
  //
  // Equivalent direct form:
  //   g' = denom - (lat - minLat)
  // since dividing each g' by denom cancels during normalization.

  float denom = headroomMult * (float)(maxLat - minLat);
  if (denom <= 0.0f) {
    float w = 1.0f / (float)nqps;
    for (int i = 0; i < nqps; i++) weights[i] = w;
    return;
  }

  float sum = 0.0f;
  for (int i = 0; i < nqps; i++) {
    uint64_t v = lat[i];
    float g;

    if (v == 0) {
      // Same as old path: Lf=0 => goodness=1
      // To stay proportional with other terms before normalization,
      // use g = denom (instead of 1.0f).
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
  }

#if CIM_DEBUG
  for (int i = 0; i < nqps; i++) {
    fprintf(stdout, "direct lw for qpIndex=%d = %.6f with latency=%lu\n",
            i, weights[i], (unsigned long)lat[i]);
  }
  fflush(stdout);
#endif
}

#endif /* _CIM_H_ */
