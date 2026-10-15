/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/

#define ENABLE_TIMER 1           // MUST be before including timer.h
#define CIM_DEBUG 0              // set to 1 only if you want lw dumps (will add noise)

#include "cim.h"
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

static void print_cyc_stats(const char* name, const CycStat* a, const CycStat* b) {
  // 'freq' is static inside timer.h,
  // It is timer ticks per microsecond.
  double f = timer_cycles_per_us();
  if (f <= 0.0) f = 0.0;

  double a_us = (f > 0.0) ? ((double)a->cyc_sum / f) : 0.0;
  double b_us = (f > 0.0) ? ((double)b->cyc_sum / f) : 0.0;

  double a_avg = (a->cnt ? a_us / (double)a->cnt : 0.0);
  double b_avg = (b->cnt ? b_us / (double)b->cnt : 0.0);

  double a_max_us = (f > 0.0) ? ((double)a->cyc_max / f) : 0.0;
  double b_max_us = (f > 0.0) ? ((double)b->cyc_max / f) : 0.0;

  fprintf(stdout,
          "%s cyc-stats "
          "[0] %g/%lu = %g us (max %g) "
          "[1] %g/%lu = %g us (max %g)\n",
          name,
          a_us, (unsigned long)a->cnt, a_avg, a_max_us,
          b_us, (unsigned long)b->cnt, b_avg, b_max_us);
}

/* -------------------- helpers -------------------- */

static void dump_lw(const char* tag, const CIMState* s) {
  float sum = 0.0f;
  fprintf(stdout, "%s: ", tag);
  for (int i = 0; i < s->nqps; i++) {
    fprintf(stdout, "qp=%d lw=%.6f ", i, s->lw[i]);
    sum += s->lw[i];
  }
  fprintf(stdout, "| sum=%.6f\n", sum);
}

// --- tiny sleep helper ---
static void msleep(int ms) {
  struct timespec ts;
  ts.tv_sec  = ms / 1000;
  ts.tv_nsec = (long)(ms % 1000) * 1000000L;
  nanosleep(&ts, NULL);
}

/* Deterministic init */

static void init_lw_example(CIMState* s) {
  int n = s->nqps;
  if (n <= 0) return;
  for (int i = 0; i < n; i++) s->lw[i] = 0.0f;

  /* For testing and comparsions 4 QPs are helpful */
  float u = 1.0f / (float)n;
  for (int i = 0; i < n; i++) s->lw[i] = u;

}

CIMState* CIM_Start(int nqps) {
  if (nqps <= 0) return NULL;
  if (nqps > CIM_MAX_QPS) {
    nqps = CIM_MAX_QPS;
  }

  CIMState* s = (CIMState*)malloc(sizeof(CIMState));
  if (!s) return NULL;
  memset(s, 0, sizeof(*s));
  s->nqps = nqps;
  s->dbg_view_count = (uint32_t)nqps;
  for (int i = 0; i < nqps; i++) s->dbg_view_qp_index[i] = (uint32_t)i;
  s->seed = (unsigned int)time(NULL) ^ (unsigned int)(uintptr_t)pthread_self();

  // Deterministic one-time init (no background thread).
  init_lw_example(s);
  s->lw_initialized = 1;

#if CIM_DEBUG
  dump_lw("INIT", s);
#endif

  return s;
}

void CIM_Stop(CIMState* s) {
  if (s == NULL) return;
  free(s);
}

// --- global state ---
static pthread_t g_lat_th;
static pthread_mutex_t g_lat_mu = PTHREAD_MUTEX_INITIALIZER;

static int g_lat_running = 0;
static int g_lat_started = 0;
static int g_lat_interval_ms = 50;

// We store a "current source" (one comm at a time).
// Using volatile uint64_t* so you can pass either:
//   - uint64_t qp_lat[] (non-atomic) OR
//   - _Atomic uint64_t qp_lat[] (atomic)
// In either case we load with __atomic_load_n(..., RELAXED).
static volatile uint64_t* g_lat_ptr = NULL;
static volatile uint64_t* g_r_lat_ptr = NULL;
static volatile uint32_t* g_bytes_ptr = NULL;
static int g_lat_nqps = 0;
static char g_lat_label[64] = "qp_lat";
// OPTIONAL: label for raw array (handy in logs)
static char g_r_lat_label[64] = "raw_lat";
static char g_bytes_label[64] = "bytes";

// --- LW source (optional) ---
static volatile float* g_lw_ptr = NULL;
static int g_lw_nqps = 0;
static char g_lw_label[64] = "lw";
static volatile uint32_t* g_dbg_seq_ptr = NULL;
static volatile uint32_t* g_dbg_view_count_ptr = NULL;
static volatile uint32_t* g_dbg_view_qp_index_ptr = NULL;

// Load float via 32-bit atomic load to avoid torn reads (debug-safe)
static inline float load_float_relaxed(const volatile float* p) {
  uint32_t u = __atomic_load_n((const volatile uint32_t*)p, __ATOMIC_RELAXED);
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

typedef struct {
  uint64_t lat[CIM_MAX_QPS];
  uint64_t raw[CIM_MAX_QPS];
  uint32_t bytes[CIM_MAX_QPS];
  float lw[CIM_MAX_QPS];
  uint32_t view_count;
  uint32_t view_qp_index[CIM_MAX_QPS];
} CIMDebugSnapshot;

static inline int cimDbgFloatDiffers(float a, float b) {
  float d = a - b;
  if (d < 0.0f) d = -d;
  return d > 1e-6f;
}

static void cimDbgCaptureSnapshot(CIMDebugSnapshot* out,
                                  volatile uint64_t* lat,
                                  volatile uint64_t* rlat,
                                  volatile uint32_t* bytes,
                                  volatile float* lw,
                                  int lw_nqps,
                                  volatile uint32_t* seqp,
                                  volatile uint32_t* viewCountp,
                                  volatile uint32_t* viewQpIndexp,
                                  int nqps) {
  while (1) {
    uint32_t s0 = 0, s1 = 0;
    if (seqp) {
      s0 = __atomic_load_n(seqp, __ATOMIC_ACQUIRE);
      if (s0 & 1u) continue;
    }

    for (int i = 0; i < nqps; i++) {
      out->lat[i] = lat ? __atomic_load_n(&lat[i], __ATOMIC_RELAXED) : 0;
      out->raw[i] = rlat ? __atomic_load_n(&rlat[i], __ATOMIC_RELAXED) : 0;
      out->bytes[i] = bytes ? __atomic_load_n(&bytes[i], __ATOMIC_RELAXED) : 0;
      out->lw[i] = (lw && i < lw_nqps) ? load_float_relaxed(&lw[i]) : 0.0f;
    }

    uint32_t viewCount = (uint32_t)nqps;
    if (viewCountp != NULL) {
      uint32_t loaded = __atomic_load_n(viewCountp, __ATOMIC_RELAXED);
      if (loaded != 0 && loaded <= (uint32_t)nqps) viewCount = loaded;
    }
    out->view_count = viewCount;
    for (uint32_t i = 0; i < viewCount; i++) {
      uint32_t qpIndex = viewQpIndexp != NULL ? __atomic_load_n(&viewQpIndexp[i], __ATOMIC_RELAXED) : i;
      out->view_qp_index[i] = qpIndex < (uint32_t)nqps ? qpIndex : i;
    }

    if (!seqp) break;
    s1 = __atomic_load_n(seqp, __ATOMIC_ACQUIRE);
    if (s0 == s1 && ((s1 & 1u) == 0)) break;
  }
}

static int cimDbgSnapshotDiffersFromLast(const CIMDebugSnapshot* snap,
                                         const uint64_t* last,
                                         const uint64_t* last_raw,
                                         const uint32_t* last_bytes,
                                         const float* last_lw,
                                         int nqps,
                                         int hasLat,
                                         int hasRaw,
                                         int hasBytes,
                                         int hasLw) {
  for (int i = 0; i < nqps; i++) {
    if (hasLat && snap->lat[i] != last[i]) return 1;
    if (hasRaw && snap->raw[i] != last_raw[i]) return 1;
    if (hasBytes && snap->bytes[i] != last_bytes[i]) return 1;
    if (hasLw && cimDbgFloatDiffers(snap->lw[i], last_lw[i])) return 1;
  }
  return 0;
}

static int cimDbgSnapshotsDiffer(const CIMDebugSnapshot* a,
                                 const CIMDebugSnapshot* b,
                                 int nqps,
                                 int hasLat,
                                 int hasRaw,
                                 int hasBytes,
                                 int hasLw) {
  for (int i = 0; i < nqps; i++) {
    if (hasLat && a->lat[i] != b->lat[i]) return 1;
    if (hasRaw && a->raw[i] != b->raw[i]) return 1;
    if (hasBytes && a->bytes[i] != b->bytes[i]) return 1;
    if (hasLw && cimDbgFloatDiffers(a->lw[i], b->lw[i])) return 1;
  }
  return 0;
}

static void cimDbgCommitLastSnapshot(const CIMDebugSnapshot* snap,
                                     uint64_t* last,
                                     uint64_t* last_raw,
                                     uint32_t* last_bytes,
                                     float* last_lw,
                                     uint32_t* last_view_count,
                                     uint32_t* last_view_qp_index,
                                     int nqps,
                                     int hasLat,
                                     int hasRaw,
                                     int hasBytes,
                                     int hasLw) {
  for (int i = 0; i < nqps; i++) {
    if (hasLat) last[i] = snap->lat[i];
    if (hasRaw) last_raw[i] = snap->raw[i];
    if (hasBytes) last_bytes[i] = snap->bytes[i];
    if (hasLw) last_lw[i] = snap->lw[i];
  }
  *last_view_count = snap->view_count;
  for (uint32_t i = 0; i < snap->view_count; i++) {
    last_view_qp_index[i] = snap->view_qp_index[i];
  }
}

static inline int cimDbgViewCount(const CIMDebugSnapshot* snap, int nqps) {
  int viewCount = (snap->view_count == 0) ? nqps : (int)snap->view_count;
  if (viewCount < 0) viewCount = 0;
  if (viewCount > nqps) viewCount = nqps;
  if (viewCount > CIM_MAX_QPS) viewCount = CIM_MAX_QPS;
  return viewCount;
}

static inline int cimDbgViewQpIndex(const CIMDebugSnapshot* snap, int slot, int nqps) {
  const int viewCount = cimDbgViewCount(snap, nqps);
  if (slot < 0 || slot >= viewCount) return slot;
  const uint32_t qpIndex = snap->view_qp_index[slot];
  return qpIndex < (uint32_t)nqps ? (int)qpIndex : slot;
}

static void cimDbgPrintSnapshot(const char* tag,
                                const char* label,
                                const char* rlabel,
                                const char* bytesLabel,
                                const char* lwlabel,
                                const CIMDebugSnapshot* snap,
                                int nqps,
                                int hasLat,
                                int hasRaw,
                                int hasBytes,
                                int hasLw) {
  if (hasLw) {
    if (hasLat && hasRaw && hasBytes) {
      fprintf(stdout, "%s/%s/%s/%s%s%s: ",
              label, rlabel, bytesLabel, lwlabel,
              (tag && tag[0]) ? " " : "",
              (tag && tag[0]) ? tag : "");
    } else if (hasLat && hasRaw) {
      fprintf(stdout, "%s/%s/%s%s%s: ",
              label, rlabel, lwlabel,
              (tag && tag[0]) ? " " : "",
              (tag && tag[0]) ? tag : "");
    } else if (hasLat && hasBytes) {
      fprintf(stdout, "%s/%s/%s%s%s: ",
              label, bytesLabel, lwlabel,
              (tag && tag[0]) ? " " : "",
              (tag && tag[0]) ? tag : "");
    } else if (hasLat) {
      fprintf(stdout, "%s/%s%s%s: ",
              label, lwlabel,
              (tag && tag[0]) ? " " : "",
              (tag && tag[0]) ? tag : "");
    } else if (hasRaw && hasBytes) {
      fprintf(stdout, "%s/%s/%s%s%s: ",
              rlabel, bytesLabel, lwlabel,
              (tag && tag[0]) ? " " : "",
              (tag && tag[0]) ? tag : "");
    } else if (hasRaw) {
      fprintf(stdout, "%s/%s%s%s: ",
              rlabel, lwlabel,
              (tag && tag[0]) ? " " : "",
              (tag && tag[0]) ? tag : "");
    } else if (hasBytes) {
      fprintf(stdout, "%s/%s%s%s: ",
              bytesLabel, lwlabel,
              (tag && tag[0]) ? " " : "",
              (tag && tag[0]) ? tag : "");
    }
  } else {
    if (hasLat && hasRaw && hasBytes) {
      fprintf(stdout, "%s/%s/%s%s%s: ",
              label, rlabel, bytesLabel,
              (tag && tag[0]) ? " " : "",
              (tag && tag[0]) ? tag : "");
    } else if (hasLat && hasRaw) {
      fprintf(stdout, "%s/%s%s%s: ",
              label, rlabel,
              (tag && tag[0]) ? " " : "",
              (tag && tag[0]) ? tag : "");
    } else if (hasLat && hasBytes) {
      fprintf(stdout, "%s/%s%s%s: ",
              label, bytesLabel,
              (tag && tag[0]) ? " " : "",
              (tag && tag[0]) ? tag : "");
    } else if (hasLat) {
      fprintf(stdout, "%s%s%s: ",
              label,
              (tag && tag[0]) ? " " : "",
              (tag && tag[0]) ? tag : "");
    } else if (hasRaw && hasBytes) {
      fprintf(stdout, "%s/%s%s%s: ",
              rlabel, bytesLabel,
              (tag && tag[0]) ? " " : "",
              (tag && tag[0]) ? tag : "");
    } else if (hasRaw) {
      fprintf(stdout, "%s%s%s: ",
              rlabel,
              (tag && tag[0]) ? " " : "",
              (tag && tag[0]) ? tag : "");
    } else if (hasBytes) {
      fprintf(stdout, "%s%s%s: ",
              bytesLabel,
              (tag && tag[0]) ? " " : "",
              (tag && tag[0]) ? tag : "");
    }
  }

  const int viewCount = cimDbgViewCount(snap, nqps);
  for (int i = 0; i < viewCount; i++) {
    const int qpIndex = cimDbgViewQpIndex(snap, i, nqps);
    if (hasLat && hasRaw && hasBytes) {
      if (hasLw) {
        fprintf(stdout, "q%d=%lu(raw=%lu,bytes=%u,w=%.4f) ",
                qpIndex,
                (unsigned long)snap->lat[qpIndex],
                (unsigned long)snap->raw[qpIndex],
                snap->bytes[qpIndex],
                snap->lw[qpIndex]);
      } else {
        fprintf(stdout, "q%d=%lu(raw=%lu,bytes=%u) ",
                qpIndex,
                (unsigned long)snap->lat[qpIndex],
                (unsigned long)snap->raw[qpIndex],
                snap->bytes[qpIndex]);
      }
    } else if (hasLat && hasRaw) {
      if (hasLw) {
        fprintf(stdout, "q%d=%lu(raw=%lu,w=%.4f) ",
                qpIndex,
                (unsigned long)snap->lat[qpIndex],
                (unsigned long)snap->raw[qpIndex],
                snap->lw[qpIndex]);
      } else {
        fprintf(stdout, "q%d=%lu(raw=%lu) ",
                qpIndex, (unsigned long)snap->lat[qpIndex], (unsigned long)snap->raw[qpIndex]);
      }
    } else if (hasLat && hasBytes) {
      if (hasLw) {
        fprintf(stdout, "q%d=%lu(bytes=%u,w=%.4f) ",
                qpIndex, (unsigned long)snap->lat[qpIndex], snap->bytes[qpIndex], snap->lw[qpIndex]);
      } else {
        fprintf(stdout, "q%d=%lu(bytes=%u) ",
                qpIndex, (unsigned long)snap->lat[qpIndex], snap->bytes[qpIndex]);
      }
    } else if (hasLat) {
      if (hasLw) {
        fprintf(stdout, "q%d=%lu(w=%.4f) ",
                qpIndex, (unsigned long)snap->lat[qpIndex], snap->lw[qpIndex]);
      } else {
        fprintf(stdout, "q%d=%lu ",
                qpIndex, (unsigned long)snap->lat[qpIndex]);
      }
    } else if (hasRaw && hasBytes) {
      if (hasLw) {
        fprintf(stdout, "q%d=%lu(bytes=%u,w=%.4f) ",
                qpIndex, (unsigned long)snap->raw[qpIndex], snap->bytes[qpIndex], snap->lw[qpIndex]);
      } else {
        fprintf(stdout, "q%d=%lu(bytes=%u) ",
                qpIndex, (unsigned long)snap->raw[qpIndex], snap->bytes[qpIndex]);
      }
    } else if (hasRaw) {
      if (hasLw) {
        fprintf(stdout, "q%d=%lu(w=%.4f) ",
                qpIndex, (unsigned long)snap->raw[qpIndex], snap->lw[qpIndex]);
      } else {
        fprintf(stdout, "q%d=%lu ",
                qpIndex, (unsigned long)snap->raw[qpIndex]);
      }
    } else if (hasBytes) {
      if (hasLw) {
        fprintf(stdout, "q%d=%u(w=%.4f) ",
                qpIndex, snap->bytes[qpIndex], snap->lw[qpIndex]);
      } else {
        fprintf(stdout, "q%d=%u ",
                qpIndex, snap->bytes[qpIndex]);
      }
    }
  }
  fprintf(stdout, "\n");
}

// -----------------------------------------------------------
// In lat_printer_thread(): add tracking for raw array
// -----------------------------------------------------------
static void* lat_printer_thread(void* _) {
  (void)_;
  uint64_t last[CIM_MAX_QPS];
  memset(last, 0, sizeof(last));
  uint64_t last_raw[CIM_MAX_QPS];
  memset(last_raw, 0, sizeof(last_raw));
  uint32_t last_bytes[CIM_MAX_QPS];
  memset(last_bytes, 0, sizeof(last_bytes));

  float last_lw[CIM_MAX_QPS];
  for (int i = 0; i < CIM_MAX_QPS; i++) last_lw[i] = -1.0f;
  uint32_t last_view_count = 0;
  uint32_t last_view_qp_index[CIM_MAX_QPS];
  memset(last_view_qp_index, 0, sizeof(last_view_qp_index));
  int haveLast = 0;

  while (1) {
    int running = 0;
    pthread_mutex_lock(&g_lat_mu);
    running = g_lat_running;
    pthread_mutex_unlock(&g_lat_mu);
    if (!running) break;

    volatile uint64_t* lat = NULL;
    volatile uint64_t* rlat = NULL;
    volatile uint32_t* bytes = NULL;
    int nqps = 0;
    char label[64];
    char rlabel[64];
    char bytesLabel[64];

    volatile float* lw = NULL;
    int lw_nqps = 0;
    char lwlabel[64];
    volatile uint32_t* seqp = NULL;
    volatile uint32_t* viewCountp = NULL;
    volatile uint32_t* viewQpIndexp = NULL;

    pthread_mutex_lock(&g_lat_mu);
    lat  = g_lat_ptr;
    rlat = g_r_lat_ptr;
    bytes = g_bytes_ptr;
    nqps = g_lat_nqps;

    strncpy(label, g_lat_label, sizeof(label));
    label[sizeof(label)-1] = 0;

    strncpy(rlabel, g_r_lat_label, sizeof(rlabel));
    rlabel[sizeof(rlabel)-1] = 0;

    strncpy(bytesLabel, g_bytes_label, sizeof(bytesLabel));
    bytesLabel[sizeof(bytesLabel)-1] = 0;

    lw = g_lw_ptr;
    lw_nqps = g_lw_nqps;
    seqp = g_dbg_seq_ptr;
    viewCountp = g_dbg_view_count_ptr;
    viewQpIndexp = g_dbg_view_qp_index_ptr;

    strncpy(lwlabel, g_lw_label, sizeof(lwlabel));
    lwlabel[sizeof(lwlabel)-1] = 0;
    pthread_mutex_unlock(&g_lat_mu);

    // If neither array is set, nothing to do.
    if ((lat == NULL && rlat == NULL && bytes == NULL) || nqps <= 0) {
      msleep(g_lat_interval_ms);
      continue;
    }

    if (nqps > CIM_MAX_QPS) nqps = CIM_MAX_QPS;

    const int hasLat = (lat != NULL);
    const int hasRaw = (rlat != NULL);
    const int hasBytes = (bytes != NULL);
    const int hasLw = (lw != NULL && lw_nqps > 0);

    CIMDebugSnapshot snap;
    cimDbgCaptureSnapshot(&snap, lat, rlat, bytes, lw, lw_nqps, seqp, viewCountp, viewQpIndexp, nqps);

    if (!haveLast) {
      cimDbgCommitLastSnapshot(&snap, last, last_raw, last_bytes, last_lw,
                               &last_view_count, last_view_qp_index,
                               nqps, hasLat, hasRaw, hasBytes, hasLw);
      haveLast = 1;
      msleep(g_lat_interval_ms);
      continue;
    }

    {
      CIMDebugSnapshot prev = {};
      for (int i = 0; i < nqps; i++) {
        prev.lat[i] = last[i];
        prev.raw[i] = last_raw[i];
        prev.bytes[i] = last_bytes[i];
        prev.lw[i] = last_lw[i];
      }
      prev.view_count = last_view_count;
      for (uint32_t i = 0; i < last_view_count; i++) {
        prev.view_qp_index[i] = last_view_qp_index[i];
      }

      cimDbgPrintSnapshot("[0]", label, rlabel, bytesLabel, lwlabel, &prev,
                          nqps, hasLat, hasRaw, hasBytes, hasLw);
      cimDbgPrintSnapshot("[1]", label, rlabel, bytesLabel, lwlabel, &snap,
                          nqps, hasLat, hasRaw, hasBytes, hasLw);
      cimDbgCommitLastSnapshot(&snap, last, last_raw, last_bytes, last_lw,
                               &last_view_count, last_view_qp_index,
                               nqps, hasLat, hasRaw, hasBytes, hasLw);
      fflush(stdout);
    }

    msleep(g_lat_interval_ms);
  }

  return NULL;
}

// --- API ---
void CIM_DebugLatPrinterStart(int interval_ms) {
  pthread_mutex_lock(&g_lat_mu);
  if (g_lat_started) {
    // Update interval only
    if (interval_ms > 0) g_lat_interval_ms = interval_ms;
    pthread_mutex_unlock(&g_lat_mu);
    return;
  }

  if (interval_ms > 0) g_lat_interval_ms = interval_ms;
  g_lat_running = 1;
  g_lat_started = 1;

  pthread_mutex_unlock(&g_lat_mu);

  // Start thread
  if (pthread_create(&g_lat_th, NULL, lat_printer_thread, NULL) != 0) {
    pthread_mutex_lock(&g_lat_mu);
    g_lat_running = 0;
    g_lat_started = 0;
    pthread_mutex_unlock(&g_lat_mu);
  }
}


// -----------------------------------------------------------
// In Stop(): clear raw pointer + label (optional)
// -----------------------------------------------------------
void CIM_DebugLatPrinterStop(void) {
  pthread_mutex_lock(&g_lat_mu);
  if (!g_lat_started) {
    pthread_mutex_unlock(&g_lat_mu);
    return;
  }
  g_lat_running = 0;
  pthread_mutex_unlock(&g_lat_mu);

  pthread_join(g_lat_th, NULL);

  pthread_mutex_lock(&g_lat_mu);
  g_lat_started = 0;
  g_lat_ptr = NULL;
  g_r_lat_ptr = NULL;     // NEW
  g_bytes_ptr = NULL;
  g_lat_nqps = 0;
  g_lw_ptr = NULL;
  g_lw_nqps = 0;
  g_dbg_seq_ptr = NULL;
  g_dbg_view_count_ptr = NULL;
  g_dbg_view_qp_index_ptr = NULL;
  pthread_mutex_unlock(&g_lat_mu);
}

// -----------------------------------------------------------
// In SetSource(): actually store g_r_lat_ptr + optional raw label
// (keeps your existing signature; uses a default raw label)
// -----------------------------------------------------------
void CIM_DebugLatPrinterSetSource(const char* label,
                                 volatile uint64_t* lat,
                                 volatile uint64_t* r_lat,
                                 int nqps) {
  pthread_mutex_lock(&g_lat_mu);

  g_lat_ptr     = lat;
  g_r_lat_ptr   = r_lat;   // NEW
  g_lat_nqps    = nqps;

  if (label && label[0]) {
    strncpy(g_lat_label, label, sizeof(g_lat_label));
    g_lat_label[sizeof(g_lat_label)-1] = 0;
  } else {
    strncpy(g_lat_label, "qp_lat", sizeof(g_lat_label));
    g_lat_label[sizeof(g_lat_label)-1] = 0;
  }

  // Optional: set default raw label based on provided label
  // e.g. label="qp_lat" -> raw="qp_lat_raw"
  if (label && label[0]) {
    snprintf(g_r_lat_label, sizeof(g_r_lat_label), "%s_raw", label);
  } else {
    strncpy(g_r_lat_label, "raw_lat", sizeof(g_r_lat_label));
    g_r_lat_label[sizeof(g_r_lat_label)-1] = 0;
  }

  pthread_mutex_unlock(&g_lat_mu);
}

void CIM_DebugLatPrinterSetWeightSource(const char* label,
                                       volatile float* lw,
                                       int nqps) {
  pthread_mutex_lock(&g_lat_mu);

  g_lw_ptr  = lw;
  g_lw_nqps = nqps;

  if (label && label[0]) {
    strncpy(g_lw_label, label, sizeof(g_lw_label));
    g_lw_label[sizeof(g_lw_label)-1] = 0;
  } else {
    strncpy(g_lw_label, "lw", sizeof(g_lw_label));
    g_lw_label[sizeof(g_lw_label)-1] = 0;
  }

  pthread_mutex_unlock(&g_lat_mu);
}

void CIM_DebugLatPrinterSetByteSource(const char* label,
                                      volatile uint32_t* bytes,
                                      int nqps) {
  pthread_mutex_lock(&g_lat_mu);

  g_bytes_ptr = bytes;
  if (nqps > 0) g_lat_nqps = nqps;

  if (label && label[0]) {
    strncpy(g_bytes_label, label, sizeof(g_bytes_label));
    g_bytes_label[sizeof(g_bytes_label)-1] = 0;
  } else {
    strncpy(g_bytes_label, "bytes", sizeof(g_bytes_label));
    g_bytes_label[sizeof(g_bytes_label)-1] = 0;
  }

  pthread_mutex_unlock(&g_lat_mu);
}

void CIM_DebugLatPrinterSetSyncSeq(volatile uint32_t* seq) {
  pthread_mutex_lock(&g_lat_mu);
  g_dbg_seq_ptr = seq;
  pthread_mutex_unlock(&g_lat_mu);
}

void CIM_DebugLatPrinterSetViewSource(volatile uint32_t* viewCount,
                                      volatile uint32_t* viewQpIndex) {
  pthread_mutex_lock(&g_lat_mu);
  g_dbg_view_count_ptr = viewCount;
  g_dbg_view_qp_index_ptr = viewQpIndex;
  pthread_mutex_unlock(&g_lat_mu);
}
