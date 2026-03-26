/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 OCCL-net. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#define ENABLE_TIMER 1
#include "timer.h"

#include "cim.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CIM_DEBUG 0

#if CIM_DEBUG
static void dump_lw(const char* tag, const CIMState* s) {
  float sum = 0.0f;
  fprintf(stdout, "%s: ", tag);
  for (int i = 0; i < s->nqps; i++) {
    fprintf(stdout, "qp=%d qp_card=%d lw=%.6f ", i, s->qp_card[i], s->lw[i]);
    sum += s->lw[i];
  }
  fprintf(stdout, "| sum=%.6f\n", sum);
}
#endif

static void msleep(int ms) {
  struct timespec ts;
  ts.tv_sec = ms / 1000;
  ts.tv_nsec = (long)(ms % 1000) * 1000000L;
  nanosleep(&ts, NULL);
}

static void init_lw_example(CIMState* s) {
  int n = s->nqps;
  if (n <= 0) return;
  for (int i = 0; i < n; i++) s->lw[i] = 0.0f;

  float u = 1.0f / (float)n;
  for (int i = 0; i < n; i++) s->lw[i] = u;
}

CIMState* CIM_Start(int nqps) {
  if (nqps <= 0) return NULL;
  if (nqps > CIM_MAX_QPS) nqps = CIM_MAX_QPS;

  CIMState* s = (CIMState*)malloc(sizeof(CIMState));
  if (s == NULL) return NULL;
  memset(s, 0, sizeof(*s));
  s->nqps = nqps;
  s->seed = (unsigned int)time(NULL) ^ (unsigned int)(uintptr_t)pthread_self();
  s->qp_card = (uint16_t*)malloc(sizeof(uint16_t) * nqps);
  if (s->qp_card == NULL) {
    free(s);
    return NULL;
  }

  init_lw_example(s);
  s->lw_initialized = 1;
  s->card_map_valid = 0;
  s->n_cards = 0;

  for (int q = 0; q < nqps; q++) {
    s->qp_card[q] = 0;
  }

#if CIM_DEBUG
  dump_lw("INIT", s);
#endif

  return s;
}

void CIM_Stop(CIMState* s) {
  if (s == NULL) return;
  free(s->qp_card);
  s->qp_card = NULL;
  free(s);
}

void CIM_SetQpCardMap(CIMState* s, const uint16_t* map, int n, uint16_t n_cards) {
  if (s == NULL || map == NULL) return;
  if (n > s->nqps) n = s->nqps;
  for (int i = 0; i < n; i++) s->qp_card[i] = map[i];
  s->n_cards = n_cards;
  s->card_map_valid = 1;
}

static pthread_t g_lat_th;
static pthread_mutex_t g_lat_mu = PTHREAD_MUTEX_INITIALIZER;

static int g_lat_running = 0;
static int g_lat_started = 0;
static int g_lat_interval_ms = 50;
static volatile uint64_t* g_lat_ptr = NULL;
static volatile uint64_t* g_r_lat_ptr = NULL;
static int g_lat_nqps = 0;
static char g_lat_label[64] = "qp_lat";
static char g_r_lat_label[64] = "raw_lat";

static volatile float* g_lw_ptr = NULL;
static int g_lw_nqps = 0;
static char g_lw_label[64] = "lw";
static volatile uint32_t* g_dbg_seq_ptr = NULL;

static inline float load_float_relaxed(const volatile float* p) {
  uint32_t u = __atomic_load_n((const volatile uint32_t*)p, __ATOMIC_RELAXED);
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

static void* lat_printer_thread(void* _) {
  (void)_;
  uint64_t last[CIM_MAX_QPS];
  memset(last, 0, sizeof(last));
  uint64_t last_raw[CIM_MAX_QPS];
  memset(last_raw, 0, sizeof(last_raw));

  float last_lw[CIM_MAX_QPS];
  for (int i = 0; i < CIM_MAX_QPS; i++) last_lw[i] = -1.0f;

  while (1) {
    int running = 0;
    pthread_mutex_lock(&g_lat_mu);
    running = g_lat_running;
    pthread_mutex_unlock(&g_lat_mu);
    if (!running) break;

    volatile uint64_t* lat = NULL;
    volatile uint64_t* rlat = NULL;
    int nqps = 0;
    char label[64];
    char rlabel[64];

    volatile float* lw = NULL;
    int lw_nqps = 0;
    char lwlabel[64];
    volatile uint32_t* seqp = NULL;

    pthread_mutex_lock(&g_lat_mu);
    lat = g_lat_ptr;
    rlat = g_r_lat_ptr;
    nqps = g_lat_nqps;

    strncpy(label, g_lat_label, sizeof(label));
    label[sizeof(label) - 1] = 0;

    strncpy(rlabel, g_r_lat_label, sizeof(rlabel));
    rlabel[sizeof(rlabel) - 1] = 0;

    lw = g_lw_ptr;
    lw_nqps = g_lw_nqps;
    seqp = g_dbg_seq_ptr;

    strncpy(lwlabel, g_lw_label, sizeof(lwlabel));
    lwlabel[sizeof(lwlabel) - 1] = 0;
    pthread_mutex_unlock(&g_lat_mu);

    if ((lat == NULL && rlat == NULL) || nqps <= 0) {
      msleep(g_lat_interval_ms);
      continue;
    }

    if (nqps > CIM_MAX_QPS) nqps = CIM_MAX_QPS;

    uint64_t snap[CIM_MAX_QPS];
    uint64_t rsnap[CIM_MAX_QPS];
    int changed = 0;
    while (1) {
      changed = 0;
      uint32_t s0 = 0;
      uint32_t s1 = 0;
      if (seqp != NULL) {
        s0 = __atomic_load_n(seqp, __ATOMIC_ACQUIRE);
        if (s0 & 1u) continue;
      }

      for (int i = 0; i < nqps; i++) {
        if (lat != NULL) {
          snap[i] = __atomic_load_n(&lat[i], __ATOMIC_RELAXED);
          if (snap[i] != last[i]) changed = 1;
        } else {
          snap[i] = 0;
        }

        if (rlat != NULL) {
          rsnap[i] = __atomic_load_n(&rlat[i], __ATOMIC_RELAXED);
          if (rsnap[i] != last_raw[i]) changed = 1;
        } else {
          rsnap[i] = 0;
        }

        if (lw != NULL && lw_nqps > 0) {
          float wv = load_float_relaxed(&lw[i]);
          float d = wv - last_lw[i];
          if (d < 0.0f) d = -d;
          if (d > 1e-6f) changed = 1;
        }
      }

      if (seqp == NULL) break;
      s1 = __atomic_load_n(seqp, __ATOMIC_ACQUIRE);
      if (s0 == s1 && ((s1 & 1u) == 0)) break;
    }

    if (changed) {
      if (lw != NULL && lw_nqps > 0) {
        if (lat != NULL && rlat != NULL) {
          fprintf(stdout, "%s/%s/%s: ", label, rlabel, lwlabel);
        } else if (lat != NULL) {
          fprintf(stdout, "%s/%s: ", label, lwlabel);
        } else {
          fprintf(stdout, "%s/%s: ", rlabel, lwlabel);
        }
      } else {
        if (lat != NULL && rlat != NULL) {
          fprintf(stdout, "%s/%s: ", label, rlabel);
        } else if (lat != NULL) {
          fprintf(stdout, "%s: ", label);
        } else {
          fprintf(stdout, "%s: ", rlabel);
        }
      }

      for (int i = 0; i < nqps; i++) {
        float wv = 0.0f;
        if (lw != NULL && lw_nqps > 0) wv = load_float_relaxed(&lw[i]);

        if (lat != NULL && rlat != NULL) {
          if (lw != NULL && lw_nqps > 0) {
            fprintf(stdout, "q%d=%lu(raw=%lu,w=%.4f) ", i, (unsigned long)snap[i], (unsigned long)rsnap[i], wv);
            last_lw[i] = wv;
          } else {
            fprintf(stdout, "q%d=%lu(raw=%lu) ", i, (unsigned long)snap[i], (unsigned long)rsnap[i]);
          }
          last[i] = snap[i];
          last_raw[i] = rsnap[i];
        } else if (lat != NULL) {
          if (lw != NULL && lw_nqps > 0) {
            fprintf(stdout, "q%d=%lu(w=%.4f) ", i, (unsigned long)snap[i], wv);
            last_lw[i] = wv;
          } else {
            fprintf(stdout, "q%d=%lu ", i, (unsigned long)snap[i]);
          }
          last[i] = snap[i];
        } else {
          if (lw != NULL && lw_nqps > 0) {
            fprintf(stdout, "q%d=%lu(w=%.4f) ", i, (unsigned long)rsnap[i], wv);
            last_lw[i] = wv;
          } else {
            fprintf(stdout, "q%d=%lu ", i, (unsigned long)rsnap[i]);
          }
          last_raw[i] = rsnap[i];
        }
      }

      fprintf(stdout, "\n");
      fflush(stdout);
    }

    msleep(g_lat_interval_ms);
  }

  return NULL;
}

void CIM_DebugLatPrinterStart(int interval_ms) {
  pthread_mutex_lock(&g_lat_mu);
  if (g_lat_started) {
    if (interval_ms > 0) g_lat_interval_ms = interval_ms;
    pthread_mutex_unlock(&g_lat_mu);
    return;
  }

  if (interval_ms > 0) g_lat_interval_ms = interval_ms;
  g_lat_running = 1;
  g_lat_started = 1;

  pthread_mutex_unlock(&g_lat_mu);

  if (pthread_create(&g_lat_th, NULL, lat_printer_thread, NULL) != 0) {
    pthread_mutex_lock(&g_lat_mu);
    g_lat_running = 0;
    g_lat_started = 0;
    pthread_mutex_unlock(&g_lat_mu);
  }
}

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
  g_r_lat_ptr = NULL;
  g_lat_nqps = 0;
  g_lw_ptr = NULL;
  g_lw_nqps = 0;
  g_dbg_seq_ptr = NULL;
  pthread_mutex_unlock(&g_lat_mu);
}

void CIM_DebugLatPrinterSetSource(const char* label, volatile uint64_t* lat, volatile uint64_t* r_lat, int nqps) {
  pthread_mutex_lock(&g_lat_mu);

  g_lat_ptr = lat;
  g_r_lat_ptr = r_lat;
  g_lat_nqps = nqps;

  if (label != NULL && label[0] != 0) {
    strncpy(g_lat_label, label, sizeof(g_lat_label));
    g_lat_label[sizeof(g_lat_label) - 1] = 0;
  } else {
    strncpy(g_lat_label, "qp_lat", sizeof(g_lat_label));
    g_lat_label[sizeof(g_lat_label) - 1] = 0;
  }

  if (label != NULL && label[0] != 0) {
    snprintf(g_r_lat_label, sizeof(g_r_lat_label), "%s_raw", label);
  } else {
    strncpy(g_r_lat_label, "raw_lat", sizeof(g_r_lat_label));
    g_r_lat_label[sizeof(g_r_lat_label) - 1] = 0;
  }

  pthread_mutex_unlock(&g_lat_mu);
}

void CIM_DebugLatPrinterSetWeightSource(const char* label, volatile float* lw, int nqps) {
  pthread_mutex_lock(&g_lat_mu);

  g_lw_ptr = lw;
  g_lw_nqps = nqps;

  if (label != NULL && label[0] != 0) {
    strncpy(g_lw_label, label, sizeof(g_lw_label));
    g_lw_label[sizeof(g_lw_label) - 1] = 0;
  } else {
    strncpy(g_lw_label, "lw", sizeof(g_lw_label));
    g_lw_label[sizeof(g_lw_label) - 1] = 0;
  }

  pthread_mutex_unlock(&g_lat_mu);
}

void CIM_DebugLatPrinterSetSyncSeq(volatile uint32_t* seq) {
  pthread_mutex_lock(&g_lat_mu);
  g_dbg_seq_ptr = seq;
  pthread_mutex_unlock(&g_lat_mu);
}
