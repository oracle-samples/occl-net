/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/


#include "debug.h"
#include "ibvwrap.h"
#include "param.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <thread>

int ncclDebugLevel = 0;
uint64_t ncclDebugMask = 0;
FILE* ncclDebugFile = nullptr;
thread_local int ncclDebugNoWarn = 0;
char ncclLastError[1024] = {};

/**
 * @brief Supplies a silent logger for unit targets that do not exercise production logging.
 * @param[in] level Ignored log severity.
 * @param[in] flags Ignored subsystem flags.
 * @param[in] file Ignored caller source file.
 * @param[in] line Ignored caller source line.
 * @param[in] format NULL or an ignored printf-style message format; trailing arguments are
 * ignored.
 * @note Side effects: None.
 */
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char* file, int line,
    const char* format, ...) {
  (void)level;
  (void)flags;
  (void)file;
  (void)line;
  (void)format;
}

/**
 * @brief Supplies a no-op thread-name hook for tests that do not own native thread naming.
 * @param[in] thread Caller-owned thread whose state is preserved.
 * @param[in] format NULL or an ignored printf-style name format; trailing arguments are ignored.
 * @note Side effects: None.
 */
void ncclSetThreadName(std::thread& thread, const char* format, ...) {
  (void)thread;
  (void)format;
}

/**
 * @brief Supplies the debug-reset symbol required by unit targets using shared support.
 * @note Side effects: None.
 */
extern "C" void ncclResetDebugInit() {
}

/**
 * @brief Reports that shared test support has no configured home-directory source.
 * @return NULL.
 * @note Side effects: None.
 */
const char* userHomeDir() {
  return nullptr;
}

/**
 * @brief Supplies a no-op environment-file loader for tests outside param coverage.
 * @param[in] filename Ignored configuration path.
 * @note Side effects: None.
 */
void setEnvFile(const char* filename) {
  (void)filename;
}

/**
 * @brief Supplies a no-op environment initializer for isolated unit targets.
 * @note Side effects: None.
 */
void initEnv() {
}

/**
 * @brief Reads one value from the process environment.
 * @param[in] name Valid environment-variable name.
 * @return Borrowed process-environment storage valid until a subsequent environment access or
 * mutation, or NULL when the name is absent.
 * @note Side effects: None.
 */
const char* ncclGetEnv(const char* name) {
  return std::getenv(name);
}

/**
 * @brief Resolves a parameter to its declared default and initializes its cache outputs.
 * @param[in] name Ignored parameter name.
 * @param[in] defaultValue Declared parameter default.
 * @param[in] uninitialized Ignored cache sentinel.
 * @param[out] cache Valid non-NULL storage receiving defaultValue.
 * @param[out] noCache Valid non-NULL storage receiving zero to enable caching.
 * @return defaultValue.
 * @note Side effects: Writes *cache and *noCache.
 */
int64_t ncclLoadParam(char const* name, int64_t defaultValue, int64_t uninitialized,
    int64_t* cache, int8_t* noCache) {
  (void)name;
  (void)uninitialized;
  *cache = defaultValue;
  *noCache = 0;
  return defaultValue;
}

/**
 * @brief Supplies a deterministic operating-system page size to allocation tests.
 * @return 4096 bytes.
 * @note Side effects: None.
 */
size_t ncclOsGetPageSize() {
  return 4096;
}

#ifndef OCCL_TEST_USE_PRODUCTION_IBV_TEXT
/**
 * @brief Supplies stable completion-status text outside the dedicated ibv text tests.
 * @param[in] status Ignored completion status.
 * @return Static test status text.
 */
const char* ibvWcStatusStr(enum ibv_wc_status status) {
  (void)status;
  return "test-wc-status";
}

/**
 * @brief Supplies stable completion-opcode text outside the dedicated ibv text tests.
 * @param[in] opcode Ignored completion opcode.
 * @return Static test opcode text.
 */
const char* ibvWcOpcodeStr(enum ibv_wc_opcode opcode) {
  (void)opcode;
  return "test-wc-opcode";
}

/**
 * @brief Supplies stable work-request opcode text outside the dedicated ibv text tests.
 * @param[in] opcode Ignored work-request opcode.
 * @return Static test opcode text.
 */
const char* ibvWrOpcodeStr(enum ibv_wr_opcode opcode) {
  (void)opcode;
  return "test-wr-opcode";
}
#endif
