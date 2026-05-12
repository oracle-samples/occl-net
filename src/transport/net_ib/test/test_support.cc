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

void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {
}

void ncclSetThreadName(std::thread&, const char*, ...) {
}

extern "C" void ncclResetDebugInit() {
}

const char* userHomeDir() {
  return nullptr;
}

void setEnvFile(const char*) {
}

void initEnv() {
}

const char* ncclGetEnv(const char* name) {
  return std::getenv(name);
}

int64_t ncclLoadParam(char const*, int64_t defaultValue, int64_t, int64_t* cache, int8_t* noCache) {
  *cache = defaultValue;
  *noCache = 0;
  return defaultValue;
}

size_t ncclOsGetPageSize() {
  return 4096;
}

const char* ibvWcStatusStr(enum ibv_wc_status) {
  return "test-wc-status";
}

const char* ibvWcOpcodeStr(enum ibv_wc_opcode) {
  return "test-wc-opcode";
}

const char* ibvWrOpcodeStr(enum ibv_wr_opcode) {
  return "test-wr-opcode";
}
