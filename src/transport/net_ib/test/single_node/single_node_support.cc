/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/

#include "env.h"
#include "gdrwrap.h"
#include "mem_manager.h"

#include <cstdlib>

gdr_t ncclGdrCopy = nullptr;

ncclResult_t ncclInitEnv(void) {
  return ncclSuccess;
}

const char* ncclEnvPluginGetEnv(const char* name) {
  return std::getenv(name);
}

bool ncclEnvPluginInitialized(void) {
  return false;
}

int ncclParamSetCpuStackSize(void) {
  return 1;
}

ncclResult_t ncclMemTrack(struct ncclMemManager* manager, void* ptr, size_t size,
                          CUmemGenericAllocationHandle handle,
                          CUmemAllocationHandleType handleType,
                          ncclMemType_t memType) {
  (void)manager;
  (void)ptr;
  (void)size;
  (void)handle;
  (void)handleType;
  (void)memType;
  return ncclSuccess;
}

ncclResult_t ncclMemUntrack(struct ncclMemManager* manager, void* ptr, size_t size) {
  (void)manager;
  (void)ptr;
  (void)size;
  return ncclSuccess;
}
