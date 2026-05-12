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
