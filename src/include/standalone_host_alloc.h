/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/

#ifndef OCCL_STANDALONE_HOST_ALLOC_H_
#define OCCL_STANDALONE_HOST_ALLOC_H_

#include "nccl.h"
#include "bitops.h"
#include "debug.h"
#include "os.h"

#include <stdlib.h>
#include <string.h>

template<typename T>
constexpr size_t occlSizeOfT() { return sizeof(T); }
template<>
constexpr size_t occlSizeOfT<void>() { return 1; }

template <typename T>
inline ncclResult_t occlHostCallocDebug(T** ptr, size_t nelem, const char* filefunc, int line) {
  if (nelem > 0) {
    T* p = static_cast<T*>(malloc(nelem * occlSizeOfT<T>()));
    if (p == NULL) {
      WARN("Failed to malloc %ld bytes", nelem * occlSizeOfT<T>());
      return ncclSystemError;
    }
    memset(p, 0, nelem * occlSizeOfT<T>());
    *ptr = p;
  } else {
    *ptr = NULL;
  }
  INFO(NCCL_ALLOC, "%s:%d Host Calloc Size %ld pointer %p", filefunc, line, nelem * occlSizeOfT<T>(), *ptr);
  return ncclSuccess;
}

template <typename T>
inline ncclResult_t occlHostRealloc(T** ptr, size_t oldNelem, size_t nelem) {
  T* oldp = *ptr;
  if (nelem < oldNelem || (oldp == NULL && oldNelem > 0)) return ncclInternalError;
  if (nelem == oldNelem) return ncclSuccess;

  T* p = static_cast<T*>(malloc(nelem * occlSizeOfT<T>()));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem * occlSizeOfT<T>());
    return ncclSystemError;
  }
  if (oldp && oldNelem) memcpy(p, oldp, oldNelem * occlSizeOfT<T>());
  if (oldp) free(oldp);
  memset(p + oldNelem, 0, (nelem - oldNelem) * occlSizeOfT<T>());
  *ptr = p;
  INFO(NCCL_ALLOC, "Host Realloc old size %ld, new size %ld pointer %p", oldNelem * occlSizeOfT<T>(), nelem * occlSizeOfT<T>(), *ptr);
  return ncclSuccess;
}

inline ncclResult_t occlIbPageAllocDebug(void** ptr, size_t size, const char* filefunc, int line) {
  if (size > 0) {
    long page_size = ncclOsGetPageSize();
    if (page_size < 0) return ncclSystemError;
    void* p = NULL;
    int size_aligned = ROUNDUP(size, page_size);
    int ret = posix_memalign(&p, page_size, size_aligned);
    if (ret != 0) return ncclSystemError;
    memset(p, 0, size);
    *ptr = p;
  } else {
    *ptr = NULL;
  }
  INFO(NCCL_ALLOC, "%s:%d Ib Alloc Size %ld pointer %p", filefunc, line, size, *ptr);
  return ncclSuccess;
}

#define ncclCalloc(...) occlHostCallocDebug(__VA_ARGS__, __FILE__, __LINE__)
#define ncclRealloc(...) occlHostRealloc(__VA_ARGS__)
#define ncclIbMalloc(...) occlIbPageAllocDebug(__VA_ARGS__, __FILE__, __LINE__)

#endif
