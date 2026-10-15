/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/


#include "debug.h"
#include "os.h"
#include "socket.h"
#include "utils.h"

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace {

/** @brief Scripts one low-level I/O result and errno value. */
struct IoStep {
  ssize_t result = 0;
  int error = 0;
};

/** @brief Records one integer socket-option request. */
struct SocketOptionCall {
  int level = 0;
  int option = 0;
  int value = 0;
  socklen_t length = 0;
};

/**
 * @brief Configures deterministic Linux boundary outcomes and records intercepted calls
 * for later assertions.
 */
struct LinuxBoundaryState {
  bool forceDlopen = false;
  bool forceDlsym = false;
  bool forceDlerror = false;
  bool syntheticDlerrorPending = false;
  void* dlopenResult = nullptr;
  void* dlsymResult = nullptr;
  std::string dlerrorText;
  std::string dlopenPath;
  std::string dlsymName;
  int dlopenMode = 0;
  int dlcloseCalls = 0;
  void* dlcloseHandle = nullptr;

  bool forceStackSize = false;
  size_t stackSize = 0;
  int pthreadAttrInitResult = 0;
  int pthreadAttrGetStackResult = 0;
  int pthreadAttrSetStackResult = 0;
  int pthreadAttrDestroyResult = 0;
  int pthreadAttrInitCalls = 0;
  int pthreadAttrDestroyCalls = 0;

  bool forceRlimit = false;
  bool failGetRlimit = false;
  bool failSetRlimit = false;
  struct rlimit rlimitValue = {};
  struct rlimit setRlimitValue = {};
  int getRlimitCalls = 0;
  int setRlimitCalls = 0;

  bool forcePoll = false;
  int pollResult = 0;
  int pollError = 0;
  int pollCalls = 0;
  int pollFd = -1;
  short pollEvents = 0;
  int pollTimeout = 0;

  bool forceAccept = false;
  int acceptResult = -1;
  int acceptError = 0;
  int acceptCalls = 0;

  int failFcntlCommand = -1;
  int failFcntlError = 0;
  int fcntlCalls = 0;
  std::vector<int> fcntlCommands;
  int failSocketOption = -1;
  int failSocketOptionError = 0;
  std::vector<SocketOptionCall> socketOptions;

  bool forceSocketFailure = false;
  int socketError = 0;
  int lastCreatedSocket = -1;
  bool forceDup2Failure = false;
  int dup2Error = 0;

  bool forceConnect = false;
  int connectResult = 0;
  int connectError = 0;
  int connectCalls = 0;
  bool forceGetSockOpt = false;
  int getSockOptResult = 0;
  int getSockOptError = 0;
  int socketPendingError = 0;

  std::deque<IoStep> sendSteps;
  std::deque<IoStep> recvSteps;
  int sendCalls = 0;
  int recvCalls = 0;

  bool forceGetIfAddrsFailure = false;
  int getIfAddrsError = 0;
  struct ifaddrs* interfaceHead = nullptr;
  int getIfAddrsCalls = 0;
  int freeIfAddrsCalls = 0;
  std::string parsedInterfaceExpression;
  bool parsedExpressionWasNull = false;
  bool lastInterfaceMatchExact = false;
  std::vector<std::string> includedInterfaceNames;
  std::vector<std::string> excludedInterfaceNames;

  bool failGetAffinity = false;
  bool failSetAffinity = false;
  int affinityError = 0;
  bool forceGetCpu = false;
  int cpuResult = 0;

  int closeProgressCalls = 0;
  int closeAfterProgressCalls = 1;
  ncclResult_t closeProgressResult = ncclSuccess;
};

LinuxBoundaryState linuxBoundary;
int stackSizeParameter = 0;
int pollTimeoutParameter = 7;
int retryTimeoutParameter = 0;
long int retryCountParameter = 2;
int receiveBufferParameter = -1;
int sendBufferParameter = -1;

/**
 * @brief Restores deterministic boundary behavior before an independent Linux OS scenario.
 * @note Side effects: Replaces the recording boundary and every configurable parameter value.
 */
void ResetLinuxBoundary() {
  linuxBoundary = LinuxBoundaryState{};
  stackSizeParameter = 0;
  pollTimeoutParameter = 7;
  retryTimeoutParameter = 0;
  retryCountParameter = 2;
  receiveBufferParameter = -1;
  sendBufferParameter = -1;
}

/**
 * @brief Returns whether a string occurs in one recorded interface-name list.
 * @param[in] values List of interface names.
 * @param[in] value Name to find.
 * @return true when value occurs in values, otherwise false.
 * @note This function has no side effects.
 */
bool ContainsName(const std::vector<std::string>& values, const char* value) {
  for (const std::string& candidate : values) {
    if (candidate == value) return true;
  }
  return false;
}

/**
 * @brief Returns the next scripted I/O result or delegates to the real system call.
 * @param[in,out] steps Scripted result queue; the first entry is consumed when present.
 * @param[in] descriptor Valid or invalid descriptor forwarded to the real call when steps is empty.
 * @param[in,out] buffer Buffer read by send or written by receive.
 * @param[in] length Requested byte count in [0, SIZE_MAX].
 * @param[in] flags System-call flags.
 * @param[in] receive true for recv and false for send.
 * @return The scripted or real system-call result.
 * @note Side effects: May consume one step, set errno, transfer bytes, and modify a receive buffer.
 */
ssize_t RunIoStep(std::deque<IoStep>& steps, int descriptor, void* buffer, size_t length, int flags,
                  bool receive) {
  if (!steps.empty()) {
    const IoStep step = steps.front();
    steps.pop_front();
    errno = step.error;
    return step.result;
  }
  return receive ? ::recv(descriptor, buffer, length, flags) : ::send(descriptor, buffer, length, flags);
}

/**
 * @brief Test boundary for dlopen with optional deterministic output.
 * @param[in] path NULL or a library path.
 * @param[in] mode Loader mode flags.
 * @return The configured handle or the real dlopen result.
 * @note Side effects: Records arguments and may open a library or arm a synthetic diagnostic.
 */
void* OsDlopenForTest(const char* path, int mode) {
  linuxBoundary.dlopenPath = path == nullptr ? "<null>" : path;
  linuxBoundary.dlopenMode = mode;
  if (!linuxBoundary.forceDlopen) return ::dlopen(path, mode);
  linuxBoundary.syntheticDlerrorPending = linuxBoundary.dlopenResult == nullptr;
  return linuxBoundary.dlopenResult;
}

/**
 * @brief Test boundary for dlsym with optional deterministic output.
 * @param[in] handle Loader handle forwarded to the real call when not forced.
 * @param[in] symbol Valid non-NULL symbol name.
 * @return The configured address or the real dlsym result.
 * @note Side effects: Records the symbol and may update native or synthetic loader diagnostics.
 */
void* OsDlsymForTest(void* handle, const char* symbol) {
  linuxBoundary.dlsymName = symbol == nullptr ? "<null>" : symbol;
  if (!linuxBoundary.forceDlsym) return ::dlsym(handle, symbol);
  linuxBoundary.syntheticDlerrorPending = linuxBoundary.dlsymResult == nullptr;
  return linuxBoundary.dlsymResult;
}

/**
 * @brief Test boundary for dlerror with consumable synthetic diagnostics.
 * @return The configured pending diagnostic, NULL when consumed, or the real dlerror result.
 * @note Side effects: May consume a native or synthetic loader diagnostic.
 */
char* OsDlerrorForTest() {
  if (!linuxBoundary.forceDlerror) return ::dlerror();
  if (!linuxBoundary.syntheticDlerrorPending) return nullptr;
  linuxBoundary.syntheticDlerrorPending = false;
  return linuxBoundary.dlerrorText.data();
}

/**
 * @brief Test boundary for dlclose that protects synthetic handles.
 * @param[in] handle Valid real or synthetic loader handle.
 * @return Zero for a synthetic handle or the real dlclose result.
 * @note Side effects: Records the handle and may close a real library.
 */
int OsDlcloseForTest(void* handle) {
  linuxBoundary.dlcloseCalls++;
  linuxBoundary.dlcloseHandle = handle;
  if (linuxBoundary.forceDlopen && handle == linuxBoundary.dlopenResult) return 0;
  return ::dlclose(handle);
}

/**
 * @brief Initializes an attribute object unless a deterministic error is configured.
 * @param[out] attr Valid storage for a pthread attribute object.
 * @return The configured nonzero error or the native pthread_attr_init result.
 * @note Side effects: Increments the init-call count and may initialize @p attr.
 */
int OsPthreadAttrInitForTest(pthread_attr_t* attr) {
  linuxBoundary.pthreadAttrInitCalls++;
  if (linuxBoundary.pthreadAttrInitResult != 0) return linuxBoundary.pthreadAttrInitResult;
  return ::pthread_attr_init(attr);
}

/**
 * @brief Returns a configured stack size or delegates to pthread_attr_getstacksize.
 * @param[in] attr Valid initialized pthread attribute object.
 * @param[out] size Valid storage for the resulting stack size.
 * @return The configured error, zero for a forced size, or the native result.
 * @note Side effects: May write @p size.
 */
int OsPthreadAttrGetStackSizeForTest(const pthread_attr_t* attr, size_t* size) {
  if (linuxBoundary.pthreadAttrGetStackResult != 0) return linuxBoundary.pthreadAttrGetStackResult;
  if (linuxBoundary.forceStackSize) {
    *size = linuxBoundary.stackSize;
    return 0;
  }
  return ::pthread_attr_getstacksize(attr, size);
}

/**
 * @brief Sets an attribute stack size unless a deterministic error is configured.
 * @param[in,out] attr Valid initialized pthread attribute object.
 * @param[in] size Stack size forwarded to pthread_attr_setstacksize.
 * @return The configured nonzero error or the native result.
 * @note Side effects: May update @p attr.
 */
int OsPthreadAttrSetStackSizeForTest(pthread_attr_t* attr, size_t size) {
  if (linuxBoundary.pthreadAttrSetStackResult != 0) return linuxBoundary.pthreadAttrSetStackResult;
  return ::pthread_attr_setstacksize(attr, size);
}

/**
 * @brief Destroys an attribute object unless a deterministic error is configured.
 * @param[in,out] attr Valid initialized pthread attribute object.
 * @return The configured nonzero error or the native pthread_attr_destroy result.
 * @note Side effects: Increments the destroy-call count and may destroy @p attr.
 */
int OsPthreadAttrDestroyForTest(pthread_attr_t* attr) {
  linuxBoundary.pthreadAttrDestroyCalls++;
  if (linuxBoundary.pthreadAttrDestroyResult != 0) return linuxBoundary.pthreadAttrDestroyResult;
  return ::pthread_attr_destroy(attr);
}

/**
 * @brief Returns a configured resource limit or delegates to getrlimit.
 * @param[in] resource Resource selector forwarded to getrlimit.
 * @param[out] limit Valid storage for the selected resource limit.
 * @return Zero on success or -1 on failure.
 * @note Side effects: Increments the call count and may write @p limit or set errno.
 */
int OsGetRlimitForTest(int resource, struct rlimit* limit) {
  linuxBoundary.getRlimitCalls++;
  if (!linuxBoundary.forceRlimit) return ::getrlimit(resource, limit);
  if (linuxBoundary.failGetRlimit) {
    errno = EACCES;
    return -1;
  }
  *limit = linuxBoundary.rlimitValue;
  return 0;
}

/**
 * @brief Records a configured resource limit or delegates to setrlimit.
 * @param[in] resource Resource selector forwarded to setrlimit.
 * @param[in] limit Valid resource limit to record or apply.
 * @return Zero on success or -1 on failure.
 * @note Side effects: Increments the call count and may record @p limit, set errno, or alter
 * the selected process resource limit.
 */
int OsSetRlimitForTest(int resource, const struct rlimit* limit) {
  linuxBoundary.setRlimitCalls++;
  if (!linuxBoundary.forceRlimit) return ::setrlimit(resource, limit);
  linuxBoundary.setRlimitValue = *limit;
  if (linuxBoundary.failSetRlimit) {
    errno = EPERM;
    return -1;
  }
  return 0;
}

/**
 * @brief Records the first poll request and returns a configured or native result.
 * @param[in,out] descriptors NULL for zero entries or valid storage for @p count entries.
 * @param[in] count Number of poll descriptors.
 * @param[in] timeout Poll timeout in milliseconds.
 * @return The configured result or the native ready-descriptor count or error.
 * @note Side effects: Increments the call count, records inputs, and may update revents or errno.
 */
int OsPollForTest(struct pollfd* descriptors, nfds_t count, int timeout) {
  linuxBoundary.pollCalls++;
  if (count > 0) {
    linuxBoundary.pollFd = descriptors[0].fd;
    linuxBoundary.pollEvents = descriptors[0].events;
  }
  linuxBoundary.pollTimeout = timeout;
  if (!linuxBoundary.forcePoll) return ::poll(descriptors, count, timeout);
  errno = linuxBoundary.pollError;
  return linuxBoundary.pollResult;
}

/**
 * @brief Accepts a connection or returns a configured descriptor and errno.
 * @param[in] descriptor Listening socket descriptor.
 * @param[out] address NULL or storage for the peer address.
 * @param[in,out] length NULL or storage containing capacity and receiving address length.
 * @return The accepted descriptor or -1 on failure.
 * @note Side effects: Increments the call count and may open a descriptor, write outputs, or
 * set errno.
 */
int OsAcceptForTest(int descriptor, struct sockaddr* address, socklen_t* length) {
  linuxBoundary.acceptCalls++;
  if (!linuxBoundary.forceAccept) return ::accept(descriptor, address, length);
  errno = linuxBoundary.acceptError;
  return linuxBoundary.acceptResult;
}

/**
 * @brief Records fcntl commands and optionally fails one selected command.
 * @param[in] descriptor Descriptor forwarded to fcntl.
 * @param[in] command Command forwarded to fcntl; commands other than F_GETFL require an int
 * variadic argument.
 * @return -1 for the selected failure or the native fcntl result.
 * @note Side effects: Records the command and may set errno or change descriptor state.
 */
int OsFcntlForTest(int descriptor, int command, ...) {
  linuxBoundary.fcntlCalls++;
  linuxBoundary.fcntlCommands.push_back(command);
  if (command == linuxBoundary.failFcntlCommand) {
    errno = linuxBoundary.failFcntlError;
    return -1;
  }
  if (command == F_GETFL) return ::fcntl(descriptor, command);
  va_list arguments;
  va_start(arguments, command);
  const int value = va_arg(arguments, int);
  va_end(arguments);
  return ::fcntl(descriptor, command, value);
}

/**
 * @brief Records a socket option and optionally fails one selected option.
 * @param[in] descriptor Socket descriptor forwarded to setsockopt.
 * @param[in] level Protocol level forwarded to setsockopt.
 * @param[in] option Option selector forwarded to setsockopt.
 * @param[in] value NULL or option-value storage of @p length bytes.
 * @param[in] length Size of @p value in bytes.
 * @return -1 for the selected failure or the native setsockopt result.
 * @note Side effects: Appends a call record and may set errno or change a socket option.
 */
int OsSetSockOptForTest(int descriptor, int level, int option, const void* value, socklen_t length) {
  SocketOptionCall call = {level, option, 0, length};
  if (value != nullptr && length == sizeof(call.value)) std::memcpy(&call.value, value, sizeof(call.value));
  linuxBoundary.socketOptions.push_back(call);
  if (option == linuxBoundary.failSocketOption) {
    errno = linuxBoundary.failSocketOptionError;
    return -1;
  }
  return ::setsockopt(descriptor, level, option, value, length);
}

/**
 * @brief Creates and records a socket unless deterministic failure is enabled.
 * @param[in] family Socket domain forwarded to socket.
 * @param[in] type Socket type and flags forwarded to socket.
 * @param[in] protocol Protocol selector forwarded to socket.
 * @return A new descriptor on success or -1 on failure.
 * @note Side effects: May set errno; otherwise records the native result, which may open a socket.
 */
int OsSocketForTest(int family, int type, int protocol) {
  if (linuxBoundary.forceSocketFailure) {
    errno = linuxBoundary.socketError;
    return -1;
  }
  linuxBoundary.lastCreatedSocket = ::socket(family, type, protocol);
  return linuxBoundary.lastCreatedSocket;
}

/**
 * @brief Duplicates a descriptor unless deterministic failure is enabled.
 * @param[in] oldDescriptor Source descriptor forwarded to dup2.
 * @param[in] newDescriptor Target descriptor forwarded to dup2.
 * @return @p newDescriptor on success or -1 on failure.
 * @note Side effects: May set errno or replace @p newDescriptor with a duplicate.
 */
int OsDup2ForTest(int oldDescriptor, int newDescriptor) {
  if (linuxBoundary.forceDup2Failure) {
    errno = linuxBoundary.dup2Error;
    return -1;
  }
  return ::dup2(oldDescriptor, newDescriptor);
}

/**
 * @brief Starts a connection or returns a configured status and errno.
 * @param[in] descriptor Socket descriptor forwarded to connect.
 * @param[in] address Valid peer address of @p length bytes.
 * @param[in] length Size of @p address.
 * @return The configured result or native connect result.
 * @note Side effects: Increments the call count and may set errno or change connection state.
 */
int OsConnectForTest(int descriptor, const struct sockaddr* address, socklen_t length) {
  linuxBoundary.connectCalls++;
  if (!linuxBoundary.forceConnect) return ::connect(descriptor, address, length);
  errno = linuxBoundary.connectError;
  return linuxBoundary.connectResult;
}

/**
 * @brief Returns a configured socket error or delegates to getsockopt.
 * @param[in] descriptor Socket descriptor forwarded to getsockopt.
 * @param[in] level Protocol level forwarded to getsockopt.
 * @param[in] option Option selector forwarded to getsockopt.
 * @param[out] value Valid option-value storage.
 * @param[in,out] length Valid storage containing capacity and receiving native output length.
 * @return Zero on success or -1 on failure.
 * @note Side effects: May write @p value and @p length or set errno.
 */
int OsGetSockOptForTest(int descriptor, int level, int option, void* value, socklen_t* length) {
  if (!linuxBoundary.forceGetSockOpt) return ::getsockopt(descriptor, level, option, value, length);
  if (linuxBoundary.getSockOptResult == -1) {
    errno = linuxBoundary.getSockOptError;
    return -1;
  }
  *static_cast<int*>(value) = linuxBoundary.socketPendingError;
  return 0;
}

/**
 * @brief Returns the next scripted send result or delegates to send.
 * @param[in] descriptor Socket descriptor forwarded to send.
 * @param[in] buffer Buffer containing @p length bytes to send.
 * @param[in] length Requested byte count.
 * @param[in] flags Send flags.
 * @return The scripted or native byte count or error.
 * @note Side effects: Increments the call count and may consume a script step, set errno, or
 * transfer bytes.
 */
ssize_t OsSendForTest(int descriptor, const void* buffer, size_t length, int flags) {
  linuxBoundary.sendCalls++;
  return RunIoStep(linuxBoundary.sendSteps, descriptor, const_cast<void*>(buffer), length, flags, false);
}

/**
 * @brief Returns the next scripted receive result or delegates to recv.
 * @param[in] descriptor Socket descriptor forwarded to recv.
 * @param[out] buffer Buffer with capacity for @p length received bytes.
 * @param[in] length Requested byte count.
 * @param[in] flags Receive flags.
 * @return The scripted or native byte count, closure indicator, or error.
 * @note Side effects: Increments the call count and may consume a script step, set errno,
 * transfer bytes, or write @p buffer.
 */
ssize_t OsRecvForTest(int descriptor, void* buffer, size_t length, int flags) {
  linuxBoundary.recvCalls++;
  return RunIoStep(linuxBoundary.recvSteps, descriptor, buffer, length, flags, true);
}

/**
 * @brief Publishes a caller-owned synthetic interface list or a configured failure.
 * @param[out] interfaces Valid storage for the synthetic list head.
 * @return Zero on success or -1 on configured failure.
 * @note Side effects: Increments the call count and may write @p interfaces or set errno.
 */
int OsGetIfAddrsForTest(struct ifaddrs** interfaces) {
  linuxBoundary.getIfAddrsCalls++;
  if (linuxBoundary.forceGetIfAddrsFailure) {
    errno = linuxBoundary.getIfAddrsError;
    return -1;
  }
  *interfaces = linuxBoundary.interfaceHead;
  return 0;
}

/**
 * @brief Records release of a synthetic interface list without freeing its nodes.
 * @param[in] interfaces Synthetic list head; accepted without inspection.
 * @note Side effects: Increments the free-call count.
 */
void OsFreeIfAddrsForTest([[maybe_unused]] struct ifaddrs* interfaces) {
  linuxBoundary.freeIfAddrsCalls++;
}

/**
 * @brief Reads affinity or returns a configured failure.
 * @param[in] pid Process or thread identifier forwarded to sched_getaffinity.
 * @param[in] size Size of @p affinity in bytes.
 * @param[out] affinity Valid affinity-mask storage.
 * @return Zero on success or -1 on failure.
 * @note Side effects: May write @p affinity or set errno.
 */
int OsGetAffinityForTest(pid_t pid, size_t size, cpu_set_t* affinity) {
  if (linuxBoundary.failGetAffinity) {
    errno = linuxBoundary.affinityError;
    return -1;
  }
  return ::sched_getaffinity(pid, size, affinity);
}

/**
 * @brief Applies affinity or returns a configured failure.
 * @param[in] pid Process or thread identifier forwarded to sched_setaffinity.
 * @param[in] size Size of @p affinity in bytes.
 * @param[in] affinity Valid affinity mask to apply.
 * @return Zero on success or -1 on failure.
 * @note Side effects: May set errno or change process or thread affinity.
 */
int OsSetAffinityForTest(pid_t pid, size_t size, const cpu_set_t* affinity) {
  if (linuxBoundary.failSetAffinity) {
    errno = linuxBoundary.affinityError;
    return -1;
  }
  return ::sched_setaffinity(pid, size, affinity);
}

/**
 * @brief Returns a configured CPU index or delegates to sched_getcpu.
 * @return The current CPU index or -1 on native failure.
 * @note Side effects: The native call may update errno.
 */
int OsGetCpuForTest() {
  return linuxBoundary.forceGetCpu ? linuxBoundary.cpuResult : ::sched_getcpu();
}

}  // namespace

int ncclDebugLevel = 0;
uint64_t ncclDebugMask = 0;
FILE* ncclDebugFile = nullptr;
thread_local int ncclDebugNoWarn = 0;
char ncclLastError[1024] = {};

/**
 * @brief Discards diagnostics emitted by the Linux OS implementation under test.
 * @param[in] level Diagnostic severity; accepted without inspection.
 * @param[in] flags Diagnostic subsystem flags; accepted without inspection.
 * @param[in] file NULL or source-file text; accepted without inspection.
 * @param[in] line Source line; accepted without inspection.
 * @param[in] format NULL or a printf-style format; accepted without inspection.
 * @note This function has no side effects.
 */
void ncclDebugLog([[maybe_unused]] ncclDebugLogLevel level, [[maybe_unused]] unsigned long flags,
                  [[maybe_unused]] const char* file, [[maybe_unused]] int line,
                  [[maybe_unused]] const char* format, ...) {
}

/**
 * @brief Returns the configured stack-adjustment enable parameter.
 * @return Zero when stack adjustment is disabled, otherwise the configured enable value.
 * @note This function has no side effects.
 */
int ncclParamSetCpuStackSize() {
  return stackSizeParameter;
}

/**
 * @brief Returns the configured socket poll timeout.
 * @return Poll timeout in milliseconds.
 * @note This function has no side effects.
 */
int ncclParamPollTimeOut() {
  return pollTimeoutParameter;
}

/**
 * @brief Returns the configured socket reconnect delay.
 * @return Reconnect delay in milliseconds.
 * @note This function has no side effects.
 */
int ncclParamRetryTimeOut() {
  return retryTimeoutParameter;
}

/**
 * @brief Returns the configured retry limit.
 * @return Maximum retry count.
 * @note This function has no side effects.
 */
long int ncclParamRetryCnt() {
  return retryCountParameter;
}

/**
 * @brief Returns the configured receive-buffer size.
 * @return Receive-buffer size in bytes, or -1 when no override is configured.
 * @note This function has no side effects.
 */
int ncclParamSocketMaxRecvBuff() {
  return receiveBufferParameter;
}

/**
 * @brief Returns the configured send-buffer size.
 * @return Send-buffer size in bytes, or -1 when no override is configured.
 * @note This function has no side effects.
 */
int ncclParamSocketMaxSendBuff() {
  return sendBufferParameter;
}

/**
 * @brief Records the user interface expression without re-testing the utility parser.
 * @param[in] expression NULL or the expression after Linux include/exclude modifiers were removed.
 * @param[out] interfaces Valid storage for at least maximum entries.
 * @param[in] maximum Maximum entry count in [0, INT_MAX].
 * @return Zero for NULL or empty input, otherwise one.
 * @note Side effects: Records expression and initializes the first output entry when present.
 */
int parseStringList(const char* expression, struct netIf* interfaces, int maximum) {
  linuxBoundary.parsedExpressionWasNull = expression == nullptr;
  linuxBoundary.parsedInterfaceExpression = expression == nullptr ? "" : expression;
  if (expression == nullptr || expression[0] == '\0' || maximum == 0) return 0;
  std::snprintf(interfaces[0].prefix, sizeof(interfaces[0].prefix), "%s", expression);
  interfaces[0].port = -1;
  return 1;
}

/**
 * @brief Returns the configured semantic match for one interface name.
 * @param[in] name Valid non-NULL interface name.
 * @param[in] port Ignored port selector.
 * @param[in] interfaces Parsed list storage; not modified.
 * @param[in] listSize Parsed list length in [0, MAX_IFS].
 * @param[in] exact Exact-match selector recorded for verification.
 * @return false for excluded names, true for included names or an empty list, otherwise false.
 * @note Side effects: Records the exact-match selector.
 */
bool matchIfList(const char* name, int port, struct netIf* interfaces, int listSize, bool exact) {
  (void)port;
  (void)interfaces;
  linuxBoundary.lastInterfaceMatchExact = exact;
  if (listSize == 0) return true;
  if (ContainsName(linuxBoundary.excludedInterfaceNames, name)) return false;
  return ContainsName(linuxBoundary.includedInterfaceNames, name);
}

/**
 * @brief Formats a stable address label for diagnostics that are not under test here.
 * @param[in] address Address accepted but not inspected.
 * @param[out] buffer Valid output storage receiving a fixed label.
 * @param[in] numericHostForm Ignored formatting selector.
 * @return buffer.
 * @note Side effects: Writes a fixed null-terminated label to buffer.
 */
const char* ncclSocketToString(const union ncclSocketAddress* address, char* buffer, const int numericHostForm) {
  (void)address;
  (void)numericHostForm;
  std::strcpy(buffer, "test-address");
  return buffer;
}

/**
 * @brief Supplies deterministic close-wait progress without performing unrelated socket-layer work.
 * @param[in] operation Expected receive operation; ignored.
 * @param[in] socket Socket accepted but not modified.
 * @param[in] data One-byte storage left unchanged.
 * @param[in] size Expected size one; ignored.
 * @param[in] offset Progress offset left unchanged.
 * @param[out] closed NULL or storage set after the configured number of calls.
 * @return The configured progress result.
 * @note Side effects: Increments the close-progress count and writes closed.
 */
ncclResult_t ncclSocketProgress(int operation, struct ncclSocket* socket, void* data, int size, int* offset,
                                int* closed) {
  (void)operation;
  (void)socket;
  (void)data;
  (void)size;
  (void)offset;
  linuxBoundary.closeProgressCalls++;
  if (closed != nullptr) *closed = linuxBoundary.closeProgressCalls >= linuxBoundary.closeAfterProgressCalls;
  return linuxBoundary.closeProgressResult;
}

/**
 * @brief Compiles the production Linux implementation against recording system-call seams.
 */
#define dlopen OsDlopenForTest
#define dlsym OsDlsymForTest
#define dlerror OsDlerrorForTest
#define dlclose OsDlcloseForTest
#define pthread_attr_init OsPthreadAttrInitForTest
#define pthread_attr_getstacksize OsPthreadAttrGetStackSizeForTest
#define pthread_attr_setstacksize OsPthreadAttrSetStackSizeForTest
#define pthread_attr_destroy OsPthreadAttrDestroyForTest
#define getrlimit OsGetRlimitForTest
#define setrlimit OsSetRlimitForTest
#define poll OsPollForTest
#define accept OsAcceptForTest
#define fcntl OsFcntlForTest
#define setsockopt OsSetSockOptForTest
#define socket OsSocketForTest
#define dup2 OsDup2ForTest
#define connect OsConnectForTest
#define getsockopt OsGetSockOptForTest
#define send OsSendForTest
#define recv OsRecvForTest
#define getifaddrs OsGetIfAddrsForTest
#define freeifaddrs OsFreeIfAddrsForTest
#define sched_getaffinity OsGetAffinityForTest
#define sched_setaffinity OsSetAffinityForTest
#define sched_getcpu OsGetCpuForTest
#include "../../../os/linux.cc"  // NOLINT(build/include)
#undef sched_getcpu
#undef sched_setaffinity
#undef sched_getaffinity
#undef freeifaddrs
#undef getifaddrs
#undef recv
#undef send
#undef getsockopt
#undef connect
#undef dup2
#undef socket
#undef setsockopt
#undef fcntl
#undef accept
#undef poll
#undef setrlimit
#undef getrlimit
#undef pthread_attr_destroy
#undef pthread_attr_setstacksize
#undef pthread_attr_getstacksize
#undef pthread_attr_init
#undef dlclose
#undef dlerror
#undef dlsym
#undef dlopen

namespace {

/** @brief Provides isolated dependency state for Linux OS wrapper tests. */
class LinuxOsTest : public ::testing::Test {
 protected:
  /**
   * @brief Restores boundary state before each test case.
   * @note Side effects: Resets all Linux test doubles and parameter values.
   */
  void SetUp() override {
    ResetLinuxBoundary();
  }
};

/** @brief Groups dynamic-loader and basic process-runtime contracts. */
class LinuxRuntimeTest : public LinuxOsTest {};

/** @brief Groups socket descriptor, connection, transfer, and close contracts. */
class LinuxSocketTest : public LinuxOsTest {};

/** @brief Groups deterministic interface enumeration and subnet contracts. */
class LinuxInterfaceTest : public LinuxOsTest {};

/** @brief Groups CPU, synchronization, and shared-memory resource contracts. */
class LinuxResourceTest : public LinuxOsTest {};

/**
 * @brief Verifies loader success, diagnostics, mode selection, closure, and NVML status
 * translation.
 * @note Side effects: Opens and closes libm and mutates the test-controlled loader state.
 */
TEST_F(LinuxRuntimeTest, LoaderBehavior) {
  SCOPED_TRACE("Linux loader wrappers must expose real symbols and preserve deterministic failure diagnostics.");

  // A real library and symbol provide an independent success oracle.
  ncclOsLibraryHandle library = ncclOsDlopen("libm.so.6");
  ASSERT_NE(library, nullptr) << "Opening the standard Linux math library unexpectedly failed";
  void* cosine = ncclOsDlsym(library, "cos");
  EXPECT_NE(cosine, nullptr) << "The loader did not resolve the standard cos symbol";
  ncclOsDlclose(library);
  EXPECT_EQ(linuxBoundary.dlcloseHandle, library) << "Library close received a different handle";

  // Both explicit modes must map to their corresponding native loader flags.
  library = ncclOsDlopen("libm.so.6", NCCL_OS_DL_LAZY);
  ASSERT_NE(library, nullptr) << "Lazy library loading unexpectedly failed";
  EXPECT_EQ(linuxBoundary.dlopenMode, RTLD_LAZY) << "Lazy mode was not forwarded as RTLD_LAZY";
  ncclOsDlclose(library);
  library = ncclOsDlopen("libm.so.6", NCCL_OS_DL_NOW);
  ASSERT_NE(library, nullptr) << "Immediate library loading unexpectedly failed";
  EXPECT_EQ(linuxBoundary.dlopenMode, RTLD_NOW) << "Immediate mode was not forwarded as RTLD_NOW";
  ncclOsDlclose(library);

  // A missing library must cache the exact loader diagnostic exposed by ncclOsDlerror.
  linuxBoundary.forceDlopen = true;
  linuxBoundary.forceDlerror = true;
  linuxBoundary.dlopenResult = nullptr;
  linuxBoundary.dlerrorText = "synthetic open failure";
  EXPECT_EQ(ncclOsDlopen("missing-test-library.so"), nullptr)
      << "A failed library open returned a non-NULL handle";
  EXPECT_STREQ(ncclOsDlerror(), "synthetic open failure")
      << "The failed library open did not preserve its loader diagnostic";

  // A missing symbol must replace the previous diagnostic rather than leaving stale text.
  linuxBoundary.forceDlopen = false;
  linuxBoundary.forceDlerror = false;
  library = ncclOsDlopen("libm.so.6");
  ASSERT_NE(library, nullptr) << "The symbol-error scenario could not open libm";
  linuxBoundary.forceDlsym = true;
  linuxBoundary.forceDlerror = true;
  linuxBoundary.dlsymResult = nullptr;
  linuxBoundary.dlerrorText = "synthetic symbol failure";
  EXPECT_EQ(ncclOsDlsym(library, "missing_test_symbol"), nullptr)
      << "A failed symbol lookup returned a non-NULL address";
  EXPECT_STREQ(ncclOsDlerror(), "synthetic symbol failure")
      << "The failed symbol lookup did not replace the saved diagnostic";
  linuxBoundary.forceDlsym = false;
  linuxBoundary.forceDlerror = false;
  ncclOsDlclose(library);

  // With no loader error, the internal capture helper must clear stale text.
  linuxBoundary.forceDlerror = true;
  linuxBoundary.syntheticDlerrorPending = false;
  saveDlError();
  EXPECT_STREQ(ncclOsDlerror(), "") << "Saving an absent loader error left stale diagnostic text";

  // Closing NULL is an explicit no-op and must not reach dlclose.
  const int closeCallsBeforeNull = linuxBoundary.dlcloseCalls;
  ncclOsDlclose(nullptr);
  EXPECT_EQ(linuxBoundary.dlcloseCalls, closeCallsBeforeNull) << "Closing NULL unexpectedly called dlclose";

  // NVML success must publish the exact loader handle requested through the fixed soname.
  linuxBoundary.forceDlopen = true;
  linuxBoundary.forceDlerror = true;
  linuxBoundary.dlopenResult = reinterpret_cast<void*>(0x1234);
  ncclOsLibraryHandle nvmlHandle = reinterpret_cast<void*>(0x5678);
  EXPECT_EQ(ncclOsNvmlOpen(&nvmlHandle), ncclSuccess) << "A successful NVML load returned an error";
  EXPECT_EQ(nvmlHandle, linuxBoundary.dlopenResult) << "NVML open lost the loader-owned handle";
  EXPECT_EQ(linuxBoundary.dlopenPath, "libnvidia-ml.so.1") << "NVML open requested the wrong soname";
  ncclOsDlclose(nvmlHandle);

  // NVML failure must return ncclSystemError and clear the caller's stale output.
  linuxBoundary.dlopenResult = nullptr;
  linuxBoundary.dlerrorText = "NVML unavailable";
  nvmlHandle = reinterpret_cast<void*>(0x5678);
  EXPECT_EQ(ncclOsNvmlOpen(&nvmlHandle), ncclSystemError)
      << "A failed NVML load did not return ncclSystemError";
  EXPECT_EQ(nvmlHandle, nullptr) << "A failed NVML load left a stale output handle";
}

/**
 * @brief Verifies process identity, time, allocation, environment, and path wrappers against
 * Linux behavior.
 * @note Side effects: Allocates memory and temporarily creates one process environment variable.
 */
TEST_F(LinuxRuntimeTest, ProcessAndMemory) {
  SCOPED_TRACE("Basic OS wrappers must return the same observable values as their Linux dependencies.");

  // Process and thread identifiers must identify the current test process and thread.
  EXPECT_EQ(ncclOsGetPid(), static_cast<uint64_t>(::getpid())) << "The process identifier changed in transit";
  EXPECT_EQ(ncclOsGetTid(), static_cast<uint64_t>(::syscall(SYS_gettid)))
      << "The thread identifier changed in transit";

  // Local time must be written into and returned through the caller's exact buffer.
  const time_t epoch = 0;
  std::tm expectedTime = {};
  std::tm actualTime = {};
  ASSERT_EQ(::localtime_r(&epoch, &expectedTime), &expectedTime) << "The Linux local-time oracle failed";
  EXPECT_EQ(ncclOsLocaltime(&epoch, &actualTime), &actualTime) << "Local time returned a different buffer";
  EXPECT_EQ(actualTime.tm_year, expectedTime.tm_year) << "Local time changed the year";
  EXPECT_EQ(actualTime.tm_mon, expectedTime.tm_mon) << "Local time changed the month";
  EXPECT_EQ(actualTime.tm_mday, expectedTime.tm_mday) << "Local time changed the day";
  EXPECT_EQ(actualTime.tm_hour, expectedTime.tm_hour) << "Local time changed the hour";
  EXPECT_EQ(actualTime.tm_min, expectedTime.tm_min) << "Local time changed the minute";
  EXPECT_EQ(actualTime.tm_sec, expectedTime.tm_sec) << "Local time changed the second";

  // Page size must match the value reported by the current Linux process.
  const long expectedPageSize = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(expectedPageSize, 0) << "The Linux page-size oracle returned an invalid value";
  EXPECT_EQ(ncclOsGetPageSize(), static_cast<size_t>(expectedPageSize)) << "The page size changed in transit";

  // A valid aligned allocation must satisfy alignment and remain writable for its complete
  // requested size.
  constexpr size_t alignment = 64;
  constexpr size_t allocationSize = 128;
  void* allocation = ncclOsAlignedAlloc(alignment, allocationSize);
  ASSERT_NE(allocation, nullptr) << "A valid aligned allocation returned NULL";
  EXPECT_EQ(reinterpret_cast<uintptr_t>(allocation) % alignment, 0U)
      << "The allocated address does not satisfy the requested alignment";
  std::memset(allocation, 0xa5, allocationSize);
  EXPECT_EQ(static_cast<unsigned char*>(allocation)[allocationSize - 1], 0xa5)
      << "The allocated region was not writable through its final byte";
  ncclOsAlignedFree(allocation);

  // Configuration-file values may fill an absent variable but must not override an existing
  // environment value.
  const std::string variable = "OCCL_LINUX_TEST_ENV_" + std::to_string(::getpid());
  ASSERT_EQ(::unsetenv(variable.c_str()), 0) << "The test could not clear its private environment variable";
  ncclOsSetEnv(variable.c_str(), "from-file");
  ASSERT_NE(::getenv(variable.c_str()), nullptr) << "Setting an absent environment variable produced no value";
  EXPECT_STREQ(::getenv(variable.c_str()), "from-file") << "The environment wrapper stored the wrong value";
  ASSERT_EQ(::setenv(variable.c_str(), "explicit", 1), 0) << "The test could not install its explicit value";
  ncclOsSetEnv(variable.c_str(), "ignored-file-value");
  EXPECT_STREQ(::getenv(variable.c_str()), "explicit") << "The environment wrapper overwrote an existing value";
  EXPECT_EQ(::unsetenv(variable.c_str()), 0) << "The test could not remove its private environment variable";

  // A stable procfs path must resolve, while an absent path must report failure without changing
  // its input string.
  const char existingPath[] = "/proc/self/exe";
  char resolvedPath[PATH_MAX] = {};
  EXPECT_EQ(ncclOsRealpath(existingPath, resolvedPath), resolvedPath)
      << "Resolving /proc/self/exe did not return the caller's output buffer";
  EXPECT_EQ(resolvedPath[0], '/') << "The resolved executable path is not absolute";
  const char missingPath[] = "/definitely/missing/occl-linux-test";
  EXPECT_EQ(ncclOsRealpath(missingPath, resolvedPath), nullptr) << "An absent path unexpectedly resolved";
  EXPECT_STREQ(missingPath, "/definitely/missing/occl-linux-test") << "Realpath modified its input path";
}

/**
 * @brief Verifies stack initialization and file-limit success and error contracts.
 * @note Side effects: Temporarily changes and restores the process default pthread attributes.
 */
TEST_F(LinuxRuntimeTest, InitializationAndLimits) {
  SCOPED_TRACE("Runtime initialization must adjust only requested resources and propagate boundary failures.");

  // Disabled stack adjustment must not touch pthread attributes even when the boundary would fail.
  stackSizeParameter = 0;
  linuxBoundary.pthreadAttrInitResult = EINVAL;
  EXPECT_EQ(ncclOsInitialize(), ncclSuccess) << "Disabled stack adjustment unexpectedly failed initialization";
  EXPECT_EQ(linuxBoundary.pthreadAttrInitCalls, 0) << "Disabled stack adjustment touched pthread attributes";

  // An enabled pthread failure must be returned as ncclSystemError.
  stackSizeParameter = 1;
  EXPECT_EQ(ncclOsInitialize(), ncclSystemError) << "A pthread initialization failure was not propagated";
  EXPECT_EQ(linuxBoundary.pthreadAttrInitCalls, 1) << "Enabled initialization did not inspect pthread defaults";

  // A forced small stack must cause new threads to receive at least the safe eight-megabyte size.
  ResetLinuxBoundary();
  pthread_attr_t originalAttributes;
  pthread_attr_t observedAttributes;
  ASSERT_EQ(::pthread_attr_init(&originalAttributes), 0) << "The test could not initialize saved pthread attributes";
  ASSERT_EQ(::pthread_getattr_default_np(&originalAttributes), 0)
      << "The test could not read the original pthread defaults";
  linuxBoundary.forceStackSize = true;
  linuxBoundary.stackSize = PTHREAD_STACK_MIN;
  const ncclResult_t stackResult = setCpuStackSize();
  const int observedInit = ::pthread_attr_init(&observedAttributes);
  const int observedGet = observedInit == 0 ? ::pthread_getattr_default_np(&observedAttributes) : observedInit;
  size_t observedStackSize = 0;
  const int observedSize = observedGet == 0
                               ? ::pthread_attr_getstacksize(&observedAttributes, &observedStackSize)
                               : observedGet;
  const int restoreResult = ::pthread_setattr_default_np(&originalAttributes);
  if (observedInit == 0) (void)::pthread_attr_destroy(&observedAttributes);
  (void)::pthread_attr_destroy(&originalAttributes);
  EXPECT_EQ(stackResult, ncclSuccess) << "A valid small-stack correction returned an error";
  EXPECT_EQ(observedGet, 0) << "The test could not read corrected pthread defaults";
  EXPECT_EQ(observedSize, 0) << "The test could not read the corrected stack size";
  EXPECT_GE(observedStackSize, 8192U * 1024U) << "New-thread stack size remained below the safe boundary";
  EXPECT_EQ(restoreResult, 0) << "The test could not restore the original pthread defaults";

  // Missing optional glibc support must remain successful and still destroy the local attribute
  // object.
  ResetLinuxBoundary();
  stackSizeParameter = 1;
  linuxBoundary.forceStackSize = true;
  linuxBoundary.stackSize = PTHREAD_STACK_MIN;
  linuxBoundary.forceDlsym = true;
  linuxBoundary.dlsymResult = nullptr;
  EXPECT_EQ(ncclOsInitialize(), ncclSuccess) << "Missing pthread_setattr_default_np was treated as an error";
  EXPECT_EQ(linuxBoundary.pthreadAttrDestroyCalls, 1)
      << "The local pthread attribute was not destroyed after the optional symbol was absent";

  // File-limit success must copy the hard limit into the requested soft limit exactly.
  ResetLinuxBoundary();
  linuxBoundary.forceRlimit = true;
  linuxBoundary.rlimitValue.rlim_cur = 17;
  linuxBoundary.rlimitValue.rlim_max = 43;
  EXPECT_EQ(ncclOsSetFilesLimit(), ncclSuccess) << "A valid file-limit update returned an error";
  EXPECT_EQ(linuxBoundary.setRlimitCalls, 1) << "The file-limit update did not call setrlimit exactly once";
  EXPECT_EQ(linuxBoundary.setRlimitValue.rlim_cur, 43U) << "The soft file limit was not raised to the hard limit";
  EXPECT_EQ(linuxBoundary.setRlimitValue.rlim_max, 43U) << "The hard file limit was unexpectedly changed";

  // Each system-call failure must return ncclSystemError without attempting later work.
  linuxBoundary.failGetRlimit = true;
  const int setCallsBeforeGetFailure = linuxBoundary.setRlimitCalls;
  EXPECT_EQ(ncclOsSetFilesLimit(), ncclSystemError) << "A getrlimit failure was not propagated";
  EXPECT_EQ(linuxBoundary.setRlimitCalls, setCallsBeforeGetFailure)
      << "setrlimit ran after getrlimit had already failed";
  linuxBoundary.failGetRlimit = false;
  linuxBoundary.failSetRlimit = true;
  EXPECT_EQ(ncclOsSetFilesLimit(), ncclSystemError) << "A setrlimit failure was not propagated";
}

/**
 * @brief Creates a socket object with observable non-default fields for mutation checks.
 * @param[in] descriptor Valid or invalid socket descriptor.
 * @param[in] state Initial socket state in the ncclSocketState domain.
 * @return A socket containing descriptor, state, and stable sentinel fields.
 * @note This function has no side effects.
 */
struct ncclSocket MakeLinuxSocket(int descriptor, enum ncclSocketState state) {
  struct ncclSocket socketValue = {};
  socketValue.socketDescriptor = descriptor;
  socketValue.acceptSocketDescriptor = NCCL_INVALID_SOCKET;
  socketValue.errorRetries = 0;
  socketValue.addr.sin.sin_family = AF_INET;
  socketValue.addr.sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  socketValue.addr.sin.sin_port = htons(12345);
  socketValue.asyncFlag = 0;
  socketValue.state = state;
  socketValue.salen = sizeof(struct sockaddr_in);
  socketValue.magic = 0x123456789abcdef0ULL;
  socketValue.type = ncclSocketTypeNetSocket;
  socketValue.customRetry = 0;
  socketValue.finalizeCounter = 3;
  std::memset(socketValue.finalizeBuffer, 0x5a, sizeof(socketValue.finalizeBuffer));
  return socketValue;
}

/**
 * @brief Verifies semantic socket fields that an operation is not expected to change.
 * @param[in] actual Socket observed after the operation.
 * @param[in] expected Snapshot captured before the operation.
 * @param[in] compareDescriptor true when the descriptor must also remain unchanged.
 * @param[in] compareState true when the state must also remain unchanged.
 * @param[in] reason Non-NULL scenario explanation included in failures.
 * @note Side effects: Records GoogleTest failures for changed fields.
 */
void ExpectLinuxSocketPreserved(const struct ncclSocket& actual, const struct ncclSocket& expected,
                                bool compareDescriptor, bool compareState, const char* reason) {
  SCOPED_TRACE(reason);
  if (compareDescriptor) {
    EXPECT_EQ(actual.socketDescriptor, expected.socketDescriptor) << "Descriptor changed";
  }
  EXPECT_EQ(actual.acceptSocketDescriptor, expected.acceptSocketDescriptor) << "Accept descriptor changed";
  EXPECT_EQ(actual.errorRetries, expected.errorRetries) << "Error-retry count changed";
  EXPECT_EQ(actual.addr.sa.sa_family, expected.addr.sa.sa_family) << "Address family changed";
  EXPECT_EQ(actual.addr.sin.sin_addr.s_addr, expected.addr.sin.sin_addr.s_addr) << "IPv4 address changed";
  EXPECT_EQ(actual.addr.sin.sin_port, expected.addr.sin.sin_port) << "IPv4 port changed";
  EXPECT_EQ(actual.abortFlag, expected.abortFlag) << "Abort pointer changed";
  EXPECT_EQ(actual.asyncFlag, expected.asyncFlag) << "Async flag changed";
  if (compareState) {
    EXPECT_EQ(actual.state, expected.state) << "Socket state changed";
  }
  EXPECT_EQ(actual.salen, expected.salen) << "Address length changed";
  EXPECT_EQ(actual.magic, expected.magic) << "Magic value changed";
  EXPECT_EQ(actual.type, expected.type) << "Socket type changed";
  EXPECT_EQ(actual.customRetry, expected.customRetry) << "Custom-retry flag changed";
  EXPECT_EQ(actual.finalizeCounter, expected.finalizeCounter) << "Finalize counter changed";
  for (size_t i = 0; i < sizeof(actual.finalizeBuffer); ++i) {
    EXPECT_EQ(actual.finalizeBuffer[i], expected.finalizeBuffer[i]) << "Finalize byte " << i << " changed";
  }
}

/**
 * @brief Opens a connected IPv4 loopback client and listener for a deterministic accept test.
 * @param[out] listener Valid storage receiving a listening descriptor or -1 on failure.
 * @param[out] client Valid storage receiving a connected client descriptor or -1 on failure.
 * @return true after both descriptors are ready, otherwise false.
 * @note Side effects: Opens up to two descriptors; the caller owns successful outputs.
 */
bool OpenLoopbackConnection(int* listener, int* client) {
  struct sockaddr_in address = {};
  socklen_t addressLength = sizeof(address);
  *listener = ::socket(AF_INET, SOCK_STREAM, 0);
  *client = -1;
  if (*listener < 0) return false;
  int reuse = 1;
  if (::setsockopt(*listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) goto fail;
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(*listener, reinterpret_cast<struct sockaddr*>(&address), sizeof(address)) != 0) goto fail;
  if (::listen(*listener, 1) != 0) goto fail;
  if (::getsockname(*listener, reinterpret_cast<struct sockaddr*>(&address), &addressLength) != 0) goto fail;
  *client = ::socket(AF_INET, SOCK_STREAM, 0);
  if (*client < 0) goto fail;
  if (::connect(*client, reinterpret_cast<struct sockaddr*>(&address), sizeof(address)) != 0) goto fail;
  return true;
fail:
  if (*client >= 0) (void)::close(*client);
  (void)::close(*listener);
  *listener = -1;
  *client = -1;
  return false;
}

/**
 * @brief Verifies descriptor validity, poll selection, socket flags, buffer options, and failure
 * preservation.
 * @note Side effects: Opens and closes local TCP descriptors and mutates the recording boundary.
 */
TEST_F(LinuxSocketTest, FlagsAndPolling) {
  SCOPED_TRACE("Socket validation and flags must affect the descriptor without corrupting the socket object.");

  // Descriptor validity has one exact Linux boundary at minus one.
  EXPECT_FALSE(ncclOsSocketDescriptorIsValid(NCCL_INVALID_SOCKET)) << "The invalid descriptor was accepted";
  EXPECT_TRUE(ncclOsSocketDescriptorIsValid(0)) << "Descriptor zero was rejected";
  struct ncclSocket socketValue = MakeLinuxSocket(NCCL_INVALID_SOCKET, ncclSocketStateInitialized);
  EXPECT_FALSE(ncclOsSocketIsValid(&socketValue)) << "A socket containing an invalid descriptor was accepted";
  socketValue.socketDescriptor = 0;
  EXPECT_TRUE(ncclOsSocketIsValid(&socketValue)) << "A socket containing descriptor zero was rejected";

  // Poll must select the event implied by the operation and forward the configured timeout.
  linuxBoundary.forcePoll = true;
  linuxBoundary.pollResult = 0;
  ncclOsPollSocket(19, NCCL_SOCKET_RECV);
  EXPECT_EQ(linuxBoundary.pollFd, 19) << "Receive polling changed the descriptor";
  EXPECT_EQ(linuxBoundary.pollEvents, POLLIN) << "Receive polling did not request POLLIN";
  EXPECT_EQ(linuxBoundary.pollTimeout, pollTimeoutParameter) << "Receive polling changed the timeout";
  ncclOsPollSocket(23, NCCL_SOCKET_SEND);
  EXPECT_EQ(linuxBoundary.pollFd, 23) << "Send polling changed the descriptor";
  EXPECT_EQ(linuxBoundary.pollEvents, POLLOUT) << "Send polling did not request POLLOUT";

  // SetFlags must reject an invalid descriptor without changing any socket field.
  socketValue = MakeLinuxSocket(NCCL_INVALID_SOCKET, ncclSocketStateInitialized);
  const struct ncclSocket invalidBefore = socketValue;
  EXPECT_EQ(ncclOsSocketSetFlags(&socketValue), ncclInvalidArgument)
      << "SetFlags did not reject an invalid descriptor";
  ExpectLinuxSocketPreserved(socketValue, invalidBefore, true, true,
                             "Invalid SetFlags input must remain unchanged");

  // A valid async TCP descriptor must become nonblocking and receive all enabled options.
  const int descriptor = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(descriptor, 0) << "The test could not create a TCP descriptor";
  socketValue = MakeLinuxSocket(descriptor, ncclSocketStateInitialized);
  socketValue.asyncFlag = 1;
  sendBufferParameter = 4096;
  receiveBufferParameter = 8192;
  const struct ncclSocket validBefore = socketValue;
  ASSERT_EQ(ncclOsSocketSetFlags(&socketValue), ncclSuccess) << "Valid async socket flags returned an error";
  const int flags = ::fcntl(descriptor, F_GETFL);
  EXPECT_NE(flags & O_NONBLOCK, 0) << "An async descriptor did not become nonblocking";
  int noDelay = 0;
  socklen_t optionLength = sizeof(noDelay);
  ASSERT_EQ(::getsockopt(descriptor, IPPROTO_TCP, TCP_NODELAY, &noDelay, &optionLength), 0)
      << "The test could not read TCP_NODELAY";
  EXPECT_EQ(noDelay, 1) << "TCP_NODELAY was not enabled";
  /**
   * @brief Finds the first recorded socket option with a selected identifier.
   * @param[in] option Socket-option identifier to match.
   * @return An iterator to the matching call, or socketOptions.end() when absent.
   * @note This function has no externally visible side effects.
   */
  const auto findOption = [](int option) {
    return std::find_if(linuxBoundary.socketOptions.begin(), linuxBoundary.socketOptions.end(),
                        [option](const SocketOptionCall& call) { return call.option == option; });
  };
  const auto noDelayCall = findOption(TCP_NODELAY);
  ASSERT_NE(noDelayCall, linuxBoundary.socketOptions.end()) << "The required TCP_NODELAY option was not applied";
  EXPECT_EQ(noDelayCall->level, IPPROTO_TCP) << "TCP_NODELAY used the wrong protocol level";
  EXPECT_EQ(noDelayCall->value, 1) << "TCP_NODELAY received the wrong value";
  EXPECT_EQ(noDelayCall->length, sizeof(int)) << "TCP_NODELAY received the wrong value size";
  const auto sendBufferCall = findOption(SO_SNDBUF);
  ASSERT_NE(sendBufferCall, linuxBoundary.socketOptions.end()) << "The configured send-buffer option was not applied";
  EXPECT_EQ(sendBufferCall->level, SOL_SOCKET) << "SO_SNDBUF used the wrong protocol level";
  EXPECT_EQ(sendBufferCall->value, sendBufferParameter) << "SO_SNDBUF received the wrong value";
  EXPECT_EQ(sendBufferCall->length, sizeof(int)) << "SO_SNDBUF received the wrong value size";
  const auto receiveBufferCall = findOption(SO_RCVBUF);
  ASSERT_NE(receiveBufferCall, linuxBoundary.socketOptions.end())
      << "The configured receive-buffer option was not applied";
  EXPECT_EQ(receiveBufferCall->level, SOL_SOCKET) << "SO_RCVBUF used the wrong protocol level";
  EXPECT_EQ(receiveBufferCall->value, receiveBufferParameter) << "SO_RCVBUF received the wrong value";
  EXPECT_EQ(receiveBufferCall->length, sizeof(int)) << "SO_RCVBUF received the wrong value size";
  ExpectLinuxSocketPreserved(socketValue, validBefore, true, true,
                             "SetFlags must change descriptor flags, not socket fields");

  // A fcntl failure must return ncclSystemError and preserve the socket object.
  linuxBoundary.failFcntlCommand = F_GETFL;
  linuxBoundary.failFcntlError = EBADF;
  EXPECT_EQ(ncclOsSocketSetFlags(&socketValue), ncclSystemError) << "A fcntl failure was not propagated";
  ExpectLinuxSocketPreserved(socketValue, validBefore, true, true,
                             "Failed SetFlags must preserve socket fields");
  linuxBoundary.failFcntlCommand = -1;

  // A required TCP_NODELAY failure must also return ncclSystemError.
  linuxBoundary.failSocketOption = TCP_NODELAY;
  linuxBoundary.failSocketOptionError = ENOPROTOOPT;
  EXPECT_EQ(ncclOsSocketSetFlags(&socketValue), ncclSystemError)
      << "A TCP_NODELAY failure was not propagated";
  ExpectLinuxSocketPreserved(socketValue, validBefore, true, true,
                             "Socket-option failure must preserve socket fields");
  (void)::close(descriptor);
}

/**
 * @brief Verifies accept success, transient and fatal errors, retry exhaustion, and accepted
 * socket reset.
 * @note Side effects: Creates and closes local TCP connections and mutates boundary records.
 */
TEST_F(LinuxSocketTest, AcceptAndReset) {
  SCOPED_TRACE("Accept must update only its documented outputs and classify Linux errors correctly.");

  // A real loopback connection must supply the accepted descriptor, peer address, and accepted
  // state.
  int listener = -1;
  int client = -1;
  ASSERT_TRUE(OpenLoopbackConnection(&listener, &client)) << "The test could not create a loopback connection";
  struct ncclSocket socketValue = MakeLinuxSocket(NCCL_INVALID_SOCKET, ncclSocketStateAccepting);
  socketValue.acceptSocketDescriptor = listener;
  ASSERT_EQ(ncclOsSocketTryAccept(&socketValue), ncclSuccess) << "A ready loopback connection failed to accept";
  EXPECT_TRUE(ncclOsSocketIsValid(&socketValue)) << "Accept did not publish a valid descriptor";
  EXPECT_EQ(socketValue.state, ncclSocketStateAccepted) << "Accept did not enter the accepted state";
  EXPECT_EQ(socketValue.addr.sa.sa_family, AF_INET) << "Accept did not publish the IPv4 peer address";
  (void)::close(socketValue.socketDescriptor);
  (void)::close(client);
  (void)::close(listener);

  // EAGAIN is normal asynchronous absence and must not mutate state or retry count.
  socketValue = MakeLinuxSocket(NCCL_INVALID_SOCKET, ncclSocketStateAccepting);
  linuxBoundary.forceAccept = true;
  linuxBoundary.acceptResult = -1;
  linuxBoundary.acceptError = EAGAIN;
  const struct ncclSocket eagainBefore = socketValue;
  EXPECT_EQ(ncclOsSocketTryAccept(&socketValue), ncclSuccess) << "EAGAIN was treated as an accept failure";
  EXPECT_EQ(socketValue.errorRetries, eagainBefore.errorRetries) << "EAGAIN consumed a network-error retry";
  ExpectLinuxSocketPreserved(socketValue, eagainBefore, true, true,
                             "EAGAIN must preserve accept state");

  // Retryable network errors must increment the counter and fail exactly at the configured limit.
  linuxBoundary.acceptError = EHOSTUNREACH;
  retryCountParameter = 2;
  EXPECT_EQ(ncclOsSocketTryAccept(&socketValue), ncclSuccess)
      << "The first retryable accept error failed too early";
  EXPECT_EQ(socketValue.errorRetries, 1) << "The first retryable accept error stored the wrong count";
  EXPECT_EQ(ncclOsSocketTryAccept(&socketValue), ncclSystemError)
      << "Accept did not fail at the configured retry limit";
  EXPECT_EQ(socketValue.errorRetries, 2) << "Retry exhaustion stored the wrong attempt count";

  // A non-retryable accept error must fail immediately without incrementing the retry counter.
  socketValue = MakeLinuxSocket(NCCL_INVALID_SOCKET, ncclSocketStateAccepting);
  linuxBoundary.acceptError = EBADF;
  EXPECT_EQ(ncclOsSocketTryAccept(&socketValue), ncclSystemError)
      << "A fatal accept error did not return ncclSystemError";
  EXPECT_EQ(socketValue.errorRetries, 0) << "A fatal accept error incorrectly consumed retries";

  // ResetAccept must close the spurious descriptor and reset only the accepted-handshake fields.
  int pair[2] = {-1, -1};
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, pair), 0) << "The test could not create reset descriptors";
  socketValue = MakeLinuxSocket(pair[0], ncclSocketStateAccepted);
  const struct ncclSocket resetBefore = socketValue;
  ncclOsSocketResetAccept(&socketValue);
  struct ncclSocket resetExpected = resetBefore;
  resetExpected.socketDescriptor = NCCL_INVALID_SOCKET;
  resetExpected.state = ncclSocketStateAccepting;
  resetExpected.finalizeCounter = 0;
  ExpectLinuxSocketPreserved(socketValue, resetExpected, true, true,
                             "ResetAccept must change only descriptor, state, and handshake progress");
  errno = 0;
  EXPECT_EQ(::fcntl(pair[0], F_GETFD), -1) << "ResetAccept did not close the spurious descriptor";
  EXPECT_EQ(errno, EBADF) << "The reset descriptor failed for an unexpected reason";
  (void)::close(pair[1]);
}

/**
 * @brief Verifies descriptor replacement and every connect state and result class.
 * @note Side effects: Opens and closes local TCP descriptors and mutates the recording boundary.
 */
TEST_F(LinuxSocketTest, DescriptorAndConnect) {
  SCOPED_TRACE("Descriptor reset and connection polling must preserve descriptors and classify states exactly.");

  // Resetting an absent descriptor must create a usable TCP descriptor and apply socket flags.
  struct ncclSocket socketValue = MakeLinuxSocket(NCCL_INVALID_SOCKET, ncclSocketStateConnecting);
  socketValue.asyncFlag = 1;
  const struct ncclSocket absentBefore = socketValue;
  ASSERT_EQ(ncclOsSocketResetFd(&socketValue), ncclSuccess) << "Resetting an absent descriptor returned an error";
  ASSERT_TRUE(ncclOsSocketIsValid(&socketValue)) << "Resetting an absent descriptor produced no descriptor";
  EXPECT_NE(::fcntl(socketValue.socketDescriptor, F_GETFL) & O_NONBLOCK, 0)
      << "The replacement descriptor did not receive async flags";
  ExpectLinuxSocketPreserved(socketValue, absentBefore, false, true,
                             "Resetting an absent descriptor must change only the descriptor");
  (void)::close(socketValue.socketDescriptor);

  // Resetting a live descriptor must preserve its externally visible descriptor number through
  // dup2.
  const int originalDescriptor = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(originalDescriptor, 0) << "The test could not create a descriptor to replace";
  socketValue = MakeLinuxSocket(originalDescriptor, ncclSocketStateConnecting);
  const struct ncclSocket liveBefore = socketValue;
  ASSERT_EQ(ncclOsSocketResetFd(&socketValue), ncclSuccess) << "Replacing a live descriptor returned an error";
  EXPECT_EQ(socketValue.socketDescriptor, originalDescriptor) << "Descriptor replacement changed its public number";
  EXPECT_NE(::fcntl(originalDescriptor, F_GETFD), -1) << "Descriptor replacement left the public descriptor closed";
  ExpectLinuxSocketPreserved(socketValue, liveBefore, true, true,
                             "Replacing a live descriptor must preserve every socket field");
  (void)::close(originalDescriptor);

  // Socket creation failure must return ncclSystemError and retain an absent descriptor.
  socketValue = MakeLinuxSocket(NCCL_INVALID_SOCKET, ncclSocketStateConnecting);
  const struct ncclSocket socketFailureBefore = socketValue;
  linuxBoundary.forceSocketFailure = true;
  linuxBoundary.socketError = EMFILE;
  EXPECT_EQ(ncclOsSocketResetFd(&socketValue), ncclSystemError)
      << "A replacement socket failure was not propagated";
  ExpectLinuxSocketPreserved(socketValue, socketFailureBefore, true, true,
                             "A replacement socket failure must preserve every socket field");
  linuxBoundary.forceSocketFailure = false;

  // dup2 failure must keep the old descriptor usable and close the temporary replacement.
  const int retainedDescriptor = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(retainedDescriptor, 0) << "The test could not create a retained descriptor";
  socketValue = MakeLinuxSocket(retainedDescriptor, ncclSocketStateConnecting);
  const struct ncclSocket dupFailureBefore = socketValue;
  linuxBoundary.forceDup2Failure = true;
  linuxBoundary.dup2Error = EMFILE;
  EXPECT_EQ(ncclOsSocketResetFd(&socketValue), ncclSystemError) << "A dup2 failure was not propagated";
  ExpectLinuxSocketPreserved(socketValue, dupFailureBefore, true, true,
                             "A dup2 failure must preserve every socket field");
  EXPECT_NE(::fcntl(retainedDescriptor, F_GETFD), -1) << "A dup2 failure closed the original descriptor";
  ASSERT_GE(linuxBoundary.lastCreatedSocket, 0) << "The dup2 scenario created no temporary descriptor";
  errno = 0;
  EXPECT_EQ(::fcntl(linuxBoundary.lastCreatedSocket, F_GETFD), -1)
      << "A dup2 failure leaked the temporary descriptor";
  EXPECT_EQ(errno, EBADF) << "The temporary descriptor probe failed for an unexpected reason";
  linuxBoundary.forceDup2Failure = false;
  (void)::close(retainedDescriptor);

  // The internal classifier must implement connected, polling, retry, remote, and system-error
  // outcomes.
  socketValue = MakeLinuxSocket(NCCL_INVALID_SOCKET, ncclSocketStateConnecting);
  struct ncclSocket successfulConnectExpected = socketValue;
  successfulConnectExpected.state = ncclSocketStateConnected;
  EXPECT_EQ(socketConnectCheck(&socketValue, 0, "test"), ncclSuccess)
      << "A successful connect status returned an error";
  ExpectLinuxSocketPreserved(socketValue, successfulConnectExpected, true, true,
                             "Successful classification must change only the connection state");
  EXPECT_EQ(socketConnectCheck(&socketValue, EINPROGRESS, "test"), ncclSuccess)
      << "EINPROGRESS returned an error";
  EXPECT_EQ(socketValue.state, ncclSocketStateConnectPolling) << "EINPROGRESS selected the wrong state";
  socketValue = MakeLinuxSocket(NCCL_INVALID_SOCKET, ncclSocketStateConnecting);
  socketValue.customRetry = 1;
  EXPECT_EQ(socketConnectCheck(&socketValue, ECONNREFUSED, "test"), ncclSuccess)
      << "A retryable connection error failed before its limit";
  EXPECT_EQ(socketValue.state, ncclSocketStateConnecting) << "A retryable error selected the wrong state";
  EXPECT_TRUE(ncclOsSocketIsValid(&socketValue)) << "A retryable error did not replace the descriptor";
  (void)::close(socketValue.socketDescriptor);
  socketValue = MakeLinuxSocket(NCCL_INVALID_SOCKET, ncclSocketStateConnecting);
  socketValue.errorRetries = retryCountParameter;
  EXPECT_EQ(socketConnectCheck(&socketValue, ECONNREFUSED, "test"), ncclRemoteError)
      << "Retry exhaustion did not return ncclRemoteError";
  EXPECT_EQ(socketValue.state, ncclSocketStateError) << "Retry exhaustion did not enter the error state";
  socketValue = MakeLinuxSocket(NCCL_INVALID_SOCKET, ncclSocketStateConnecting);
  EXPECT_EQ(socketConnectCheck(&socketValue, EBADF, "test"), ncclSystemError)
      << "A fatal connection error did not return ncclSystemError";
  EXPECT_EQ(socketValue.state, ncclSocketStateError) << "A fatal connection error did not enter the error state";

  // StartConnect must translate the real connect result through the same public states and
  // statuses.
  linuxBoundary.forceConnect = true;
  socketValue = MakeLinuxSocket(37, ncclSocketStateConnecting);
  struct ncclSocket successfulStartExpected = socketValue;
  successfulStartExpected.state = ncclSocketStateConnected;
  linuxBoundary.connectResult = 0;
  EXPECT_EQ(ncclOsSocketStartConnect(&socketValue), ncclSuccess) << "Successful StartConnect returned an error";
  ExpectLinuxSocketPreserved(socketValue, successfulStartExpected, true, true,
                             "Successful StartConnect must change only the connection state");
  linuxBoundary.connectResult = -1;
  linuxBoundary.connectError = EINPROGRESS;
  EXPECT_EQ(ncclOsSocketStartConnect(&socketValue), ncclSuccess) << "StartConnect rejected EINPROGRESS";
  EXPECT_EQ(socketValue.state, ncclSocketStateConnectPolling) << "StartConnect mishandled EINPROGRESS";
  socketValue.errorRetries = retryCountParameter;
  linuxBoundary.connectError = ECONNREFUSED;
  EXPECT_EQ(ncclOsSocketStartConnect(&socketValue), ncclRemoteError)
      << "StartConnect retry exhaustion did not return ncclRemoteError";
  linuxBoundary.connectError = EBADF;
  EXPECT_EQ(ncclOsSocketStartConnect(&socketValue), ncclSystemError)
      << "StartConnect fatal failure did not return ncclSystemError";

  // PollConnect must preserve state for no event and EINTR, then classify socket completion and
  // failures.
  linuxBoundary.forcePoll = true;
  socketValue = MakeLinuxSocket(41, ncclSocketStateConnectPolling);
  const struct ncclSocket pollingBefore = socketValue;
  linuxBoundary.pollResult = 0;
  EXPECT_EQ(ncclOsSocketPollConnect(&socketValue), ncclSuccess) << "Poll timeout returned an error";
  ExpectLinuxSocketPreserved(socketValue, pollingBefore, true, true, "Poll timeout must preserve socket state");
  linuxBoundary.pollResult = -1;
  linuxBoundary.pollError = EINTR;
  EXPECT_EQ(ncclOsSocketPollConnect(&socketValue), ncclSuccess) << "Interrupted polling returned an error";
  ExpectLinuxSocketPreserved(socketValue, pollingBefore, true, true, "EINTR must preserve polling state");
  linuxBoundary.pollError = EBADF;
  EXPECT_EQ(ncclOsSocketPollConnect(&socketValue), ncclSystemError)
      << "A fatal poll failure did not return ncclSystemError";
  linuxBoundary.pollResult = 1;
  linuxBoundary.forceGetSockOpt = true;
  linuxBoundary.getSockOptResult = 0;
  linuxBoundary.socketPendingError = 0;
  EXPECT_EQ(ncclOsSocketPollConnect(&socketValue), ncclSuccess)
      << "A completed connection poll returned an error";
  EXPECT_EQ(socketValue.state, ncclSocketStateConnected) << "Completed polling selected the wrong state";
  socketValue = MakeLinuxSocket(41, ncclSocketStateConnectPolling);
  socketValue.errorRetries = retryCountParameter;
  linuxBoundary.socketPendingError = ECONNREFUSED;
  EXPECT_EQ(ncclOsSocketPollConnect(&socketValue), ncclRemoteError)
      << "A polled remote refusal did not return ncclRemoteError";
  linuxBoundary.getSockOptResult = -1;
  linuxBoundary.getSockOptError = EBADF;
  EXPECT_EQ(ncclOsSocketPollConnect(&socketValue), ncclSystemError)
      << "A getsockopt failure did not return ncclSystemError";
}

/**
 * @brief Verifies synchronous and asynchronous transfer progress, closure, transient errors,
 * abort, and close.
 * @note Side effects: Transfers bytes through and closes local sockets and mutates boundary
 * records.
 */
TEST_F(LinuxSocketTest, ProgressAndClose) {
  SCOPED_TRACE("Socket progress must update only offset, closed, and receive data according to the transfer result.");

  // Real send and receive operations must transfer exact bytes without changing the send buffer
  // or socket fields.
  int pair[2] = {-1, -1};
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, pair), 0) << "The test could not create transfer descriptors";
  struct ncclSocket socketValue = MakeLinuxSocket(pair[0], ncclSocketStateReady);
  const struct ncclSocket socketBefore = socketValue;
  std::array<char, 5> sent = {'h', 'e', 'l', 'l', 'o'};
  const auto sentBefore = sent;
  int offset = 0;
  int closed = 9;
  ASSERT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_SEND, &socketValue, sent.data(), sent.size(), &offset, 1,
                                   &closed), ncclSuccess)
      << "A real local send returned an error";
  EXPECT_EQ(offset, static_cast<int>(sent.size())) << "The send stored the wrong final offset";
  EXPECT_EQ(closed, 0) << "The send incorrectly reported peer closure";
  EXPECT_EQ(sent, sentBefore) << "The send modified its input buffer";
  std::array<char, 5> peerReceived = {};
  ASSERT_EQ(::recv(pair[1], peerReceived.data(), peerReceived.size(), 0),
            static_cast<ssize_t>(peerReceived.size()))
      << "The peer did not receive the complete send buffer";
  EXPECT_EQ(peerReceived, sent) << "The peer received different send data";
  ExpectLinuxSocketPreserved(socketValue, socketBefore, true, true, "Send progress must preserve socket fields");

  ASSERT_EQ(::send(pair[1], sent.data(), sent.size(), 0), static_cast<ssize_t>(sent.size()))
      << "The peer could not prepare receive data";
  std::array<char, 5> received = {};
  offset = 0;
  closed = 9;
  ASSERT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, &socketValue, received.data(), received.size(), &offset, 1,
                                   &closed), ncclSuccess)
      << "A real local receive returned an error";
  EXPECT_EQ(received, sent) << "The receive wrote different data";
  EXPECT_EQ(offset, static_cast<int>(received.size())) << "The receive stored the wrong final offset";
  EXPECT_EQ(closed, 0) << "The receive incorrectly reported peer closure";
  ExpectLinuxSocketPreserved(socketValue, socketBefore, true, true, "Receive progress must preserve socket fields");

  // Blocking progress must consume repeated partial transfers until the requested size is complete.
  linuxBoundary.sendSteps.push_back({2, 0});
  linuxBoundary.sendSteps.push_back({3, 0});
  const int sendCallsBeforePartials = linuxBoundary.sendCalls;
  offset = 0;
  closed = 9;
  ASSERT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_SEND, &socketValue, sent.data(), sent.size(), &offset, 1,
                                   &closed), ncclSuccess)
      << "Valid blocking partial transfers returned an error";
  EXPECT_EQ(offset, static_cast<int>(sent.size())) << "Blocking partial transfers did not reach the requested size";
  EXPECT_EQ(linuxBoundary.sendCalls - sendCallsBeforePartials, 2)
      << "Blocking progress did not continue after a positive partial transfer";

  // Async progress must perform at most one transfer even when bytes remain.
  socketValue.asyncFlag = 1;
  linuxBoundary.sendSteps.push_back({2, 0});
  const int sendCallsBeforeAsync = linuxBoundary.sendCalls;
  offset = 0;
  ASSERT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_SEND, &socketValue, sent.data(), sent.size(), &offset, 1,
                                   &closed), ncclSuccess)
      << "A valid async partial transfer returned an error";
  EXPECT_EQ(offset, 2) << "Async progress did not retain its partial offset";
  EXPECT_EQ(linuxBoundary.sendCalls - sendCallsBeforeAsync, 1) << "Async progress performed more than one transfer";
  socketValue.asyncFlag = 0;

  // EOF and connection-reset classes are successful closure notifications.
  linuxBoundary.recvSteps.push_back({0, 0});
  offset = 0;
  closed = 9;
  EXPECT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, &socketValue, received.data(), received.size(), &offset, 0,
                                   &closed), ncclSuccess)
      << "Receive EOF returned an error";
  EXPECT_EQ(closed, 1) << "Receive EOF did not report closure";
  linuxBoundary.sendSteps.push_back({-1, EPIPE});
  closed = 9;
  EXPECT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_SEND, &socketValue, sent.data(), sent.size(), &offset, 0,
                                   &closed), ncclSuccess)
      << "EPIPE was not translated into successful closure";
  EXPECT_EQ(closed, 1) << "EPIPE did not report closure";
  linuxBoundary.recvSteps.push_back({-1, ECONNRESET});
  closed = 9;
  EXPECT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, &socketValue, received.data(), received.size(), &offset, 0,
                                   &closed), ncclSuccess)
      << "ECONNRESET was not translated into successful closure";
  EXPECT_EQ(closed, 1) << "ECONNRESET did not report closure";

  // EAGAIN must preserve progress, while an unrelated hard error must return ncclRemoteError.
  linuxBoundary.recvSteps.push_back({-1, EAGAIN});
  offset = 1;
  closed = 9;
  EXPECT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, &socketValue, received.data(), received.size(), &offset, 0,
                                   &closed), ncclSuccess)
      << "EAGAIN was treated as a remote failure";
  EXPECT_EQ(offset, 1) << "EAGAIN changed the receive offset";
  EXPECT_EQ(closed, 0) << "EAGAIN incorrectly reported closure";
  linuxBoundary.recvSteps.push_back({-1, EIO});
  EXPECT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, &socketValue, received.data(), received.size(), &offset, 0,
                                   &closed), ncclRemoteError)
      << "A hard receive failure did not return ncclRemoteError";
  EXPECT_EQ(offset, 1) << "A hard receive failure changed the offset";
  EXPECT_EQ(closed, 0) << "A hard receive failure incorrectly reported closure";

  // An observed abort must return ncclInternalError after leaving offset unchanged.
  volatile uint32_t abortFlag = 1;
  socketValue.abortFlag = &abortFlag;
  linuxBoundary.sendSteps.push_back({0, 0});
  offset = 0;
  EXPECT_EQ(ncclOsSocketProgressOpt(NCCL_SOCKET_SEND, &socketValue, sent.data(), sent.size(), &offset, 1,
                                   &closed), ncclInternalError)
      << "An active abort flag did not return ncclInternalError";
  EXPECT_EQ(offset, 0) << "An aborted zero-byte transfer changed the offset";
  EXPECT_EQ(closed, 0) << "An aborted transfer incorrectly reported closure";
  EXPECT_EQ(sent, sentBefore) << "An aborted transfer modified its send buffer";
  socketValue.abortFlag = nullptr;

  // Closing NULL and an invalid descriptor is safe; a live descriptor must be shut down and
  // invalidated.
  EXPECT_EQ(ncclSocketClose(nullptr, false), ncclSuccess) << "Closing a NULL socket returned an error";
  struct ncclSocket invalid = MakeLinuxSocket(NCCL_INVALID_SOCKET, ncclSocketStateReady);
  EXPECT_EQ(ncclSocketClose(&invalid, false), ncclSuccess) << "Closing an invalid descriptor returned an error";
  EXPECT_EQ(invalid.state, ncclSocketStateClosed) << "Invalid close did not enter the closed state";
  EXPECT_EQ(invalid.socketDescriptor, NCCL_INVALID_SOCKET) << "Invalid close changed the descriptor incorrectly";
  struct ncclSocket live = MakeLinuxSocket(pair[0], ncclSocketStateReady);
  EXPECT_EQ(ncclSocketClose(&live, false), ncclSuccess) << "Closing a live descriptor returned an error";
  EXPECT_EQ(live.state, ncclSocketStateClosed) << "Live close did not enter the closed state";
  EXPECT_EQ(live.socketDescriptor, NCCL_INVALID_SOCKET) << "Live close retained its descriptor";
  errno = 0;
  EXPECT_EQ(::fcntl(pair[0], F_GETFD), -1) << "Live close did not close the descriptor";
  EXPECT_EQ(errno, EBADF) << "The closed-descriptor probe failed for an unexpected reason";
  pair[0] = -1;

  // wait=true must keep requesting receive progress until closure is reported.
  int waitPair[2] = {-1, -1};
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, waitPair), 0) << "The test could not create close-wait descriptors";
  live = MakeLinuxSocket(waitPair[0], ncclSocketStateReady);
  linuxBoundary.closeAfterProgressCalls = 2;
  EXPECT_EQ(ncclSocketClose(&live, true), ncclSuccess) << "Graceful close-wait returned an error";
  EXPECT_EQ(linuxBoundary.closeProgressCalls, 2) << "Graceful close stopped before closure was reported";
  EXPECT_EQ(live.state, ncclSocketStateClosed) << "Graceful close did not enter the closed state";
  (void)::close(waitPair[1]);
  (void)::close(pair[1]);
}

/**
 * @brief Owns a fixed synthetic ifaddrs chain whose pointers remain stable throughout one test.
 * @note Side effects: Construction clears the private interface, address, mask, and name storage.
 */
class SyntheticInterfaceList {
 public:
  /**
   * @brief Initializes an empty synthetic interface list.
   * @note Side effects: Clears all list storage.
   */
  SyntheticInterfaceList() = default;

  /**
   * @brief Adds one IPv4 interface with a parseable address and network mask.
   * @param[in] name Valid non-NULL interface name shorter than IFNAMSIZ.
   * @param[in] address Valid numeric IPv4 address.
   * @param[in] mask Valid numeric IPv4 network mask.
   * @param[in] flags Interface flags, normally including IFF_RUNNING for eligible entries.
   * @return true when storage and both addresses are valid, otherwise false.
   * @note Side effects: Appends one node on success.
   */
  bool AddIpv4(const char* name, const char* address, const char* mask, unsigned int flags) {
    if (count_ == nodes_.size() || std::strlen(name) >= IFNAMSIZ) return false;
    struct sockaddr_in parsedAddress = {};
    struct sockaddr_in parsedMask = {};
    parsedAddress.sin_family = AF_INET;
    parsedMask.sin_family = AF_INET;
    if (::inet_pton(AF_INET, address, &parsedAddress.sin_addr) != 1 ||
        ::inet_pton(AF_INET, mask, &parsedMask.sin_addr) != 1) {
      return false;
    }
    std::memcpy(&addresses_[count_], &parsedAddress, sizeof(parsedAddress));
    std::memcpy(&masks_[count_], &parsedMask, sizeof(parsedMask));
    AppendNode(name, flags, &addresses_[count_], &masks_[count_]);
    return true;
  }

  /**
   * @brief Adds one IPv6 interface with a parseable address, network mask, and scope.
   * @param[in] name Valid non-NULL interface name shorter than IFNAMSIZ.
   * @param[in] address Valid numeric IPv6 address.
   * @param[in] mask Valid numeric IPv6 network mask.
   * @param[in] scope Scope identifier in [0, UINT32_MAX].
   * @param[in] flags Interface flags, normally including IFF_RUNNING for eligible entries.
   * @return true when storage and both addresses are valid, otherwise false.
   * @note Side effects: Appends one node on success.
   */
  bool AddIpv6(const char* name, const char* address, const char* mask, uint32_t scope,
               unsigned int flags) {
    if (count_ == nodes_.size() || std::strlen(name) >= IFNAMSIZ) return false;
    struct sockaddr_in6 parsedAddress = {};
    struct sockaddr_in6 parsedMask = {};
    parsedAddress.sin6_family = AF_INET6;
    parsedAddress.sin6_scope_id = scope;
    parsedMask.sin6_family = AF_INET6;
    if (::inet_pton(AF_INET6, address, &parsedAddress.sin6_addr) != 1 ||
        ::inet_pton(AF_INET6, mask, &parsedMask.sin6_addr) != 1) {
      return false;
    }
    std::memcpy(&addresses_[count_], &parsedAddress, sizeof(parsedAddress));
    std::memcpy(&masks_[count_], &parsedMask, sizeof(parsedMask));
    AppendNode(name, flags, &addresses_[count_], &masks_[count_]);
    return true;
  }

  /**
   * @brief Adds an interface with no address or with a non-IPv4/IPv6 address.
   * @param[in] name Valid non-NULL interface name shorter than IFNAMSIZ.
   * @param[in] family AF_UNSPEC for no address, or the family stored in a synthetic address.
   * @param[in] flags Interface flags.
   * @return true when fixed storage is available, otherwise false.
   * @note Side effects: Appends one node on success.
   */
  bool AddOther(const char* name, int family, unsigned int flags) {
    if (count_ == nodes_.size() || std::strlen(name) >= IFNAMSIZ) return false;
    if (family == AF_UNSPEC) {
      AppendNode(name, flags, nullptr, nullptr);
    } else {
      addresses_[count_].ss_family = family;
      masks_[count_].ss_family = family;
      AppendNode(name, flags, &addresses_[count_], &masks_[count_]);
    }
    return true;
  }

  /**
   * @brief Returns the first node or NULL for an empty list.
   * @return Stable list head owned by this object.
   * @note This function has no side effects.
   */
  struct ifaddrs* Head() {
    return count_ == 0 ? nullptr : &nodes_[0];
  }

 private:
  /**
   * @brief Wires one already validated node into the stable list storage.
   * @param[in] name Valid non-NULL interface name shorter than IFNAMSIZ.
   * @param[in] flags Interface flags.
   * @param[in] address NULL or stable address storage.
   * @param[in] mask NULL or stable mask storage.
   * @note Side effects: Initializes one node, links the previous node, and increments the list
   * size.
   */
  void AppendNode(const char* name, unsigned int flags, struct sockaddr_storage* address,
                  struct sockaddr_storage* mask) {
    std::snprintf(names_[count_].data(), names_[count_].size(), "%s", name);
    nodes_[count_].ifa_name = names_[count_].data();
    nodes_[count_].ifa_flags = flags;
    nodes_[count_].ifa_addr = reinterpret_cast<struct sockaddr*>(address);
    nodes_[count_].ifa_netmask = reinterpret_cast<struct sockaddr*>(mask);
    if (count_ > 0) nodes_[count_ - 1].ifa_next = &nodes_[count_];
    count_++;
  }

  std::array<struct ifaddrs, 8> nodes_ = {};
  std::array<struct sockaddr_storage, 8> addresses_ = {};
  std::array<struct sockaddr_storage, 8> masks_ = {};
  std::array<std::array<char, IFNAMSIZ>, 8> names_ = {};
  size_t count_ = 0;
};

/**
 * @brief Verifies interface filtering, family selection, deduplication, limits, and enumeration
 * errors.
 * @note Side effects: Publishes synthetic interface lists through the recording getifaddrs
 * boundary.
 */
TEST_F(LinuxInterfaceTest, Enumeration) {
  SCOPED_TRACE("Interface discovery must select only eligible distinct interfaces and preserve unused outputs.");

  // Exact IPv4 selection must skip missing, unsupported, down, loopback-v6, duplicate, and
  // unmatched entries.
  SyntheticInterfaceList interfaces;
  ASSERT_TRUE(interfaces.AddOther("missing", AF_UNSPEC, IFF_RUNNING)) << "Could not add the missing-address entry";
  ASSERT_TRUE(interfaces.AddOther("unix0", AF_UNIX, IFF_RUNNING)) << "Could not add the unsupported-family entry";
  ASSERT_TRUE(interfaces.AddIpv4("down0", "10.1.0.1", "255.255.255.0", IFF_UP))
      << "Could not add the non-running entry";
  ASSERT_TRUE(interfaces.AddIpv6("loop6", "::1", "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", 0,
                                 IFF_RUNNING))
      << "Could not add the IPv6 loopback entry";
  ASSERT_TRUE(interfaces.AddIpv4("eth0", "10.2.3.4", "255.255.255.0", IFF_RUNNING))
      << "Could not add the selected IPv4 entry";
  ASSERT_TRUE(interfaces.AddIpv4("eth0", "10.2.3.5", "255.255.255.0", IFF_RUNNING))
      << "Could not add the duplicate IPv4 entry";
  ASSERT_TRUE(interfaces.AddIpv4("eth1", "10.3.4.5", "255.255.255.0", IFF_RUNNING))
      << "Could not add the unmatched IPv4 entry";
  ASSERT_TRUE(interfaces.AddIpv6("ib6", "2001:db8::8", "ffff:ffff:ffff:ffff::", 6, IFF_RUNNING))
      << "Could not add the selectable IPv6 entry";
  linuxBoundary.interfaceHead = interfaces.Head();
  linuxBoundary.includedInterfaceNames = {"eth0"};
  const char exactExpression[] = "=eth0";
  std::array<char, 2 * MAX_IF_NAME_SIZE> names;
  names.fill(static_cast<char>(0x5a));
  const auto namesBefore = names;
  std::array<union ncclSocketAddress, 2> addresses;
  std::memset(addresses.data(), 0xa5, sizeof(addresses));
  const auto addressesBefore = addresses;
  int found = 7;
  ASSERT_EQ(ncclOsFindInterfaces(exactExpression, names.data(), addresses.data(), AF_INET,
                                MAX_IF_NAME_SIZE, addresses.size(), &found), ncclSuccess)
      << "A valid exact IPv4 enumeration returned an error";
  EXPECT_EQ(found, 1) << "Exact IPv4 enumeration returned the wrong number of interfaces";
  EXPECT_STREQ(names.data(), "eth0") << "Exact IPv4 enumeration returned the wrong name";
  EXPECT_EQ(addresses[0].sa.sa_family, AF_INET) << "Exact IPv4 enumeration returned the wrong family";
  EXPECT_EQ(addresses[0].sin.sin_addr.s_addr, htonl(0x0a020304U))
      << "Exact IPv4 enumeration returned the wrong address";
  EXPECT_EQ(linuxBoundary.parsedInterfaceExpression, "eth0") << "The exact marker was not removed before parsing";
  EXPECT_TRUE(linuxBoundary.lastInterfaceMatchExact) << "Exact matching was not requested";
  EXPECT_STREQ(exactExpression, "=eth0") << "Enumeration modified its input expression";
  EXPECT_EQ(std::memcmp(names.data() + MAX_IF_NAME_SIZE, namesBefore.data() + MAX_IF_NAME_SIZE,
                        MAX_IF_NAME_SIZE), 0)
      << "Enumeration modified an unused name slot";
  EXPECT_EQ(std::memcmp(&addresses[1], &addressesBefore[1], sizeof(addresses[1])), 0)
      << "Enumeration modified an unused address slot";
  EXPECT_EQ(linuxBoundary.freeIfAddrsCalls, 1) << "Enumeration did not release its interface list";

  // A NULL filter with an IPv6 family must skip loopback and publish the first eligible complete
  // IPv6 address.
  ResetLinuxBoundary();
  linuxBoundary.interfaceHead = interfaces.Head();
  names.fill(0);
  std::memset(addresses.data(), 0, sizeof(addresses));
  found = 7;
  ASSERT_EQ(ncclOsFindInterfaces(nullptr, names.data(), addresses.data(), AF_INET6, MAX_IF_NAME_SIZE, 1,
                                &found), ncclSuccess)
      << "Unfiltered IPv6 enumeration returned an error";
  EXPECT_EQ(found, 1) << "Unfiltered IPv6 enumeration returned the wrong number of interfaces";
  EXPECT_STREQ(names.data(), "ib6") << "Unfiltered IPv6 enumeration returned the wrong interface";
  EXPECT_EQ(addresses[0].sa.sa_family, AF_INET6) << "Unfiltered IPv6 enumeration returned the wrong family";
  EXPECT_EQ(addresses[0].sin6.sin6_scope_id, 6U) << "Unfiltered IPv6 enumeration lost the scope identifier";
  struct in6_addr expectedIpv6 = {};
  ASSERT_EQ(::inet_pton(AF_INET6, "2001:db8::8", &expectedIpv6), 1)
      << "Could not parse the expected IPv6 address";
  EXPECT_EQ(std::memcmp(&addresses[0].sin6.sin6_addr, &expectedIpv6, sizeof(expectedIpv6)), 0)
      << "Unfiltered IPv6 enumeration returned the wrong address";
  EXPECT_TRUE(linuxBoundary.parsedExpressionWasNull) << "Unfiltered enumeration did not forward a NULL filter";

  // Exclusion syntax and maxIfs must omit a listed name and stop after the first eligible
  // alternative.
  ResetLinuxBoundary();
  SyntheticInterfaceList excludedInterfaces;
  ASSERT_TRUE(excludedInterfaces.AddIpv4("skip0", "10.4.0.1", "255.255.255.0", IFF_RUNNING))
      << "Could not add the excluded entry";
  ASSERT_TRUE(excludedInterfaces.AddIpv4("keep0", "10.4.0.2", "255.255.255.0", IFF_RUNNING))
      << "Could not add the retained entry";
  ASSERT_TRUE(excludedInterfaces.AddIpv4("later0", "10.4.0.3", "255.255.255.0", IFF_RUNNING))
      << "Could not add the max-count entry";
  linuxBoundary.interfaceHead = excludedInterfaces.Head();
  linuxBoundary.includedInterfaceNames = {"skip0"};
  names.fill(0);
  std::memset(addresses.data(), 0, sizeof(addresses));
  found = 7;
  ASSERT_EQ(ncclOsFindInterfaces("^skip0", names.data(), addresses.data(), -1, MAX_IF_NAME_SIZE, 1,
                                &found), ncclSuccess)
      << "A valid exclusion enumeration returned an error";
  EXPECT_EQ(found, 1) << "Exclusion enumeration ignored the maximum interface count";
  EXPECT_STREQ(names.data(), "keep0") << "Exclusion enumeration retained the excluded interface";
  EXPECT_FALSE(linuxBoundary.lastInterfaceMatchExact) << "Exclusion unexpectedly requested exact matching";
  EXPECT_EQ(linuxBoundary.parsedInterfaceExpression, "skip0")
      << "The exclusion marker was not removed before parsing";

  // A zero output capacity must still release the list and leave all caller buffers unchanged.
  ResetLinuxBoundary();
  linuxBoundary.interfaceHead = excludedInterfaces.Head();
  names.fill(static_cast<char>(0x31));
  const auto zeroCapacityNames = names;
  std::memset(addresses.data(), 0x42, sizeof(addresses));
  const auto zeroCapacityAddresses = addresses;
  found = 7;
  EXPECT_EQ(ncclOsFindInterfaces(nullptr, names.data(), addresses.data(), -1, MAX_IF_NAME_SIZE, 0,
                                &found), ncclSuccess)
      << "Zero-capacity enumeration returned an error";
  EXPECT_EQ(found, 0) << "Zero-capacity enumeration reported an interface";
  EXPECT_TRUE(linuxBoundary.parsedExpressionWasNull) << "A NULL expression was not forwarded as NULL";
  EXPECT_EQ(names, zeroCapacityNames) << "Zero-capacity enumeration modified name storage";
  EXPECT_EQ(std::memcmp(addresses.data(), zeroCapacityAddresses.data(), sizeof(addresses)), 0)
      << "Zero-capacity enumeration modified address storage";
  EXPECT_EQ(linuxBoundary.freeIfAddrsCalls, 1) << "Zero-capacity enumeration did not release its list";

  // getifaddrs failure must return ncclSystemError, clear found, and preserve both output arrays.
  ResetLinuxBoundary();
  linuxBoundary.forceGetIfAddrsFailure = true;
  linuxBoundary.getIfAddrsError = EIO;
  names.fill(static_cast<char>(0x6b));
  const auto failedNames = names;
  std::memset(addresses.data(), 0x7c, sizeof(addresses));
  const auto failedAddresses = addresses;
  found = 7;
  EXPECT_EQ(ncclOsFindInterfaces("eth0", names.data(), addresses.data(), -1, MAX_IF_NAME_SIZE,
                                addresses.size(), &found), ncclSystemError)
      << "getifaddrs failure did not return ncclSystemError";
  EXPECT_EQ(found, 0) << "Failed enumeration retained a stale found count";
  EXPECT_EQ(names, failedNames) << "Failed enumeration modified name storage";
  EXPECT_EQ(std::memcmp(addresses.data(), failedAddresses.data(), sizeof(addresses)), 0)
      << "Failed enumeration modified address storage";
  EXPECT_EQ(linuxBoundary.freeIfAddrsCalls, 0) << "Failed enumeration freed a list it never received";
}

/**
 * @brief Verifies IPv4 and IPv6 subnet rules plus public matching success, absence, and error
 * outcomes.
 * @note Side effects: Publishes synthetic interface lists through the recording getifaddrs
 * boundary.
 */
TEST_F(LinuxInterfaceTest, SubnetMatching) {
  SCOPED_TRACE("Subnet matching must honor address families, masks, IPv6 scopes, and output ownership.");

  // IPv4 matching must compare only network bits and reject another subnet or address family.
  SyntheticInterfaceList ipv4Interfaces;
  ASSERT_TRUE(ipv4Interfaces.AddIpv4("v4", "192.168.7.9", "255.255.255.0", IFF_RUNNING))
      << "Could not create the IPv4 subnet oracle";
  const struct ifaddrs ipv4 = *ipv4Interfaces.Head();
  union ncclSocketAddress remote = {};
  remote.sin.sin_family = AF_INET;
  ASSERT_EQ(::inet_pton(AF_INET, "192.168.7.200", &remote.sin.sin_addr), 1)
      << "Could not parse the matching IPv4 address";
  EXPECT_TRUE(matchSubnet(ipv4, &remote)) << "IPv4 addresses in the same /24 subnet did not match";
  ASSERT_EQ(::inet_pton(AF_INET, "192.168.8.1", &remote.sin.sin_addr), 1)
      << "Could not parse the different IPv4 address";
  EXPECT_FALSE(matchSubnet(ipv4, &remote)) << "IPv4 addresses in different /24 subnets matched";
  remote.sin6.sin6_family = AF_INET6;
  EXPECT_FALSE(matchSubnet(ipv4, &remote)) << "Different address families matched";

  // IPv6 matching must apply all 128 mask bits and require equal scope identifiers.
  SyntheticInterfaceList ipv6Interfaces;
  ASSERT_TRUE(ipv6Interfaces.AddIpv6("v6", "fe80::10", "ffff:ffff:ffff:ffff::", 4, IFF_RUNNING))
      << "Could not create the IPv6 subnet oracle";
  const struct ifaddrs ipv6 = *ipv6Interfaces.Head();
  std::memset(&remote, 0, sizeof(remote));
  remote.sin6.sin6_family = AF_INET6;
  remote.sin6.sin6_scope_id = 4;
  ASSERT_EQ(::inet_pton(AF_INET6, "fe80::20", &remote.sin6.sin6_addr), 1)
      << "Could not parse the matching IPv6 address";
  EXPECT_TRUE(matchSubnet(ipv6, &remote)) << "IPv6 addresses in the same /64 and scope did not match";
  remote.sin6.sin6_scope_id = 5;
  EXPECT_FALSE(matchSubnet(ipv6, &remote)) << "IPv6 addresses in different scopes matched";
  remote.sin6.sin6_scope_id = 4;
  ASSERT_EQ(::inet_pton(AF_INET6, "fe81::20", &remote.sin6.sin6_addr), 1)
      << "Could not parse the different IPv6 address";
  EXPECT_FALSE(matchSubnet(ipv6, &remote)) << "IPv6 addresses in different /64 subnets matched";

  // Unsupported local families must return false without inspecting a network mask.
  struct sockaddr unsupportedAddress = {};
  unsupportedAddress.sa_family = AF_UNIX;
  struct ifaddrs unsupported = {};
  unsupported.ifa_addr = &unsupportedAddress;
  remote.sa.sa_family = AF_UNIX;
  EXPECT_FALSE(matchSubnet(unsupported, &remote)) << "An unsupported family was classified as a matching subnet";

  // Public lookup must skip a nonmatching entry and publish the first matching name and complete
  // address.
  ResetLinuxBoundary();
  SyntheticInterfaceList lookupInterfaces;
  ASSERT_TRUE(lookupInterfaces.AddIpv4("wrong0", "10.7.1.8", "255.255.255.0", IFF_RUNNING))
      << "Could not add the nonmatching lookup entry";
  ASSERT_TRUE(lookupInterfaces.AddIpv4("right0", "10.7.2.9", "255.255.255.0", IFF_RUNNING))
      << "Could not add the matching lookup entry";
  linuxBoundary.interfaceHead = lookupInterfaces.Head();
  std::memset(&remote, 0, sizeof(remote));
  remote.sin.sin_family = AF_INET;
  remote.sin.sin_port = htons(3210);
  ASSERT_EQ(::inet_pton(AF_INET, "10.7.2.200", &remote.sin.sin_addr), 1)
      << "Could not parse the public lookup remote address";
  const auto remoteBefore = remote;
  std::array<char, MAX_IF_NAME_SIZE> interfaceName;
  interfaceName.fill(static_cast<char>(0x44));
  union ncclSocketAddress localAddress;
  std::memset(&localAddress, 0x55, sizeof(localAddress));
  int found = 9;
  ASSERT_EQ(ncclFindInterfaceMatchSubnet(interfaceName.data(), &localAddress, &remote,
                                        interfaceName.size(), &found), ncclSuccess)
      << "A valid subnet lookup returned an error";
  EXPECT_EQ(found, 1) << "Subnet lookup did not report its matching interface";
  EXPECT_STREQ(interfaceName.data(), "right0") << "Subnet lookup returned the wrong interface name";
  EXPECT_EQ(localAddress.sa.sa_family, AF_INET) << "Subnet lookup returned the wrong address family";
  EXPECT_EQ(localAddress.sin.sin_addr.s_addr, htonl(0x0a070209U))
      << "Subnet lookup returned the wrong local address";
  EXPECT_EQ(std::memcmp(&remote, &remoteBefore, sizeof(remote)), 0) << "Subnet lookup modified its remote input";
  EXPECT_EQ(linuxBoundary.freeIfAddrsCalls, 1) << "Subnet lookup did not release its interface list";

  // No match is a successful search with found cleared and both data outputs unchanged.
  ResetLinuxBoundary();
  linuxBoundary.interfaceHead = lookupInterfaces.Head();
  ASSERT_EQ(::inet_pton(AF_INET, "10.7.3.1", &remote.sin.sin_addr), 1)
      << "Could not parse the absent-subnet address";
  interfaceName.fill(static_cast<char>(0x66));
  const auto noMatchName = interfaceName;
  std::memset(&localAddress, 0x77, sizeof(localAddress));
  const auto noMatchAddress = localAddress;
  found = 9;
  EXPECT_EQ(ncclFindInterfaceMatchSubnet(interfaceName.data(), &localAddress, &remote,
                                        interfaceName.size(), &found), ncclSuccess)
      << "An absent subnet returned an error";
  EXPECT_EQ(found, 0) << "An absent subnet reported a match";
  EXPECT_EQ(interfaceName, noMatchName) << "An absent subnet modified the interface-name output";
  EXPECT_EQ(std::memcmp(&localAddress, &noMatchAddress, sizeof(localAddress)), 0)
      << "An absent subnet modified the local-address output";

  // getifaddrs failure must return ncclSystemError and preserve data outputs while clearing found.
  ResetLinuxBoundary();
  linuxBoundary.forceGetIfAddrsFailure = true;
  linuxBoundary.getIfAddrsError = EACCES;
  interfaceName.fill(static_cast<char>(0x22));
  const auto failedName = interfaceName;
  std::memset(&localAddress, 0x33, sizeof(localAddress));
  const auto failedAddress = localAddress;
  found = 9;
  EXPECT_EQ(ncclFindInterfaceMatchSubnet(interfaceName.data(), &localAddress, &remote,
                                        interfaceName.size(), &found), ncclSystemError)
      << "A subnet enumeration failure did not return ncclSystemError";
  EXPECT_EQ(found, 0) << "A failed subnet lookup retained a stale found flag";
  EXPECT_EQ(interfaceName, failedName) << "A failed subnet lookup modified the interface-name output";
  EXPECT_EQ(std::memcmp(&localAddress, &failedAddress, sizeof(localAddress)), 0)
      << "A failed subnet lookup modified the local-address output";
  EXPECT_EQ(linuxBoundary.freeIfAddrsCalls, 0) << "A failed subnet lookup freed a list it never received";
}

/**
 * @brief Holds synchronization objects and a predicate in shared storage.
 * @note The objects become process-shared only after ncclOsSetMutexCondShared configures them.
 */
struct SharedSignal {
  std::mutex mutex;
  std::condition_variable condition;
  int ready = 0;
};

/**
 * @brief Returns whether a filesystem path currently exists.
 * @param[in] path Valid non-NULL path.
 * @return true when stat succeeds, otherwise false.
 * @note Side effects: The delegated stat call may update errno.
 */
bool PathExists(const char* path) {
  struct stat status = {};
  return ::stat(path, &status) == 0;
}

/**
 * @brief Verifies affinity set operations, real process affinity, deterministic failures, and CPU
 * reporting.
 * @note Side effects: Reapplies the process's existing affinity mask and mutates the recording
 * boundary.
 */
TEST_F(LinuxResourceTest, CpuAffinity) {
  SCOPED_TRACE("CPU helpers must implement exact set algebra and preserve affinity inputs on system errors.");

  // Pure affinity helpers must support the first and last representable CPUs and exact
  // intersection.
  ncclAffinity first;
  ncclAffinity second;
  ncclOsCpuZero(first);
  ncclOsCpuZero(second);
  EXPECT_EQ(ncclOsCpuCount(first), 0) << "A zeroed affinity mask contained CPUs";
  ncclOsCpuSet(first, 0);
  ncclOsCpuSet(first, CPU_SETSIZE - 1);
  ncclOsCpuSet(second, CPU_SETSIZE - 1);
  EXPECT_TRUE(ncclOsCpuIsSet(first, 0)) << "The first representable CPU was not set";
  EXPECT_TRUE(ncclOsCpuIsSet(first, CPU_SETSIZE - 1)) << "The last representable CPU was not set";
  EXPECT_EQ(ncclOsCpuCount(first), 2) << "Affinity count did not reflect two distinct CPUs";
  const ncclAffinity intersection = ncclOsCpuAnd(first, second);
  EXPECT_FALSE(ncclOsCpuIsSet(intersection, 0)) << "Affinity intersection retained a CPU absent from one operand";
  EXPECT_TRUE(ncclOsCpuIsSet(intersection, CPU_SETSIZE - 1))
      << "Affinity intersection lost the shared CPU";
  EXPECT_EQ(ncclOsCpuCount(intersection), 1) << "Affinity intersection reported the wrong CPU count";

  // Real affinity retrieval must return a nonempty mask that can be reapplied without semantic
  // change.
  ncclAffinity current;
  ncclOsCpuZero(current);
  ASSERT_EQ(ncclOsGetAffinity(&current), ncclSuccess) << "Reading current process affinity returned an error";
  EXPECT_GT(ncclOsCpuCount(current), 0) << "Current process affinity was empty";
  EXPECT_EQ(ncclOsSetAffinity(current), ncclSuccess) << "Reapplying current process affinity returned an error";
  const int currentCpu = ncclOsGetCpu();
  EXPECT_GE(currentCpu, 0) << "Current CPU query returned an error";
  if (currentCpu >= 0 && currentCpu < CPU_SETSIZE) {
    EXPECT_TRUE(ncclOsCpuIsSet(current, currentCpu)) << "Current CPU was outside the process affinity mask";
  }

  // A deterministic CPU value must be returned unchanged.
  linuxBoundary.forceGetCpu = true;
  linuxBoundary.cpuResult = 17;
  EXPECT_EQ(ncclOsGetCpu(), 17) << "CPU query changed the boundary result";

  // Retrieval failure must preserve the caller's mask and report ncclSystemError.
  ncclAffinity failedOutput;
  ncclOsCpuZero(failedOutput);
  ncclOsCpuSet(failedOutput, 3);
  const ncclAffinity failedOutputBefore = failedOutput;
  linuxBoundary.failGetAffinity = true;
  linuxBoundary.affinityError = EIO;
  EXPECT_EQ(ncclOsGetAffinity(&failedOutput), ncclSystemError)
      << "sched_getaffinity failure did not return ncclSystemError";
  EXPECT_EQ(std::memcmp(&failedOutput, &failedOutputBefore, sizeof(failedOutput)), 0)
      << "Failed affinity retrieval modified its output mask";

  // Setting failure must report ncclSystemError.
  linuxBoundary.failGetAffinity = false;
  linuxBoundary.failSetAffinity = true;
  EXPECT_EQ(ncclOsSetAffinity(current), ncclSystemError)
      << "sched_setaffinity failure did not return ncclSystemError";
}

/**
 * @brief Verifies that the mutex and condition-variable configuration synchronizes two processes.
 * @note Side effects: Maps shared memory, forks one child, waits for it, and releases the mapping.
 */
TEST_F(LinuxResourceTest, SharedSynchronization) {
  SCOPED_TRACE("Process-shared synchronization must transfer a predicate update from child to parent.");

  void* storage = ::mmap(nullptr, sizeof(SharedSignal), PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(storage, MAP_FAILED) << "The test could not map process-shared synchronization storage";
  SharedSignal* signal = new (storage) SharedSignal();
  ncclOsSetMutexCondShared(signal->mutex, signal->condition);

  const pid_t child = ::fork();
  ASSERT_GE(child, 0) << "The test could not fork a synchronization peer";
  if (child == 0) {
    {
      std::lock_guard<std::mutex> lock(signal->mutex);
      signal->ready = 1;
    }
    signal->condition.notify_one();
    ::_exit(0);
  }

  std::unique_lock<std::mutex> lock(signal->mutex);
  const bool observed = signal->condition.wait_for(lock, std::chrono::seconds(2),
                                                    [signal] { return signal->ready == 1; });
  lock.unlock();
  int childStatus = -1;
  ASSERT_EQ(::waitpid(child, &childStatus, 0), child) << "The test could not reap its synchronization peer";
  EXPECT_TRUE(observed) << "The parent did not observe the child predicate through the shared condition variable";
  EXPECT_TRUE(WIFEXITED(childStatus)) << "The synchronization peer did not exit normally";
  if (WIFEXITED(childStatus)) {
    EXPECT_EQ(WEXITSTATUS(childStatus), 0) << "The synchronization peer failed";
  }
  signal->~SharedSignal();
  EXPECT_EQ(::munmap(storage, sizeof(SharedSignal)), 0) << "The test could not release synchronization storage";
}

/**
 * @brief Verifies shared-memory handle field ownership for creator and attacher initialization.
 * @note Side effects: Clears a host payload for the creator case and allocates one copied path.
 */
TEST_F(LinuxResourceTest, SharedMemoryHandle) {
  SCOPED_TRACE("Handle initialization must publish exact fields, own only creator paths, and clear only creator data.");

  constexpr size_t payloadSize = 32;
  std::array<char, payloadSize + sizeof(int)> storage;
  storage.fill(static_cast<char>(0x5a));
  char path[] = "/dev/shm/occl-linux-handle";
  struct ncclShmHandleInternal creator = {};
  void* devicePointer = reinterpret_cast<void*>(0x1234);
  ncclOsShmHandleInit(17, path, payloadSize, storage.size(), storage.data(), devicePointer, true, &creator);
  EXPECT_EQ(creator.shmDesc, 17) << "Creator initialization changed the descriptor";
  EXPECT_EQ(creator.shmPtr, storage.data()) << "Creator initialization changed the host pointer";
  EXPECT_EQ(creator.devShmPtr, devicePointer) << "Creator initialization changed the device pointer";
  EXPECT_EQ(creator.shmSize, payloadSize) << "Creator initialization changed the payload size";
  EXPECT_EQ(creator.realShmSize, storage.size()) << "Creator initialization changed the mapped size";
  EXPECT_EQ(creator.refcount, reinterpret_cast<int*>(storage.data() + payloadSize))
      << "Creator initialization placed the refcount at the wrong offset";
  ASSERT_NE(creator.shmPath, nullptr) << "Creator initialization did not copy its path";
  EXPECT_NE(creator.shmPath, path) << "Creator initialization retained caller-owned path storage";
  EXPECT_STREQ(creator.shmPath, path) << "Creator initialization copied the wrong path";
  EXPECT_TRUE(std::all_of(storage.begin(), storage.begin() + payloadSize,
                          [](char value) { return value == 0; }))
      << "Creator initialization did not clear the complete payload";
  std::free(creator.shmPath);

  // An attaching handle must not own the path or clear existing shared data.
  storage.fill(static_cast<char>(0x6b));
  const auto storageBeforeAttach = storage;
  struct ncclShmHandleInternal attacher = {};
  ncclOsShmHandleInit(23, path, payloadSize, storage.size(), storage.data(), nullptr, false, &attacher);
  EXPECT_EQ(attacher.shmDesc, 23) << "Attacher initialization changed the descriptor";
  EXPECT_EQ(attacher.shmPath, nullptr) << "Attacher initialization incorrectly copied the creator path";
  EXPECT_EQ(attacher.refcount, reinterpret_cast<int*>(storage.data() + payloadSize))
      << "Attacher initialization placed the refcount at the wrong offset";
  EXPECT_EQ(storage, storageBeforeAttach) << "Attacher initialization modified existing shared data";
  EXPECT_STREQ(path, "/dev/shm/occl-linux-handle") << "Handle initialization modified its input path";

  // A NULL host mapping must produce no refcount pointer while retaining the remaining supplied
  // metadata.
  struct ncclShmHandleInternal empty = {};
  ncclOsShmHandleInit(-1, path, payloadSize, storage.size(), nullptr, devicePointer, false, &empty);
  EXPECT_EQ(empty.shmDesc, -1) << "NULL-mapping initialization changed the invalid descriptor";
  EXPECT_EQ(empty.shmPtr, nullptr) << "NULL-mapping initialization published a host pointer";
  EXPECT_EQ(empty.refcount, nullptr) << "NULL-mapping initialization published a refcount pointer";
  EXPECT_EQ(empty.devShmPtr, devicePointer) << "NULL-mapping initialization changed the device pointer";
  EXPECT_EQ(empty.shmPath, nullptr) << "NULL-mapping attachment incorrectly took path ownership";
}

/**
 * @brief Verifies shared-memory creation, attachment, sharing, reference decrement, unlink,
 * and close.
 * @note Side effects: Creates, maps, unlinks, and closes temporary files under /dev/shm.
 */
TEST_F(LinuxResourceTest, SharedMemoryLifecycle) {
  SCOPED_TRACE("Shared-memory lifecycle must expose shared data and remove creator-owned files at the right time.");

  constexpr size_t payloadSize = 64;
  std::array<char, 128> path = {};
  void* creatorPointer = reinterpret_cast<void*>(0x1111);
  struct ncclShmHandleInternal* creatorHandle = reinterpret_cast<struct ncclShmHandleInternal*>(0x2222);
  ASSERT_EQ(ncclOsShmOpen(path.data(), path.size(), payloadSize, &creatorPointer, nullptr, 2,
                          &creatorHandle), ncclSuccess)
      << "Creating valid shared memory returned an error";
  ASSERT_NE(creatorPointer, nullptr) << "Shared-memory creation returned no host mapping";
  ASSERT_NE(creatorHandle, nullptr) << "Shared-memory creation returned no handle";
  EXPECT_TRUE(PathExists(path.data())) << "Shared-memory creation did not publish its backing file";
  struct stat status = {};
  ASSERT_EQ(::stat(path.data(), &status), 0) << "The test could not inspect the shared-memory backing file";
  EXPECT_EQ(status.st_size, static_cast<off_t>(payloadSize + sizeof(int)))
      << "The shared-memory backing file has the wrong size";
  EXPECT_EQ(*creatorHandle->refcount, 2) << "Shared-memory creation stored the wrong reference count";
  EXPECT_TRUE(std::all_of(static_cast<char*>(creatorPointer), static_cast<char*>(creatorPointer) + payloadSize,
                          [](char value) { return value == 0; }))
      << "Shared-memory creation did not zero the complete payload";
  static_cast<char*>(creatorPointer)[payloadSize - 1] = static_cast<char>(0x5d);

  // Attaching must see creator data, decrement the shared count, and leave the file while one
  // reference remains.
  void* attachedPointer = reinterpret_cast<void*>(0x3333);
  struct ncclShmHandleInternal* attachedHandle = reinterpret_cast<struct ncclShmHandleInternal*>(0x4444);
  const std::string pathBeforeAttach = path.data();
  ASSERT_EQ(ncclOsShmOpen(path.data(), path.size(), payloadSize, &attachedPointer, nullptr, 0,
                          &attachedHandle), ncclSuccess)
      << "Attaching valid shared memory returned an error";
  ASSERT_NE(attachedPointer, nullptr) << "Shared-memory attachment returned no host mapping";
  ASSERT_NE(attachedHandle, nullptr) << "Shared-memory attachment returned no handle";
  EXPECT_EQ(static_cast<char*>(attachedPointer)[payloadSize - 1], static_cast<char>(0x5d))
      << "The attached mapping did not observe creator data";
  EXPECT_EQ(*creatorHandle->refcount, 1) << "Shared-memory attachment did not decrement the reference count";
  EXPECT_TRUE(PathExists(path.data())) << "Attachment removed a backing file with one reference remaining";
  EXPECT_EQ(path.data(), pathBeforeAttach) << "Attachment modified its input path";
  EXPECT_EQ(attachedHandle->shmPath, nullptr) << "The attaching handle incorrectly owned the path";
  EXPECT_EQ(ncclOsShmClose(attachedHandle), ncclSuccess) << "Closing the attaching mapping returned an error";

  const int creatorDescriptor = creatorHandle->shmDesc;
  EXPECT_EQ(ncclOsShmClose(creatorHandle), ncclSuccess) << "Closing the creator mapping returned an error";
  EXPECT_FALSE(PathExists(path.data())) << "Closing the final creator-owned mapping did not remove the file";
  errno = 0;
  EXPECT_EQ(::fcntl(creatorDescriptor, F_GETFD), -1) << "Shared-memory close retained the backing descriptor";
  EXPECT_EQ(errno, EBADF) << "The closed shared-memory descriptor failed for an unexpected reason";

  // A caller-supplied path must be used unchanged, created when absent, and removed by creator
  // close.
  std::array<char, 128> explicitPath = {};
  std::snprintf(explicitPath.data(), explicitPath.size(), "/dev/shm/occl-linux-explicit-%d", ::getpid());
  (void)::unlink(explicitPath.data());
  const std::string explicitPathBefore = explicitPath.data();
  creatorPointer = nullptr;
  creatorHandle = nullptr;
  ASSERT_EQ(ncclOsShmOpen(explicitPath.data(), explicitPath.size(), payloadSize, &creatorPointer, nullptr, 1,
                          &creatorHandle), ncclSuccess)
      << "Creating shared memory at a caller-supplied path returned an error";
  ASSERT_NE(creatorHandle, nullptr) << "Caller-supplied path creation returned no handle";
  EXPECT_EQ(explicitPath.data(), explicitPathBefore) << "Shared-memory creation modified the supplied path";
  EXPECT_TRUE(PathExists(explicitPath.data())) << "Shared-memory creation did not use the supplied path";
  EXPECT_EQ(ncclOsShmClose(creatorHandle), ncclSuccess)
      << "Closing shared memory at a caller-supplied path returned an error";
  EXPECT_FALSE(PathExists(explicitPath.data())) << "Closing shared memory retained the supplied backing path";

  // Explicit unlink must remove the file, clear path ownership, and be safely repeatable before
  // close.
  path.fill(0);
  creatorPointer = nullptr;
  creatorHandle = nullptr;
  ASSERT_EQ(ncclOsShmOpen(path.data(), path.size(), payloadSize, &creatorPointer, nullptr, 1,
                          &creatorHandle), ncclSuccess)
      << "Creating the explicit-unlink mapping returned an error";
  ASSERT_NE(creatorHandle, nullptr) << "Explicit-unlink creation returned no handle";
  EXPECT_EQ(ncclOsShmUnlink(creatorHandle), ncclSuccess) << "Explicit shared-memory unlink returned an error";
  EXPECT_FALSE(PathExists(path.data())) << "Explicit shared-memory unlink retained the backing file";
  EXPECT_EQ(creatorHandle->shmPath, nullptr) << "Explicit shared-memory unlink retained path ownership";
  EXPECT_EQ(ncclOsShmUnlink(creatorHandle), ncclSuccess) << "Repeating shared-memory unlink returned an error";
  EXPECT_EQ(ncclOsShmClose(creatorHandle), ncclSuccess) << "Closing an explicitly unlinked mapping returned an error";

  // Attaching the sole outstanding reference must decrement it to zero and remove the backing
  // file immediately.
  path.fill(0);
  creatorPointer = nullptr;
  creatorHandle = nullptr;
  ASSERT_EQ(ncclOsShmOpen(path.data(), path.size(), payloadSize, &creatorPointer, nullptr, 1,
                          &creatorHandle), ncclSuccess)
      << "Creating the final-reference mapping returned an error";
  void* finalPointer = nullptr;
  struct ncclShmHandleInternal* finalHandle = nullptr;
  ASSERT_EQ(ncclOsShmOpen(path.data(), path.size(), payloadSize, &finalPointer, nullptr, 0,
                          &finalHandle), ncclSuccess)
      << "Attaching the final shared-memory reference returned an error";
  EXPECT_EQ(*creatorHandle->refcount, 0) << "Final attachment did not decrement the reference count to zero";
  EXPECT_FALSE(PathExists(path.data())) << "Final attachment did not remove the zero-reference backing file";
  EXPECT_EQ(ncclOsShmClose(finalHandle), ncclSuccess) << "Closing the final attacher returned an error";
  EXPECT_EQ(ncclOsShmClose(creatorHandle), ncclSuccess) << "Closing the final creator returned an error";
  EXPECT_EQ(ncclOsShmClose(nullptr), ncclSuccess) << "Closing a NULL shared-memory handle returned an error";
  EXPECT_EQ(ncclOsShmUnlink(nullptr), ncclSuccess) << "Unlinking a NULL shared-memory handle returned an error";
}

/**
 * @brief Verifies shared-memory open, device-map, unlink, and close failures with exact output
 * cleanup.
 * @note Side effects: Creates and removes temporary files under /dev/shm.
 */
TEST_F(LinuxResourceTest, SharedMemoryErrors) {
  SCOPED_TRACE(
      "Shared-memory failures must return their documented status, clear outputs, and release acquired resources.");

  constexpr size_t payloadSize = 64;

  // Attaching an absent file must return ncclSystemError, clear stale outputs, and preserve the
  // path input.
  std::array<char, 128> missingPath = {};
  std::snprintf(missingPath.data(), missingPath.size(), "/dev/shm/occl-linux-missing-%d", ::getpid());
  (void)::unlink(missingPath.data());
  const std::string missingPathBefore = missingPath.data();
  void* hostPointer = reinterpret_cast<void*>(0x1111);
  struct ncclShmHandleInternal* handle = reinterpret_cast<struct ncclShmHandleInternal*>(0x2222);
  EXPECT_EQ(ncclOsShmOpen(missingPath.data(), missingPath.size(), payloadSize, &hostPointer, nullptr, 0,
                         &handle), ncclSystemError)
      << "Attaching an absent shared-memory file did not return ncclSystemError";
  EXPECT_EQ(hostPointer, nullptr) << "Failed attachment retained a stale host pointer";
  EXPECT_EQ(handle, nullptr) << "Failed attachment retained a stale handle";
  EXPECT_EQ(missingPath.data(), missingPathBefore) << "Failed attachment modified its input path";

  // A requested device mapping is unsupported in this unit build and must clean every acquired
  // resource.
  std::array<char, 128> devicePath = {};
  hostPointer = reinterpret_cast<void*>(0x3333);
  void* devicePointer = reinterpret_cast<void*>(0x4444);
  handle = reinterpret_cast<struct ncclShmHandleInternal*>(0x5555);
  EXPECT_EQ(ncclOsShmOpen(devicePath.data(), devicePath.size(), payloadSize, &hostPointer, &devicePointer, 1,
                         &handle), ncclInternalError)
      << "A disabled-runtime device mapping did not return ncclInternalError";
  EXPECT_EQ(hostPointer, nullptr) << "Failed device mapping retained a stale host pointer";
  EXPECT_EQ(devicePointer, nullptr) << "Failed device mapping retained a stale device pointer";
  EXPECT_EQ(handle, nullptr) << "Failed device mapping retained a stale handle";
  EXPECT_FALSE(PathExists(devicePath.data())) << "Failed device mapping leaked its backing file";

  // Unlink failure must return ncclSystemError, release path ownership, and permit ordinary close
  // cleanup.
  std::array<char, 128> unlinkPath = {};
  hostPointer = nullptr;
  handle = nullptr;
  ASSERT_EQ(ncclOsShmOpen(unlinkPath.data(), unlinkPath.size(), payloadSize, &hostPointer, nullptr, 1,
                          &handle), ncclSuccess)
      << "Creating the unlink-failure mapping returned an error";
  ASSERT_NE(handle, nullptr) << "Unlink-failure creation returned no handle";
  ASSERT_EQ(::unlink(unlinkPath.data()), 0) << "The test could not remove the backing file before the error check";
  EXPECT_EQ(ncclOsShmUnlink(handle), ncclSystemError)
      << "A backing-file unlink failure did not return ncclSystemError";
  EXPECT_EQ(handle->shmPath, nullptr) << "Failed unlink retained path ownership";
  EXPECT_EQ(ncclOsShmClose(handle), ncclSuccess) << "Closing after failed unlink returned an error";

  // Close must report its own unlink failure while still closing the descriptor and releasing the
  // mapping.
  std::array<char, 128> closePath = {};
  hostPointer = nullptr;
  handle = nullptr;
  ASSERT_EQ(ncclOsShmOpen(closePath.data(), closePath.size(), payloadSize, &hostPointer, nullptr, 1,
                          &handle), ncclSuccess)
      << "Creating the close-failure mapping returned an error";
  ASSERT_NE(handle, nullptr) << "Close-failure creation returned no handle";
  const int descriptor = handle->shmDesc;
  ASSERT_EQ(::unlink(closePath.data()), 0) << "The test could not remove the backing file before close";
  EXPECT_EQ(ncclOsShmClose(handle), ncclSystemError)
      << "A backing-file close unlink failure did not return ncclSystemError";
  errno = 0;
  EXPECT_EQ(::fcntl(descriptor, F_GETFD), -1) << "Failed shared-memory close retained its descriptor";
  EXPECT_EQ(errno, EBADF) << "The failed-close descriptor probe failed for an unexpected reason";
}

}  // namespace
