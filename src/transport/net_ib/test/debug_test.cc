/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/

#include "debug.h"
#include "env.h"
#include "os.h"
#include "param.h"
#include "utils.h"

#include <gtest/gtest.h>

#include <pthread.h>

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

/** @brief Records one debug-parameter lookup. */
struct LoadCall {
  std::string environment;
  int64_t defaultValue = 0;
};

/** @brief Records one debug-output file-open request. */
struct FileCall {
  std::string path;
  std::string mode;
};

/** @brief Records one native thread-naming request. */
struct NativeNameCall {
  pthread_t handle = {};
  std::string name;
};

/**
 * @brief Scripts the injected debug.cc dependencies and records requests made through those
 * seams.
 */
struct DependencyState {
  ncclResult_t hostResult = ncclSuccess;
  std::string hostValue = "test-host";
  int hostCalls = 0;
  int hostMaxLength = 0;
  char hostDelimiter = '\0';
  bool pluginInitialized = true;
  std::unordered_map<std::string, std::string> environment;
  uint64_t processId = 41;
  uint64_t threadId = 73;
  int processIdCalls = 0;
  int threadIdCalls = 0;
  int64_t parameterValue = 0;
  bool parameterCaching = false;
  std::vector<LoadCall> loadCalls;
  bool fileOpenAllowed = false;
  std::vector<FileCall> fileCalls;
  int fileCloseCalls = 0;
  std::vector<FILE*> closedFiles;
  std::vector<NativeNameCall> nativeNameCalls;
};

DependencyState dependencyState;

}  // namespace

/**
 * @brief Supplies a deterministic hostname to the included production source.
 * @param[out] output NULL, or writable storage for @p maxLength bytes when it is positive.
 * @param[in] maxLength Any integer capacity; output is written only when it is positive.
 * @param[in] delimiter Requested hostname delimiter.
 * @return The configured result without modifying output on failure.
 * @note Side effects: Records the request and writes output on success when storage is
 * available.
 */
ncclResult_t HostNameDependencyForTest(char* output, int maxLength, const char delimiter) {
  dependencyState.hostCalls++;
  dependencyState.hostMaxLength = maxLength;
  dependencyState.hostDelimiter = delimiter;
  if (dependencyState.hostResult != ncclSuccess) return dependencyState.hostResult;
  if (output != nullptr && maxLength > 0) {
    std::snprintf(output, static_cast<size_t>(maxLength), "%s", dependencyState.hostValue.c_str());
  }
  return ncclSuccess;
}

/**
 * @brief Returns whether the deterministic environment provider is active.
 * @return dependencyState.pluginInitialized.
 * @note Side effects: None.
 */
bool EnvironmentPluginInitializedForTest() {
  return dependencyState.pluginInitialized;
}

/**
 * @brief Looks up one value in the deterministic environment map.
 * @param[in] name Non-NULL environment key.
 * @return Borrowed value storage valid until the mapped value or dependencyState changes, or
 * NULL when the key is absent.
 * @note Side effects: None.
 */
const char* EnvironmentValueForTest(const char* name) {
  const auto value = dependencyState.environment.find(name);
  return value == dependencyState.environment.end() ? nullptr : value->second.c_str();
}

/**
 * @brief Supplies a deterministic process identifier.
 * @return dependencyState.processId.
 * @note Side effects: Increments dependencyState.processIdCalls.
 */
uint64_t ProcessIdForTest() {
  dependencyState.processIdCalls++;
  return dependencyState.processId;
}

/**
 * @brief Supplies a deterministic thread identifier.
 * @return dependencyState.threadId.
 * @note Side effects: Increments dependencyState.threadIdCalls.
 */
uint64_t ThreadIdForTest() {
  dependencyState.threadIdCalls++;
  return dependencyState.threadId;
}

/**
 * @brief Converts epoch time to UTC through the reentrant C library helper.
 * @param[in] timer Valid non-NULL input epoch time.
 * @param[out] output Valid non-NULL destination calendar structure.
 * @return output on success and NULL on conversion failure.
 * @note Side effects: Writes output on success.
 */
std::tm* LocalTimeForTest(const time_t* timer, std::tm* output) {
  return ::gmtime_r(timer, output);
}

/**
 * @brief Supplies a deterministic real-time value to timestamp formatting.
 * @param[out] output Valid non-NULL destination timestamp.
 * @note Side effects: Replaces output with the configured timestamp.
 */
void RealtimeForTest(struct timespec* output) {
  output->tv_sec = 0;
  output->tv_nsec = 123456789;
}

/**
 * @brief Models cached or uncached parameter loading and records the requested key and default.
 * @param[in] environment NULL or an environment key recorded as an empty or exact string.
 * @param[in] defaultValue Production default recorded without modification.
 * @param[in] uninitializedValue Cache sentinel written when caching is disabled.
 * @param[out] cache NULL or storage receiving the configured value or uninitializedValue.
 * @param[out] noCache NULL or storage receiving zero when caching is enabled and one otherwise.
 * @return The configured parameter value.
 * @note Side effects: Appends to loadCalls and conditionally replaces cache and noCache.
 */
int64_t LoadParameterForTest(const char* environment, int64_t defaultValue, int64_t uninitializedValue,
    int64_t* cache, int8_t* noCache) {
  dependencyState.loadCalls.push_back(
      {environment == nullptr ? "" : environment, defaultValue});
  if (cache != nullptr) {
    *cache = dependencyState.parameterCaching ? dependencyState.parameterValue : uninitializedValue;
  }
  if (noCache != nullptr) *noCache = dependencyState.parameterCaching ? 0 : 1;
  return dependencyState.parameterValue;
}

/**
 * @brief Supplies an optional anonymous stream to debug-file initialization.
 * @param[in] path NULL or a file path recorded as an empty or exact string.
 * @param[in] mode NULL or an open mode recorded as an empty or exact string.
 * @return A caller-owned writable stream when enabled and allocation succeeds; otherwise NULL.
 * @note Side effects: Appends to fileCalls and may create a temporary stream.
 */
FILE* OpenFileForTest(const char* path, const char* mode) {
  dependencyState.fileCalls.push_back({path == nullptr ? "" : path, mode == nullptr ? "" : mode});
  return dependencyState.fileOpenAllowed ? std::tmpfile() : nullptr;
}

/**
 * @brief Implements debug-stream closure for the included production source.
 * @param[in] stream Valid non-NULL stream whose ownership is released.
 * @return The underlying close result.
 * @note Side effects: Increments fileCloseCalls, appends to closedFiles, and closes stream.
 */
int CloseFileForTest(FILE* stream) {
  dependencyState.fileCloseCalls++;
  dependencyState.closedFiles.push_back(stream);
  return std::fclose(stream);
}

/**
 * @brief Supplies deterministic success to native thread naming.
 * @param[in] handle Native thread handle recorded without modification.
 * @param[in] name NULL or a thread name recorded as an empty or exact string.
 * @return 0.
 * @note Side effects: Appends the normalized handle and name to nativeNameCalls.
 */
int SetNativeThreadNameForTest(pthread_t handle, const char* name) {
  dependencyState.nativeNameCalls.push_back({handle, name == nullptr ? "" : name});
  return 0;
}

/**
 * @brief Compiles debug.cc with deterministic seams and undefines them before the tests.
 */
#define getHostName HostNameDependencyForTest
#define ncclEnvPluginInitialized EnvironmentPluginInitializedForTest
#define ncclGetEnv EnvironmentValueForTest
#define ncclOsGetPid ProcessIdForTest
#define ncclOsGetTid ThreadIdForTest
#define ncclOsLocaltime LocalTimeForTest
#define clockRealtime RealtimeForTest
#define ncclLoadParam LoadParameterForTest
#define fopen OpenFileForTest
#define fclose CloseFileForTest
#define pthread_setname_np SetNativeThreadNameForTest
#include "../../../debug.cc"  // NOLINT(build/include)
#undef pthread_setname_np
#undef fclose
#undef fopen
#undef ncclLoadParam
#undef clockRealtime
#undef ncclOsLocaltime
#undef ncclOsGetTid
#undef ncclOsGetPid
#undef ncclGetEnv
#undef ncclEnvPluginInitialized
#undef getHostName

namespace {

/**
 * @brief Restores the debug globals and deterministic dependencies exercised by these tests.
 * @note Side effects: Closes any owned non-stdout debug stream and replaces dependency and
 * debug state.
 */
void ResetSourceStateForTest() {
  if (ncclDebugFile != nullptr && ncclDebugFile != stdout) std::fclose(ncclDebugFile);
  dependencyState = DependencyState{};
  ncclDebugLevel = -1;
  ncclDebugTimestampLevels = 0;
  std::memset(ncclDebugTimestampFormat, 0, sizeof(ncclDebugTimestampFormat));
  ncclDebugTimestampSubsecondsStart = -1;
  ncclDebugTimestampMaxSubseconds = 0;
  ncclDebugTimestampSubsecondDigits = 0;
  pid = -1;
  std::memset(hostname, 0, sizeof(hostname));
  ncclDebugNoWarn = 0;
  std::memset(ncclLastError, 0, sizeof(ncclLastError));
  ncclDebugMask = 0;
  ncclDebugFile = stdout;
  ncclEpoch = std::chrono::steady_clock::time_point{};
  ncclWarnSetDebugInfo = false;
  tid = -1;
}

/**
 * @brief Flushes, rewinds, and reads one seekable test stream.
 * @param[in,out] stream Valid non-NULL stream flushed, rewound, and read to exhaustion.
 * @return Bytes read after successful preparation, or an empty string if flushing or rewinding
 * fails.
 * @note Side effects: Changes stream position and records a GoogleTest failure after any I/O
 * failure.
 */
std::string ReadOutputForTest(FILE* stream) {
  if (std::fflush(stream) != 0) {
    ADD_FAILURE() << "The test could not flush the captured debug stream";
    return {};
  }
  if (std::fseek(stream, 0, SEEK_SET) != 0) {
    ADD_FAILURE() << "The test could not rewind the captured debug stream";
    return {};
  }
  std::string output;
  char chunk[256];
  size_t count = 0;
  while ((count = std::fread(chunk, 1, sizeof(chunk), stream)) != 0) output.append(chunk, count);
  if (std::ferror(stream) != 0) ADD_FAILURE() << "The test could not read the complete captured debug stream";
  return output;
}

/** @brief Provides isolated production and dependency state for debug-source tests. */
class DebugSourceTest : public ::testing::Test {
 protected:
  /**
   * @brief Starts each test with isolated production and dependency state.
   * @note Side effects: Replaces production and dependency state and may close an owned stream.
   */
  void SetUp() override { ResetSourceStateForTest(); }

  /**
   * @brief Releases a debug stream that the test did not transfer elsewhere.
   * @note Side effects: Replaces production and dependency state and may close an owned stream.
   */
  void TearDown() override { ResetSourceStateForTest(); }
};

/**
 * @brief Verifies hostname sanitization, bounded output, dependency forwarding, and error
 * propagation.
 */
TEST_F(DebugSourceTest, SanitizesHostNameAndPropagatesFailure) {
  dependencyState.hostValue = "rack%2/node";
  char output[32] = "unchanged";

  EXPECT_EQ(getHostNameForLog(output, sizeof(output), '.'), ncclSuccess)
      << "A successful hostname lookup should be reported as successful";
  EXPECT_STREQ(output, "rack-2-node")
      << "Percent and slash characters should be replaced while other hostname bytes remain intact";
  EXPECT_EQ(dependencyState.hostCalls, 1) << "The hostname provider should be called exactly once";
  EXPECT_EQ(dependencyState.hostMaxLength, static_cast<int>(sizeof(output)))
      << "The destination capacity should be forwarded unchanged";
  EXPECT_EQ(dependencyState.hostDelimiter, '.') << "The hostname delimiter should be forwarded unchanged";

  struct {
    char output[5];
    unsigned char canary;
  } bounded = {{}, 0xa5};
  dependencyState.hostValue = "a/%b-tail";
  EXPECT_EQ(getHostNameForLog(bounded.output, sizeof(bounded.output), '.'), ncclSuccess)
      << "A minimum useful bounded hostname should still be reported as successful";
  EXPECT_STREQ(bounded.output, "a--b")
      << "Sanitization should retain the provider's termination at the caller's exact capacity";
  EXPECT_EQ(bounded.canary, 0xa5) << "Sanitization must not write beyond the declared destination capacity";

  dependencyState.hostResult = ncclSystemError;
  EXPECT_EQ(getHostNameForLog(output, sizeof(output), '/'), ncclSystemError)
      << "The hostname provider error should be returned unchanged";
  EXPECT_EQ(dependencyState.hostCalls, 3)
      << "Each successful or failing request should reach the provider exactly once";
}

/**
 * @brief Verifies all supported debug levels plus default and explicit initialization policy.
 */
TEST_F(DebugSourceTest, InitializesLevelsMasksAndTimestampPolicy) {
  dependencyState.hostValue = "node%1/rack";
  dependencyState.processId = 912;
  {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    ncclDebugInit();
  }
  EXPECT_EQ(ncclDebugLevel, NCCL_LOG_NONE) << "An absent debug-level setting should disable logging";
  EXPECT_EQ(ncclDebugMask, static_cast<uint64_t>(NCCL_INIT | NCCL_BOOTSTRAP | NCCL_ENV))
      << "Default initialization should enable only the documented startup subsystems";
  EXPECT_EQ(ncclDebugTimestampLevels, static_cast<uint32_t>(1U << NCCL_LOG_WARN))
      << "Warnings should be the only timestamped level by default";
  EXPECT_STREQ(ncclDebugTimestampFormat, "[%F %T] ") << "The documented timestamp format should be installed";
  EXPECT_STREQ(hostname, "node-1-rack") << "The cached hostname should use log-safe characters";
  EXPECT_EQ(pid, 912) << "The process identifier should be cached from the OS provider";
  EXPECT_EQ(dependencyState.hostCalls, 1) << "Initialization should resolve the hostname once";
  EXPECT_EQ(dependencyState.processIdCalls, 1) << "Initialization should resolve the process identifier once";

  const struct {
    const char* text;
    int expected;
  } levels[] = {{"VERSION", NCCL_LOG_VERSION}, {"WARN", NCCL_LOG_WARN}, {"iNfO", NCCL_LOG_INFO},
      {"ABORT", NCCL_LOG_ABORT}, {"TRACE", NCCL_LOG_TRACE}};
  for (const auto& level : levels) {
    ResetSourceStateForTest();
    dependencyState.environment["NCCL_DEBUG"] = level.text;
    {
      std::lock_guard<std::mutex> lock(ncclDebugMutex);
      ncclDebugInit();
    }
    EXPECT_EQ(ncclDebugLevel, level.expected)
        << "Supported debug level '" << level.text << "' should map to its declared enum value";
  }

  ResetSourceStateForTest();
  dependencyState.environment["NCCL_DEBUG"] = "TRACE";
  dependencyState.environment["NCCL_DEBUG_SUBSYS"] = "NET,P2P,UNKNOWN";
  dependencyState.environment["NCCL_WARN_ENABLE_DEBUG_INFO"] = "1";
  dependencyState.environment["NCCL_DEBUG_TIMESTAMP_LEVELS"] = "INFO,TRACE";
  dependencyState.environment["NCCL_DEBUG_TIMESTAMP_FORMAT"] = "T%3f_";
  const std::string subsystemSetting = dependencyState.environment["NCCL_DEBUG_SUBSYS"];
  const std::string timestampLevelSetting = dependencyState.environment["NCCL_DEBUG_TIMESTAMP_LEVELS"];
  const std::string timestampFormatSetting = dependencyState.environment["NCCL_DEBUG_TIMESTAMP_FORMAT"];
  {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    ncclDebugInit();
  }
  EXPECT_EQ(ncclDebugMask, static_cast<uint64_t>(NCCL_NET | NCCL_P2P))
      << "Explicit subsystems should replace the default mask and unknown tokens should be ignored";
  EXPECT_EQ(ncclDebugTimestampLevels,
      static_cast<uint32_t>((1U << NCCL_LOG_INFO) | (1U << NCCL_LOG_TRACE)))
      << "Only explicitly selected timestamp levels should remain enabled";
  EXPECT_EQ(dependencyState.environment["NCCL_DEBUG_SUBSYS"], subsystemSetting)
      << "Subsystem parsing should not modify the provider-owned environment string";
  EXPECT_EQ(dependencyState.environment["NCCL_DEBUG_TIMESTAMP_LEVELS"], timestampLevelSetting)
      << "Timestamp-level parsing should not modify the provider-owned environment string";
  EXPECT_EQ(dependencyState.environment["NCCL_DEBUG_TIMESTAMP_FORMAT"], timestampFormatSetting)
      << "Timestamp-format parsing should not modify the provider-owned environment string";

  FILE* stream = std::tmpfile();
  ASSERT_NE(stream, nullptr) << "The timestamp policy requires a seekable output stream";
  ncclDebugFile = stream;
  ncclDebugLog(NCCL_LOG_INFO, NCCL_NET, "timestamp.cc", 9, "timestamped");
  ncclDebugLog(NCCL_LOG_WARN, NCCL_NET, "warning.cc", 10, "escalate");
  const std::string initializedOutput = ReadOutputForTest(stream);
  EXPECT_EQ(initializedOutput.find("T123 "), 0U)
      << "The initialized three-digit subsecond format should produce the deterministic T123 prefix";
  EXPECT_EQ(ncclDebugLevel, NCCL_LOG_INFO)
      << "An enabled warning escalation policy should raise the active debug level to INFO after a warning";
  EXPECT_STREQ(ncclLastError, "escalate") << "The warning used for escalation should still update the last error";

  ResetSourceStateForTest();
  dependencyState.environment["NCCL_DEBUG"] = "WARN";
  dependencyState.environment["NCCL_DEBUG_SUBSYS"] = "^NET";
  dependencyState.environment["NCCL_DEBUG_TIMESTAMP_LEVELS"] = "^WARN";
  {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    ncclDebugInit();
  }
  EXPECT_EQ(ncclDebugMask, ~static_cast<uint64_t>(NCCL_NET))
      << "An inverted subsystem list should enable every subsystem except the named one";
  EXPECT_EQ(ncclDebugTimestampLevels, ~static_cast<uint32_t>(1U << NCCL_LOG_WARN))
      << "An inverted timestamp list should enable every level except the named one";

  ResetSourceStateForTest();
  const char* previousValue = std::getenv("NCCL_DEBUG");
  const bool hadPreviousValue = previousValue != nullptr;
  const std::string previousValueCopy = hadPreviousValue ? previousValue : "";
  ASSERT_EQ(::setenv("NCCL_DEBUG", "INFO", 1), 0)
      << "The environment-source selection scenario must install its process value";
  dependencyState.pluginInitialized = false;
  dependencyState.environment["NCCL_DEBUG"] = "TRACE";
  {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    ncclDebugInit();
  }
  const int processEnvironmentLevel = ncclDebugLevel;
  const int restoreResult = hadPreviousValue ? ::setenv("NCCL_DEBUG", previousValueCopy.c_str(), 1)
                                             : ::unsetenv("NCCL_DEBUG");
  EXPECT_EQ(restoreResult, 0) << "The process debug-level environment should be restored after the scenario";
  EXPECT_EQ(processEnvironmentLevel, NCCL_LOG_INFO)
      << "Before plugin initialization, debug initialization should read the process environment instead of the map";
}

/**
 * @brief Verifies debug-file threshold, path expansion, open failure, and reset-time ownership
 * release.
 */
TEST_F(DebugSourceTest, ManagesConfiguredDebugFile) {
  dependencyState.environment["NCCL_DEBUG"] = "VERSION";
  dependencyState.environment["NCCL_DEBUG_FILE"] = "trace-%h-%p.log";
  {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    ncclDebugInit();
  }
  EXPECT_TRUE(dependencyState.fileCalls.empty())
      << "VERSION logging should not create a configured debug file";
  EXPECT_EQ(ncclDebugFile, stdout) << "The output stream should remain stdout below the file-opening threshold";

  ResetSourceStateForTest();
  dependencyState.hostValue = "host%slot/node";
  dependencyState.processId = 808;
  dependencyState.fileOpenAllowed = false;
  dependencyState.environment["NCCL_DEBUG"] = "INFO";
  dependencyState.environment["NCCL_DEBUG_FILE"] = "trace-%h-%p-%%-%q.log";
  const std::string fileSetting = dependencyState.environment["NCCL_DEBUG_FILE"];
  {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    ncclDebugInit();
  }
  ASSERT_EQ(dependencyState.fileCalls.size(), 1U)
      << "INFO initialization should attempt to open exactly one configured debug file";
  EXPECT_EQ(dependencyState.fileCalls[0].path, "trace-host-slot-node-808-%-%q.log")
      << "Hostname, process, literal-percent, and unknown placeholders should expand as documented";
  EXPECT_EQ(dependencyState.fileCalls[0].mode, "w") << "The configured debug file should be opened for replacement";
  EXPECT_EQ(ncclDebugFile, stdout) << "A failed file open should preserve stdout as the output stream";
  EXPECT_EQ(dependencyState.environment["NCCL_DEBUG_FILE"], fileSetting)
      << "Path expansion should not modify the provider-owned environment string";

  ResetSourceStateForTest();
  dependencyState.fileOpenAllowed = true;
  dependencyState.environment["NCCL_DEBUG"] = "INFO";
  dependencyState.environment["NCCL_DEBUG_FILE"] = "accepted.log";
  {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    ncclDebugInit();
  }
  ASSERT_NE(ncclDebugFile, stdout) << "A successful file open should replace the output stream";
  FILE* openedStream = ncclDebugFile;
  ncclResetDebugInitInternal();
  dependencyState.environment.erase("NCCL_DEBUG_FILE");
  {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    ncclDebugInit();
  }
  EXPECT_EQ(dependencyState.fileCloseCalls, 1) << "Reinitialization should close the previously owned debug file once";
  ASSERT_EQ(dependencyState.closedFiles.size(), 1U)
      << "Reinitialization should release exactly one production-owned stream";
  EXPECT_EQ(dependencyState.closedFiles[0], openedStream)
      << "Reinitialization should close the exact stream returned by the configured opener";
  EXPECT_EQ(ncclDebugFile, stdout) << "Reinitialization without a file setting should restore stdout";
}

/**
 * @brief Verifies both reset entry points defer cleanup and make the next log reload
 * environment policy.
 */
TEST_F(DebugSourceTest, ResetEntryPointsReloadPolicyAndReleaseResources) {
  FILE* internalStream = std::tmpfile();
  ASSERT_NE(internalStream, nullptr) << "The internal-reset scenario requires a live owned debug stream";
  ncclDebugFile = internalStream;
  ncclDebugLevel = NCCL_LOG_TRACE;
  ncclDebugMask = NCCL_NET;

  ncclResetDebugInitInternal();
  EXPECT_EQ(ncclDebugFile, internalStream)
      << "The internal reset should defer file ownership release until reinitialization";
  EXPECT_EQ(dependencyState.fileCloseCalls, 0)
      << "The internal reset should not close the active stream before the next initialization";
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "internal-reset.cc", 1, "filtered after reset");
  EXPECT_EQ(ncclDebugLevel, NCCL_LOG_NONE)
      << "The first subsequent log should reload the absent debug-level policy";
  EXPECT_EQ(ncclDebugFile, stdout) << "Internal-reset reinitialization should restore stdout";
  ASSERT_EQ(dependencyState.closedFiles.size(), 1U)
      << "Internal-reset reinitialization should close exactly one previously owned stream";
  EXPECT_EQ(dependencyState.closedFiles[0], internalStream)
      << "Internal-reset reinitialization should close the exact previously owned stream";
  EXPECT_EQ(dependencyState.threadIdCalls, 0)
      << "The INFO message should remain filtered by the newly loaded NONE policy";

  FILE* publicStream = std::tmpfile();
  ASSERT_NE(publicStream, nullptr) << "The public-reset scenario requires a second live owned debug stream";
  ncclDebugFile = publicStream;
  ncclDebugLevel = NCCL_LOG_TRACE;
  ncclDebugMask = NCCL_INIT;
  dependencyState.environment["NCCL_DEBUG"] = "WARN";
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  ncclResetDebugInit();
#pragma GCC diagnostic pop
  EXPECT_EQ(ncclDebugFile, publicStream)
      << "The public reset should also defer file ownership release until reinitialization";
  EXPECT_EQ(dependencyState.fileCloseCalls, 1)
      << "The public reset should not itself close the second active stream";
  ncclDebugLog(NCCL_LOG_INFO, NCCL_INIT, "public-reset.cc", 2, "filtered after reset");
  EXPECT_EQ(ncclDebugLevel, NCCL_LOG_WARN)
      << "The first subsequent log should reload the newly configured WARN policy";
  EXPECT_EQ(ncclDebugFile, stdout) << "Public-reset reinitialization should restore stdout";
  ASSERT_EQ(dependencyState.closedFiles.size(), 2U)
      << "Both reset workflows should release exactly their one previously owned stream";
  EXPECT_EQ(dependencyState.closedFiles[1], publicStream)
      << "Public-reset reinitialization should close the exact second owned stream";
  EXPECT_EQ(dependencyState.threadIdCalls, 0)
      << "Both post-reset INFO messages should be filtered before resolving a thread identifier";
}

/**
 * @brief Verifies level/subsystem filtering, warning error capture, and warning downgrading.
 */
TEST_F(DebugSourceTest, FiltersLogsAndMaintainsLastError) {
  FILE* stream = std::tmpfile();
  ASSERT_NE(stream, nullptr) << "The test requires a seekable temporary output stream";
  ncclDebugFile = stream;
  ncclDebugLevel = NCCL_LOG_INFO;
  ncclDebugMask = NCCL_NET;
  ncclDebugTimestampLevels = 0;
  std::strcpy(hostname, "filter-host");
  pid = 14;
  std::strcpy(ncclLastError, "previous error");

  ncclDebugLog(NCCL_LOG_INFO, NCCL_P2P, "filter", 11, "hidden info %d", 1);
  EXPECT_TRUE(ReadOutputForTest(stream).empty()) << "A log from a disabled subsystem should emit no bytes";
  EXPECT_EQ(dependencyState.threadIdCalls, 0) << "A filtered log should return before resolving the thread identifier";

  ncclDebugLog(NCCL_LOG_TRACE, NCCL_NET, "filter", 11, "hidden trace");
  EXPECT_TRUE(ReadOutputForTest(stream).empty()) << "A log above the configured level should emit no bytes";
  EXPECT_EQ(dependencyState.threadIdCalls, 0)
      << "A level-filtered log should return before resolving the thread identifier";

  ncclDebugLog(NCCL_LOG_WARN, NCCL_P2P, "filter.cc", 12, "hidden warning %d", 2);
  EXPECT_TRUE(ReadOutputForTest(stream).empty()) << "A subsystem-filtered warning should emit no bytes";
  EXPECT_STREQ(ncclLastError, "hidden warning 2")
      << "A warning should update the human-readable last error even when output is filtered";
  EXPECT_EQ(dependencyState.threadIdCalls, 0)
      << "A filtered warning should update the error text without resolving the thread identifier";

  ncclDebugNoWarn = NCCL_NET;
  ncclDebugLog(NCCL_LOG_WARN, NCCL_P2P, "filter.cc", 13, "downgraded %d", 3);
  ncclDebugNoWarn = 0;
  const std::string output = ReadOutputForTest(stream);
  EXPECT_NE(output.find("[-1] NCCL INFO downgraded 3\n"), std::string::npos)
      << "A suppressed warning should be emitted as INFO with the replacement subsystem flags";
  EXPECT_STREQ(ncclLastError, "hidden warning 2")
      << "A warning downgraded before classification should not replace the last warning error";
  EXPECT_EQ(dependencyState.threadIdCalls, 1) << "The first emitted message should resolve the thread identifier once";
}

/**
 * @brief Verifies every output-format branch and variadic substitution in the common logger.
 */
TEST_F(DebugSourceTest, FormatsSupportedLogLevels) {
  FILE* stream = std::tmpfile();
  ASSERT_NE(stream, nullptr) << "The test requires a seekable temporary output stream";
  ncclDebugFile = stream;
  ncclDebugLevel = NCCL_LOG_TRACE;
  ncclDebugMask = NCCL_ALL;
  ncclDebugTimestampLevels = 0;
  std::strcpy(hostname, "format-host");
  pid = 27;
  dependencyState.threadId = 38;

  ncclDebugLog(NCCL_LOG_VERSION, NCCL_ALL, "version.cc", 1, "version %d", 1);
  ncclDebugLog(NCCL_LOG_WARN, NCCL_NET, "warn.cc", 2, "warning %s", "two");
  ncclDebugLog(NCCL_LOG_INFO, NCCL_NET, "info.cc", 3, "information %d", 3);
  ncclDebugLog(NCCL_LOG_ABORT, NCCL_NET, "abort.cc", 4, "abort %d", 4);
  ncclDebugLog(NCCL_LOG_TRACE, NCCL_CALL, "call.cc", 5, "call %d", 5);
  ncclDebugLog(NCCL_LOG_TRACE, NCCL_NET, "trace.cc", 6, "trace %d", 6);

  const std::string output = ReadOutputForTest(stream);
  EXPECT_EQ(output.find("version 1\n"), 0U)
      << "VERSION output should begin with only the substituted message and newline when timestamps are disabled";
  EXPECT_NE(output.find("\nformat-host:27:38 [-1] warn.cc:2 NCCL WARN warning two\n"), std::string::npos)
      << "WARN output should include its leading newline, source location, level marker, and arguments";
  EXPECT_NE(output.find("format-host:27:38 [-1] NCCL INFO information 3\n"), std::string::npos)
      << "INFO output should include the device marker and substituted message";
  EXPECT_NE(output.find("format-host:27:38 abort 4\n"), std::string::npos)
      << "ABORT output should use the generic level format";
  EXPECT_NE(output.find("format-host:27:38 NCCL CALL call 5\n"), std::string::npos)
      << "CALL tracing should use its dedicated format without a device lookup";
  EXPECT_TRUE(std::regex_search(output,
      std::regex("format-host:27:38 \\[-1\\] [0-9]+\\.[0-9]+ trace.cc:6 NCCL TRACE trace 6\\n")))
      << "Ordinary TRACE output should include host, process, thread, device, elapsed time, source, and message";
  EXPECT_STREQ(ncclLastError, "warning two") << "The emitted warning should be retained as the last error";
  EXPECT_EQ(dependencyState.threadIdCalls, 1)
      << "All messages from one test thread should reuse the cached thread identifier";

  ASSERT_EQ(std::fseek(stream, 0, SEEK_END), 0)
      << "The captured stream should be repositioned before appending the boundary message";
  const std::string longFormat(1200, 'x');
  const std::string formatSnapshot = longFormat;
  ncclDebugLog(NCCL_LOG_INFO, NCCL_NET, "long.cc", 7, longFormat.c_str(), 0);
  const std::string boundedOutput = ReadOutputForTest(stream);
  ASSERT_GT(boundedOutput.size(), output.size()) << "The long enabled message should append output";
  EXPECT_LE(boundedOutput.size() - output.size(), 1023U)
      << "A prefixed format longer than the internal output buffer should be truncated to its capacity";
  EXPECT_EQ(boundedOutput.back(), '\n') << "A truncated prefixed format should still terminate with a newline";
  EXPECT_EQ(longFormat, formatSnapshot) << "Logging should not modify the caller-owned format string";
}

/**
 * @brief Verifies disabled/enabled parameter values, loader wiring, and process-lifetime caching.
 */
TEST_F(DebugSourceTest, LoadsThreadNameParameterDirectly) {
  ASSERT_EXIT(
      {
        ResetSourceStateForTest();
        dependencyState.parameterCaching = true;
        dependencyState.parameterValue = 1;
        const int64_t firstValue = ncclParamSetThreadName();
        dependencyState.parameterValue = 0;
        const int64_t secondValue = ncclParamSetThreadName();
        if (firstValue != 1) {
          std::fprintf(stderr, "The accessor did not return the provider's initial enabled value\n");
          std::_Exit(1);
        }
        if (secondValue != 1) {
          std::fprintf(stderr, "The accessor did not retain its process-lifetime cached value\n");
          std::_Exit(1);
        }
        if (dependencyState.loadCalls.size() != 1) {
          std::fprintf(stderr, "The cached accessor consulted its loader more than once\n");
          std::_Exit(1);
        }
        const LoadCall& call = dependencyState.loadCalls[0];
        if (call.environment != "NCCL_SET_THREAD_NAME" || call.defaultValue != 0) {
          std::fprintf(stderr, "The accessor passed an incorrect key or default to its loader\n");
          std::_Exit(1);
        }
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "")
      << "The generated accessor should load once with the declared contract and cache the result";

  ASSERT_EXIT(
      {
        ResetSourceStateForTest();
        dependencyState.parameterCaching = true;
        dependencyState.parameterValue = 0;
        if (ncclParamSetThreadName() != 0) {
          std::fprintf(stderr, "The accessor did not return the provider's disabled value\n");
          std::_Exit(1);
        }
        if (dependencyState.loadCalls.size() != 1 ||
            dependencyState.loadCalls[0].environment != "NCCL_SET_THREAD_NAME" ||
            dependencyState.loadCalls[0].defaultValue != 0) {
          std::fprintf(stderr, "The disabled accessor call used an incorrect loader contract\n");
          std::_Exit(1);
        }
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "")
      << "The generated accessor should return the disabled value with the same declared loader contract";
}

/**
 * @brief Verifies disabled naming, enabled formatting, native-handle forwarding, and length
 * truncation.
 */
TEST_F(DebugSourceTest, SetsThreadNameOnlyWhenEnabled) {
  dependencyState.parameterValue = 0;
  /**
   * @brief Provides an idle thread for the disabled naming path.
   * @note Side effects: Starts a thread that exits immediately.
   */
  std::thread disabledWorker([] {});
  const pthread_t disabledHandle = disabledWorker.native_handle();
  ncclSetThreadName(disabledWorker, "disabled-%d", 1);
  EXPECT_TRUE(disabledWorker.joinable()) << "A disabled naming request should not consume the std::thread reference";
  EXPECT_NE(::pthread_equal(disabledWorker.native_handle(), disabledHandle), 0)
      << "A disabled naming request should preserve the std::thread native handle";
  disabledWorker.join();
  EXPECT_TRUE(dependencyState.nativeNameCalls.empty())
      << "A disabled thread-name parameter should skip the native naming API";

  dependencyState.parameterValue = 2;
  /**
   * @brief Provides an idle thread for the non-enabling parameter path.
   * @note Side effects: Starts a thread that exits immediately.
   */
  std::thread nonUnitWorker([] {});
  ncclSetThreadName(nonUnitWorker, "non-unit");
  nonUnitWorker.join();
  EXPECT_TRUE(dependencyState.nativeNameCalls.empty())
      << "Only the exact parameter value one should enable the native naming API";

  dependencyState.parameterValue = 1;
  /**
   * @brief Provides an idle thread for the enabled naming path.
   * @note Side effects: Starts a thread that exits immediately.
   */
  std::thread enabledWorker([] {});
  const pthread_t enabledHandle = enabledWorker.native_handle();
  char enabledFormat[] = "worker-%d";
  ncclSetThreadName(enabledWorker, enabledFormat, 27);
  EXPECT_TRUE(enabledWorker.joinable()) << "An enabled naming request should not consume the std::thread reference";
  EXPECT_NE(::pthread_equal(enabledWorker.native_handle(), enabledHandle), 0)
      << "An enabled naming request should preserve the std::thread native handle";
  EXPECT_STREQ(enabledFormat, "worker-%d") << "Formatting should not modify the caller-owned format buffer";
  enabledWorker.join();
  ASSERT_EQ(dependencyState.nativeNameCalls.size(), 1U)
      << "An enabled thread-name parameter should invoke the native naming API once";
  EXPECT_NE(::pthread_equal(dependencyState.nativeNameCalls[0].handle, enabledHandle), 0)
      << "The native naming call should receive the selected std::thread handle";
  EXPECT_EQ(dependencyState.nativeNameCalls[0].name, "worker-27")
      << "The variadic thread-name format should be expanded before the native call";

  /**
   * @brief Provides an idle thread for the long-name truncation path.
   * @note Side effects: Starts a thread that exits immediately.
   */
  std::thread longNameWorker([] {});
  ncclSetThreadName(longNameWorker, "abcdefghijklmnop-%d", 9);
  longNameWorker.join();
  ASSERT_EQ(dependencyState.nativeNameCalls.size(), 2U)
      << "Each enabled naming request should invoke the native API exactly once";
  EXPECT_EQ(dependencyState.nativeNameCalls[1].name, "abcdefghijklmno")
      << "A long name should be truncated to fifteen bytes plus the required terminator";
}

}  // namespace
