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

#include <gtest/gtest.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pwd.h>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

/** @brief Records one parameter-file open request and its fake stream. */
struct FileOpenCall {
  std::string path;
  std::string mode;
  FILE* stream = nullptr;
};

/** @brief Records one parsed environment name and value. */
struct EnvironmentAssignment {
  std::string name;
  std::string value;
};

/**
 * @brief Owns virtual streams and records user, file, assignment, and environment-plugin
 * dependency interactions.
 */
struct DependencyState {
  uid_t userId = 1001;
  bool passwordEntryAvailable = true;
  bool homeDirectoryAvailable = true;
  std::string homeDirectory = "/home/tester";
  struct passwd passwordEntry = {};
  std::vector<uid_t> passwordLookups;
  int userIdLookups = 0;
  std::unordered_map<std::string, std::string> fileContents;
  std::vector<FileOpenCall> fileOpenCalls;
  std::vector<FILE*> activeFiles;
  std::vector<FILE*> closedFiles;
  std::vector<EnvironmentAssignment> assignments;
  std::unordered_map<std::string, std::string> pluginEnvironment;
  std::vector<std::string> pluginLookups;
  int pluginInitCalls = 0;
};

DependencyState dependencyState;

/**
 * @brief Closes every still-owned virtual file before restoring deterministic dependency state.
 * @note Side effects: Closes active streams and replaces dependencyState.
 */
void ResetDependencyStateForTest() {
  for (FILE* stream : dependencyState.activeFiles) std::fclose(stream);
  dependencyState = DependencyState{};
}

/**
 * @brief Restores a process environment variable to its captured value or absence.
 */
class ProcessEnvironmentGuard {
 public:
  /**
   * @brief Captures the value and presence of one process environment variable.
   * @param[in] name Valid non-NULL variable name copied by the guard.
   * @note The constructor does not modify the process environment.
   */
  explicit ProcessEnvironmentGuard(const char* name) : name_(name) {
    const char* value = std::getenv(name_.c_str());
    if (value != nullptr) {
      hadValue_ = true;
      value_ = value;
    }
  }

  /**
   * @brief Restores the captured environment state.
   * @note Side effects: Sets or removes the guarded process environment variable.
   */
  ~ProcessEnvironmentGuard() {
    if (hadValue_) {
      setenv(name_.c_str(), value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  /**
   * @brief Overrides the guarded variable for one scoped scenario.
   * @param[in] value Value copied into the process environment.
   * @note Side effects: Sets the guarded process environment variable.
   */
  void Set(const std::string& value) { setenv(name_.c_str(), value.c_str(), 1); }

  /**
   * @brief Makes the guarded variable absent for one scoped scenario.
   * @note Side effects: Removes the guarded process environment variable.
   */
  void Unset() { unsetenv(name_.c_str()); }

 private:
  std::string name_;
  bool hadValue_ = false;
  std::string value_;
};

}  // namespace

/**
 * @brief Supplies the deterministic user identifier consumed by the production source.
 * @return dependencyState.userId.
 * @note Side effects: Increments dependencyState.userIdLookups.
 */
uid_t UserIdForTest() {
  dependencyState.userIdLookups++;
  return dependencyState.userId;
}

/**
 * @brief Returns a deterministic password entry and records the requested identifier.
 * @param[in] userId Identifier recorded without interpretation.
 * @return A borrowed entry valid until dependencyState changes, or NULL when unavailable.
 * @note Side effects: Records userId and updates the borrowed entry's home-directory pointer.
 */
struct passwd* PasswordEntryForTest(uid_t userId) {
  dependencyState.passwordLookups.push_back(userId);
  if (!dependencyState.passwordEntryAvailable) return nullptr;
  dependencyState.passwordEntry.pw_dir = dependencyState.homeDirectoryAvailable
      ? const_cast<char*>(dependencyState.homeDirectory.c_str())
      : nullptr;
  return &dependencyState.passwordEntry;
}

/**
 * @brief Opens a virtual configuration file and records its path, mode, and stream identity.
 * @param[in] path Configuration path; NULL is recorded as an empty string.
 * @param[in] mode Open mode; NULL is recorded as an empty string.
 * @return A caller-owned readable stream when configured and created; otherwise NULL.
 * @note Side effects: Records the attempt and may create, populate, rewind, and track a stream.
 */
FILE* OpenParameterFileForTest(const char* path, const char* mode) {
  const std::string pathValue = path == nullptr ? "" : path;
  const std::string modeValue = mode == nullptr ? "" : mode;
  const auto contents = dependencyState.fileContents.find(pathValue);
  if (contents == dependencyState.fileContents.end()) {
    dependencyState.fileOpenCalls.push_back({pathValue, modeValue, nullptr});
    return nullptr;
  }

  FILE* stream = std::tmpfile();
  if (stream != nullptr) {
    std::fwrite(contents->second.data(), 1, contents->second.size(), stream);
    std::rewind(stream);
    dependencyState.activeFiles.push_back(stream);
  }
  dependencyState.fileOpenCalls.push_back({pathValue, modeValue, stream});
  return stream;
}

/**
 * @brief Releases a configuration stream and records the ownership transfer.
 * @param[in] stream Valid open stream owned by the caller.
 * @return The result of fclose(stream).
 * @note Side effects: Records and closes stream and removes it from activeFiles.
 */
int CloseParameterFileForTest(FILE* stream) {
  dependencyState.closedFiles.push_back(stream);
  const auto active = std::find(
      dependencyState.activeFiles.begin(), dependencyState.activeFiles.end(), stream);
  if (active != dependencyState.activeFiles.end()) dependencyState.activeFiles.erase(active);
  return std::fclose(stream);
}

/**
 * @brief Captures one parsed environment assignment without applying it.
 * @param[in] name Assignment name; NULL is recorded as an empty string.
 * @param[in] value Assignment value; NULL is recorded as an empty string.
 * @note Side effects: Appends one entry to dependencyState.assignments.
 */
void SetOsEnvironmentForTest(const char* name, const char* value) {
  dependencyState.assignments.push_back(
      {name == nullptr ? "" : name, value == nullptr ? "" : value});
}

/**
 * @brief Models successful environment-plugin initialization.
 * @note Side effects: Increments dependencyState.pluginInitCalls.
 */
void InitializeEnvironmentPluginForTest() {
  dependencyState.pluginInitCalls++;
}

/**
 * @brief Looks up and records one deterministic plugin-environment key.
 * @param[in] name Lookup key; NULL is recorded as an empty string.
 * @return A borrowed value valid until its entry or dependencyState changes, or NULL.
 * @note Side effects: Appends the normalized key to dependencyState.pluginLookups.
 */
const char* PluginEnvironmentValueForTest(const char* name) {
  const std::string nameValue = name == nullptr ? "" : name;
  dependencyState.pluginLookups.push_back(nameValue);
  const auto value = dependencyState.pluginEnvironment.find(nameValue);
  return value == dependencyState.pluginEnvironment.end() ? nullptr : value->second.c_str();
}

int debugLevelForTest = NCCL_LOG_WARN;
unsigned long debugMaskForTest = NCCL_ALL;

/**
 * @brief Discards diagnostics emitted by the parameter implementation.
 * @param[in] level Ignored log severity.
 * @param[in] flags Ignored subsystem flags.
 * @param[in] file Ignored caller source file.
 * @param[in] line Ignored caller source line.
 * @param[in] format Ignored printf-style format; trailing arguments are also ignored.
 * @note Side effects: None.
 */
void ParameterDebugLogForTest(
    ncclDebugLogLevel level, unsigned long flags, const char* file, int line,
    const char* format, ...) {
  (void)level;
  (void)flags;
  (void)file;
  (void)line;
  (void)format;
}

/**
 * @brief Compiles the parameter implementation against virtual user, file, environment,
 * and logging dependencies.
 */
#define getuid UserIdForTest
#define getpwuid PasswordEntryForTest
#define fopen OpenParameterFileForTest
#define fclose CloseParameterFileForTest
#define ncclOsSetEnv SetOsEnvironmentForTest
#define ncclInitEnv InitializeEnvironmentPluginForTest
#define ncclEnvPluginGetEnv PluginEnvironmentValueForTest
#define ncclDebugLevel debugLevelForTest
#define ncclDebugMask debugMaskForTest
#define ncclDebugLog ParameterDebugLogForTest
#include "../../../misc/param.cc"
#undef ncclDebugLog
#undef ncclDebugMask
#undef ncclDebugLevel
#undef ncclEnvPluginGetEnv
#undef ncclInitEnv
#undef ncclOsSetEnv
#undef fclose
#undef fopen
#undef getpwuid
#undef getuid

namespace {

/** @brief Provides isolated dependency and no-cache state for parameter tests. */
class ParamTest : public ::testing::Test {
 protected:
  /**
   * @brief Starts a test with deterministic dependency and no-cache state.
   * @note Side effects: Closes active streams, replaces dependencyState, clears
   * noCacheSet, and resets noCacheAll.
   */
  void SetUp() override {
    ResetDependencyStateForTest();
    noCacheSet.clear();
    noCacheAll = false;
  }

  /**
   * @brief Releases virtual streams still owned after a test.
   * @note Side effects: Closes active streams and replaces dependencyState.
   */
  void TearDown() override { ResetDependencyStateForTest(); }
};

/**
 * @brief Verifies home-directory success and lookup failure, including UID forwarding.
 */
TEST_F(ParamTest, ResolvesUserHomeDirectory) {
  dependencyState.userId = 4242;
  dependencyState.homeDirectory = "/users/direct-test";

  const char* home = userHomeDir();
  ASSERT_NE(home, nullptr) << "a successful password lookup must expose its home directory";
  EXPECT_STREQ(home, "/users/direct-test")
      << "the returned home directory must match the password entry";
  EXPECT_EQ(dependencyState.userIdLookups, 1)
      << "home lookup must request the current UID exactly once";
  ASSERT_EQ(dependencyState.passwordLookups.size(), 1u)
      << "home lookup must perform exactly one password-database lookup";
  EXPECT_EQ(dependencyState.passwordLookups[0], static_cast<uid_t>(4242))
      << "the password lookup must receive the UID returned by getuid";

  ResetDependencyStateForTest();
  dependencyState.passwordEntryAvailable = false;
  EXPECT_EQ(userHomeDir(), nullptr)
      << "a missing password entry must be reported as a missing home directory";
  EXPECT_EQ(dependencyState.userIdLookups, 1)
      << "the failure path must still obtain the current UID once";
  ASSERT_EQ(dependencyState.passwordLookups.size(), 1u)
      << "the failure path must perform one password lookup";
  EXPECT_EQ(dependencyState.passwordLookups[0], dependencyState.userId)
      << "the failure path must forward the current UID unchanged";
}

/**
 * @brief Verifies file-open failure, parsing, bounded values, ordering, and stream release.
 */
TEST_F(ParamTest, ParsesEnvironmentFileAndReleasesStream) {
  char missingPath[] = "/virtual/missing.conf";
  const std::string missingSnapshot = missingPath;
  setEnvFile(missingPath);

  ASSERT_EQ(dependencyState.fileOpenCalls.size(), 1u)
      << "a missing configuration file must cause one open attempt";
  EXPECT_EQ(dependencyState.fileOpenCalls[0].path, missingSnapshot)
      << "the open attempt must use the caller's complete path";
  EXPECT_EQ(dependencyState.fileOpenCalls[0].mode, "r")
      << "configuration files must be opened for reading";
  EXPECT_EQ(dependencyState.fileOpenCalls[0].stream, nullptr)
      << "the missing-file seam must exercise the open-failure path";
  EXPECT_TRUE(dependencyState.assignments.empty())
      << "open failure must not create environment assignments";
  EXPECT_TRUE(dependencyState.closedFiles.empty())
      << "open failure must not close a stream it does not own";
  EXPECT_EQ(std::string(missingPath), missingSnapshot)
      << "setEnvFile must not modify its input path";

  ResetDependencyStateForTest();
  char path[] = "/virtual/parameters.conf";
  const std::string pathSnapshot = path;
  const std::string longValue(1024, 'x');
  dependencyState.fileContents[path] =
      "# comment\n"
      "\n"
      "MALFORMED\n"
      "ALPHA=one\n"
      "EMPTY=\n"
      "EQUALS=a=b=c\n"
      "LONG=" + longValue + "\n"
      "FINAL=end";

  setEnvFile(path);

  ASSERT_EQ(dependencyState.fileOpenCalls.size(), 1u)
      << "one configured file must be opened exactly once";
  ASSERT_NE(dependencyState.fileOpenCalls[0].stream, nullptr)
      << "the configured virtual file must provide a readable stream";
  EXPECT_EQ(dependencyState.fileOpenCalls[0].path, pathSnapshot)
      << "the parser must open the exact caller-supplied path";
  EXPECT_EQ(dependencyState.fileOpenCalls[0].mode, "r")
      << "the parser must request read mode";
  ASSERT_EQ(dependencyState.assignments.size(), 5u)
      << "comments, blank lines, and lines without '=' must be ignored";
  EXPECT_EQ(dependencyState.assignments[0].name, "ALPHA")
      << "the first valid variable name must be parsed before '='";
  EXPECT_EQ(dependencyState.assignments[0].value, "one")
      << "the first valid variable value must be parsed after '='";
  EXPECT_EQ(dependencyState.assignments[1].name, "EMPTY")
      << "an explicit empty value must retain its variable name";
  EXPECT_TRUE(dependencyState.assignments[1].value.empty())
      << "an explicit empty value must remain empty";
  EXPECT_EQ(dependencyState.assignments[2].name, "EQUALS")
      << "only the first '=' must delimit the variable name";
  EXPECT_EQ(dependencyState.assignments[2].value, "a=b=c")
      << "additional '=' characters must remain part of the value";
  EXPECT_EQ(dependencyState.assignments[3].name, "LONG")
      << "the bounded-value case must retain its variable name";
  EXPECT_EQ(dependencyState.assignments[3].value, std::string(1023, 'x'))
      << "values exceeding the fixed destination must be truncated and terminated at 1023 bytes";
  EXPECT_EQ(dependencyState.assignments[4].name, "FINAL")
      << "a final line without a newline must still be parsed";
  EXPECT_EQ(dependencyState.assignments[4].value, "end")
      << "the final line's value must be preserved";
  ASSERT_EQ(dependencyState.closedFiles.size(), 1u)
      << "an opened configuration stream must be closed exactly once";
  EXPECT_EQ(dependencyState.closedFiles[0], dependencyState.fileOpenCalls[0].stream)
      << "the parser must close the exact stream it opened";
  EXPECT_TRUE(dependencyState.activeFiles.empty())
      << "no opened configuration stream may remain owned after parsing";
  EXPECT_EQ(std::string(path), pathSnapshot)
      << "parsing must not modify the caller's file-name buffer";
}

/**
 * @brief Verifies custom, home-derived, and system configuration source selection and order.
 */
TEST_F(ParamTest, SelectsConfigurationSources) {
  ProcessEnvironmentGuard guard("NCCL_CONF_FILE");
  const std::string customPath = "/virtual/custom.conf";
  guard.Set(customPath);

  initEnvFunc();

  ASSERT_EQ(dependencyState.fileOpenCalls.size(), 2u)
      << "a custom configuration must be followed by the system configuration";
  EXPECT_EQ(dependencyState.fileOpenCalls[0].path, customPath)
      << "the explicit configuration path must have first precedence";
  EXPECT_EQ(dependencyState.fileOpenCalls[1].path, "/etc/nccl.conf")
      << "the system configuration must always be considered last";
  EXPECT_EQ(dependencyState.userIdLookups, 0)
      << "an explicit configuration path must bypass home-directory lookup";
  const char* customAfter = std::getenv("NCCL_CONF_FILE");
  ASSERT_NE(customAfter, nullptr)
      << "initEnvFunc must not remove the caller's configuration setting";
  EXPECT_EQ(std::string(customAfter), customPath)
      << "initEnvFunc must preserve the configuration setting text";

  ResetDependencyStateForTest();
  dependencyState.homeDirectory = "/home/config-user";
  guard.Set("");
  initEnvFunc();

  ASSERT_EQ(dependencyState.fileOpenCalls.size(), 2u)
      << "an empty custom setting must select the home and system configurations";
  EXPECT_EQ(dependencyState.fileOpenCalls[0].path, "/home/config-user/.nccl.conf")
      << "the user configuration must be derived from the resolved home directory";
  EXPECT_EQ(dependencyState.fileOpenCalls[1].path, "/etc/nccl.conf")
      << "the system configuration must follow the home configuration";
  EXPECT_EQ(dependencyState.userIdLookups, 1)
      << "the home fallback must resolve the current user exactly once";
  const char* emptyAfter = std::getenv("NCCL_CONF_FILE");
  ASSERT_NE(emptyAfter, nullptr)
      << "an explicitly empty configuration setting must remain present";
  EXPECT_STREQ(emptyAfter, "")
      << "the empty configuration setting must remain unchanged";

  ResetDependencyStateForTest();
  dependencyState.passwordEntryAvailable = false;
  guard.Unset();
  initEnvFunc();

  ASSERT_EQ(dependencyState.fileOpenCalls.size(), 1u)
      << "without a custom path or home directory, only the system file is applicable";
  EXPECT_EQ(dependencyState.fileOpenCalls[0].path, "/etc/nccl.conf")
      << "the system configuration path must remain the final fallback";
  EXPECT_EQ(dependencyState.userIdLookups, 1)
      << "the absent-home partition must attempt user resolution once";
  EXPECT_EQ(std::getenv("NCCL_CONF_FILE"), nullptr)
      << "initialization must not synthesize the absent process setting";
}

/**
 * @brief Verifies that concurrent and repeated initialization selects each source only once.
 */
TEST_F(ParamTest, InitializesEnvironmentExactlyOnce) {
  /**
   * @brief Uses a fresh process because initEnv one-time state cannot be reset safely
   * between tests.
   * @note Side effects: Runs concurrent initialization calls in an isolated child process.
   */
  ASSERT_EXIT(
      {
        ResetDependencyStateForTest();
        setenv("NCCL_CONF_FILE", "/virtual/once.conf", 1);

        std::vector<std::thread> callers;
        for (int index = 0; index < 8; index++) callers.emplace_back([] { initEnv(); });
        for (std::thread& caller : callers) caller.join();
        initEnv();

        if (dependencyState.fileOpenCalls.size() != 2u) {
          std::fprintf(stderr, "initEnv opened %zu files instead of exactly two\n",
              dependencyState.fileOpenCalls.size());
          std::_Exit(1);
        }
        if (dependencyState.fileOpenCalls[0].path != "/virtual/once.conf" ||
            dependencyState.fileOpenCalls[1].path != "/etc/nccl.conf") {
          std::fprintf(stderr, "initEnv selected configuration sources in the wrong order\n");
          std::_Exit(1);
        }
        if (dependencyState.fileOpenCalls[0].mode != "r" ||
            dependencyState.fileOpenCalls[1].mode != "r") {
          std::fprintf(stderr, "initEnv did not open both sources for reading\n");
          std::_Exit(1);
        }
        if (dependencyState.userIdLookups != 0) {
          std::fprintf(stderr, "initEnv resolved a home directory despite an explicit file\n");
          std::_Exit(1);
        }
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "")
      << "initEnv must execute configuration discovery once across concurrent and repeated callers";
}

/**
 * @brief Verifies missing, empty, named, and global no-cache settings through policy outputs.
 */
TEST_F(ParamTest, BuildsNoCachePolicy) {
  /**
   * @brief Runs each policy partition in a fresh process so mutations of the shared
   * no-cache policy cannot leak between scenarios.
   * @note Side effects: Executes each policy scenario in an isolated child process.
   */
  ASSERT_EXIT(
      {
        ResetDependencyStateForTest();
        int8_t missingPolicy = -1;
        int8_t emptyPolicy = -1;

        ncclGetEnvNoCacheOnce();
        ncclGetCachePolicy("PARAM_A", &missingPolicy);
        dependencyState.pluginEnvironment["NCCL_NO_CACHE"] = "";
        ncclGetEnvNoCacheOnce();
        ncclGetCachePolicy("PARAM_A", &emptyPolicy);

        if (missingPolicy != 0 || emptyPolicy != 0) {
          std::fprintf(stderr, "missing or empty no-cache settings disabled caching\n");
          std::_Exit(1);
        }
        if (dependencyState.pluginInitCalls != 2 ||
            dependencyState.pluginLookups.size() != 2u) {
          std::fprintf(stderr, "no-cache parsing did not perform two exact provider lookups\n");
          std::_Exit(1);
        }
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "")
      << "missing and empty no-cache settings must leave caching enabled";

  ASSERT_EXIT(
      {
        ResetDependencyStateForTest();
        dependencyState.pluginEnvironment["NCCL_NO_CACHE"] = "PARAM_A,PARAM_B";
        const std::string sourceSnapshot =
            dependencyState.pluginEnvironment["NCCL_NO_CACHE"];
        int8_t firstPolicy = -1;
        int8_t secondPolicy = -1;
        int8_t unrelatedPolicy = -1;
        int8_t prefixPolicy = -1;

        ncclGetEnvNoCacheOnce();
        ncclGetCachePolicy("PARAM_A", &firstPolicy);
        ncclGetCachePolicy("PARAM_B", &secondPolicy);
        ncclGetCachePolicy("PARAM_C", &unrelatedPolicy);
        ncclGetCachePolicy("PARAM", &prefixPolicy);

        if (firstPolicy != 1 || secondPolicy != 1 || unrelatedPolicy != 0 ||
            prefixPolicy != 0) {
          std::fprintf(stderr, "named no-cache policy did not use exact token matching\n");
          std::_Exit(1);
        }
        if (dependencyState.pluginEnvironment["NCCL_NO_CACHE"] != sourceSnapshot) {
          std::fprintf(stderr, "no-cache parsing modified provider-owned input text\n");
          std::_Exit(1);
        }
        if (dependencyState.pluginInitCalls != 1 ||
            dependencyState.pluginLookups.size() != 1u ||
            dependencyState.pluginLookups[0] != "NCCL_NO_CACHE") {
          std::fprintf(stderr, "named no-cache parsing used the provider incorrectly\n");
          std::_Exit(1);
        }
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "")
      << "named no-cache entries must disable only exact listed parameters and preserve input";

  ASSERT_EXIT(
      {
        ResetDependencyStateForTest();
        dependencyState.pluginEnvironment["NCCL_NO_CACHE"] = "ALL";
        int8_t firstPolicy = -1;
        int8_t arbitraryPolicy = -1;

        ncclGetEnvNoCacheOnce();
        ncclGetCachePolicy("PARAM_A", &firstPolicy);
        ncclGetCachePolicy("UNLISTED_VALUE", &arbitraryPolicy);

        if (firstPolicy != 1 || arbitraryPolicy != 1) {
          std::fprintf(stderr, "ALL did not disable caching for every queried parameter\n");
          std::_Exit(1);
        }
        if (dependencyState.pluginEnvironment["NCCL_NO_CACHE"] != "ALL") {
          std::fprintf(stderr, "ALL parsing modified provider-owned input text\n");
          std::_Exit(1);
        }
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "")
      << "ALL must disable caching for both listed and arbitrary parameters";
}

/**
 * @brief Verifies parsed values, defaults, overflow handling, and cache/no-cache behavior.
 */
TEST_F(ParamTest, LoadsParametersAccordingToPolicy) {
  /**
   * @brief Isolates this matrix from previously initialized process-wide no-cache policy state.
   * @note Side effects: Executes parameter loading in a child process.
   */
  ASSERT_EXIT(
      {
        ResetDependencyStateForTest();
        dependencyState.pluginEnvironment["NCCL_NO_CACHE"] = "DYNAMIC_VALUE";
        dependencyState.pluginEnvironment["CACHED_VALUE"] = "0x2a";
        char cachedName[] = "CACHED_VALUE";
        const std::string cachedNameSnapshot = cachedName;
        constexpr int64_t uninitialized = INT64_MIN;
        int64_t cachedStorage = uninitialized;
        int8_t cachedPolicy = -1;

        const int64_t cachedFirst = ncclLoadParam(
            cachedName, 5, uninitialized, &cachedStorage, &cachedPolicy);
        dependencyState.pluginEnvironment["CACHED_VALUE"] = "99";
        const int64_t cachedSecond = ncclLoadParam(
            cachedName, 5, uninitialized, &cachedStorage, &cachedPolicy);
        if (cachedFirst != 42 || cachedSecond != 42 || cachedStorage != 42 ||
            cachedPolicy != 0) {
          std::fprintf(stderr, "cached parameter did not retain its first parsed value\n");
          std::_Exit(1);
        }
        if (std::string(cachedName) != cachedNameSnapshot) {
          std::fprintf(stderr, "parameter loading modified the environment-name input\n");
          std::_Exit(1);
        }

        int64_t dynamicStorage = uninitialized;
        int8_t dynamicPolicy = -1;
        dependencyState.pluginEnvironment["DYNAMIC_VALUE"] = "7";
        const int64_t dynamicFirst = ncclLoadParam(
            "DYNAMIC_VALUE", 3, uninitialized, &dynamicStorage, &dynamicPolicy);
        dependencyState.pluginEnvironment["DYNAMIC_VALUE"] = "8";
        const int64_t dynamicSecond = ncclLoadParam(
            "DYNAMIC_VALUE", 3, uninitialized, &dynamicStorage, &dynamicPolicy);
        if (dynamicFirst != 7 || dynamicSecond != 8 ||
            dynamicStorage != uninitialized || dynamicPolicy != 1) {
          std::fprintf(stderr, "no-cache parameter did not reload without mutating cache storage\n");
          std::_Exit(1);
        }

        int64_t absentStorage = uninitialized;
        int8_t absentPolicy = -1;
        const int64_t absentValue = ncclLoadParam(
            "ABSENT_VALUE", 17, uninitialized, &absentStorage, &absentPolicy);
        if (absentValue != 17 || absentStorage != 17 || absentPolicy != 0) {
          std::fprintf(stderr, "absent parameter did not return and cache its default\n");
          std::_Exit(1);
        }

        dependencyState.pluginEnvironment["OVERFLOW_VALUE"] =
            "999999999999999999999999999999999999";
        int64_t overflowStorage = uninitialized;
        int8_t overflowPolicy = -1;
        const int64_t overflowValue = ncclLoadParam(
            "OVERFLOW_VALUE", 23, uninitialized, &overflowStorage, &overflowPolicy);
        if (overflowValue != 23 || overflowStorage != 23 || overflowPolicy != 0) {
          std::fprintf(stderr, "overflowing parameter did not fall back to and cache its default\n");
          std::_Exit(1);
        }

        const int cachedLookups = static_cast<int>(std::count(
            dependencyState.pluginLookups.begin(), dependencyState.pluginLookups.end(),
            "CACHED_VALUE"));
        const int dynamicLookups = static_cast<int>(std::count(
            dependencyState.pluginLookups.begin(), dependencyState.pluginLookups.end(),
            "DYNAMIC_VALUE"));
        if (cachedLookups != 1 || dynamicLookups != 2) {
          std::fprintf(stderr, "cache policy caused an incorrect number of provider reads\n");
          std::_Exit(1);
        }
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "")
      << "ncclLoadParam must implement value parsing, defaults, errors, and cache policy";
}

/**
 * @brief Verifies environment-plugin initialization, lookup forwarding, results, and input
 * preservation.
 */
TEST_F(ParamTest, ForwardsEnvironmentLookup) {
  dependencyState.pluginEnvironment["PRESENT_VALUE"] = "configured";
  char presentName[] = "PRESENT_VALUE";
  const std::string presentSnapshot = presentName;

  const char* present = ncclGetEnv(presentName);
  ASSERT_NE(present, nullptr)
      << "an existing provider entry must produce a non-NULL result";
  EXPECT_STREQ(present, "configured")
      << "the public lookup must return the provider's value";
  EXPECT_EQ(std::string(presentName), presentSnapshot)
      << "the public lookup must not modify the environment name";

  const char* missing = ncclGetEnv("MISSING_VALUE");
  EXPECT_EQ(missing, nullptr)
      << "a missing provider entry must remain a NULL result";
  EXPECT_EQ(dependencyState.pluginInitCalls, 2)
      << "each public lookup must initialize the environment provider before access";
  ASSERT_EQ(dependencyState.pluginLookups.size(), 2u)
      << "two public lookups must cause exactly two provider accesses";
  EXPECT_EQ(dependencyState.pluginLookups[0], "PRESENT_VALUE")
      << "the first environment name must be forwarded unchanged";
  EXPECT_EQ(dependencyState.pluginLookups[1], "MISSING_VALUE")
      << "the missing environment name must also be forwarded unchanged";
}

}  // namespace
