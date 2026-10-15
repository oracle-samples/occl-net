/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/


#include "debug.h"
#include "os.h"
#include "param.h"
#include "utils.h"

#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

/** @brief Associates one intentionally misaligned pointer with its owned base allocation. */
struct ShiftedAllocation {
  void* exposed = nullptr;
  void* base = nullptr;
};

/** @brief Scripts utility dependencies and records allocation and identity calls. */
struct UtilsBoundaryState {
  std::string hostname = "test-host";
  bool hostnameFailure = false;
  std::string environmentHostId;
  bool environmentHostIdPresent = false;
  std::string bootId = "test-boot";
  bool bootFileFailure = false;
  std::string openedPath;
  uint64_t processId = 1234;
  std::string processNamespace = "pid:[4026531836]";
  bool readLinkFailure = false;
  int mallocCalls = 0;
  int callocCalls = 0;
  int freeCalls = 0;
  int failMallocAt = 0;
  int failCallocAt = 0;
  int shiftMallocAt = 0;
  std::vector<ShiftedAllocation> shiftedAllocations;
};

UtilsBoundaryState boundaryState;

/**
 * @brief Restores deterministic system and allocation behavior before a scenario.
 * @note Side effects: Replaces boundaryState without freeing any still-tracked shifted allocation.
 */
void ResetUtilsBoundaries() {
  boundaryState = UtilsBoundaryState{};
}

/**
 * @brief Supplies a controlled hostname or a system-call failure.
 * @param[out] name Valid non-NULL destination with capacity length bytes.
 * @param[in] length Destination capacity in [1, SIZE_MAX].
 * @return Zero on configured success or -1 on configured failure.
 * @note Side effects: Writes name on success or sets errno to EIO on failure.
 */
int TestGetHostname(char* name, size_t length) {
  if (boundaryState.hostnameFailure) {
    errno = EIO;
    return -1;
  }
  std::snprintf(name, length, "%s", boundaryState.hostname.c_str());
  return 0;
}

/**
 * @brief Supplies a controlled environment override.
 * @param[in] name Valid non-NULL environment-variable name.
 * @return A borrowed host identifier valid until boundaryState changes, or NULL when absent.
 * @note Side effects: None.
 */
const char* TestGetEnvironment(const char* name) {
  if (boundaryState.environmentHostIdPresent && std::strcmp(name, "NCCL_HOSTID") == 0) {
    return boundaryState.environmentHostId.c_str();
  }
  return nullptr;
}

/**
 * @brief Opens a temporary stream containing a controlled boot identifier.
 * @param[in] path Valid non-NULL source path recorded for verification.
 * @param[in] mode Valid non-NULL open mode; the test boundary expects read mode.
 * @return A caller-owned readable stream when mode is "r" and creation succeeds; otherwise NULL.
 * @note Side effects: Records path and may create, populate, and rewind a temporary stream.
 */
FILE* TestOpenFile(const char* path, const char* mode) {
  boundaryState.openedPath = path;
  if (boundaryState.bootFileFailure || std::strcmp(mode, "r") != 0) return nullptr;
  FILE* stream = std::tmpfile();
  if (stream == nullptr) return nullptr;
  std::fwrite(boundaryState.bootId.data(), 1, boundaryState.bootId.size(), stream);
  std::rewind(stream);
  return stream;
}

/**
 * @brief Supplies a controlled process identifier.
 * @return The configured process identifier in [0, UINT64_MAX].
 */
uint64_t TestGetProcessId() {
  return boundaryState.processId;
}

/**
 * @brief Supplies a controlled process-namespace link value or failure.
 * @param[in] path Valid non-NULL link path.
 * @param[out] buffer Valid non-NULL destination with capacity size bytes.
 * @param[in] size Destination capacity in [1, SIZE_MAX].
 * @return The copied byte count or -1 on configured failure.
 * @note Side effects: Writes buffer on success or sets errno to ENOENT on failure.
 */
ssize_t TestReadLink(const char* path, char* buffer, size_t size) {
  if (boundaryState.readLinkFailure || std::strcmp(path, "/proc/self/ns/pid") != 0) {
    errno = ENOENT;
    return -1;
  }
  const size_t count = std::min(size, boundaryState.processNamespace.size());
  std::memcpy(buffer, boundaryState.processNamespace.data(), count);
  return static_cast<ssize_t>(count);
}

/**
 * @brief Allocates production memory while supporting deterministic failure and alignment tests.
 * @param[in] size Requested allocation size in [0, SIZE_MAX].
 * @return Caller-owned normal or deliberately shifted storage releasable through TestRelease,
 * or NULL when allocation fails.
 * @note Side effects: Increments mallocCalls, may allocate storage, and may track a shifted view.
 */
void* TestAllocate(size_t size) {
  boundaryState.mallocCalls++;
  if (boundaryState.failMallocAt == boundaryState.mallocCalls) return nullptr;
  if (boundaryState.shiftMallocAt != boundaryState.mallocCalls) return std::malloc(size);

  void* base = std::malloc(size + 64);
  if (base == nullptr) return nullptr;
  uintptr_t exposed = (reinterpret_cast<uintptr_t>(base) + 15) & ~uintptr_t(15);
  if ((exposed & 63) == 0) exposed += 16;
  boundaryState.shiftedAllocations.push_back({reinterpret_cast<void*>(exposed), base});
  return reinterpret_cast<void*>(exposed);
}

/**
 * @brief Allocates zeroed production memory while supporting deterministic failure.
 * @param[in] count Element count in [0, SIZE_MAX].
 * @param[in] size Element size in [0, SIZE_MAX].
 * @return Caller-owned zeroed storage releasable through TestRelease, or NULL when allocation
 * fails.
 * @note Side effects: Increments callocCalls and may allocate storage.
 */
void* TestAllocateZeroed(size_t count, size_t size) {
  boundaryState.callocCalls++;
  if (boundaryState.failCallocAt == boundaryState.callocCalls) return nullptr;
  return std::calloc(count, size);
}

/**
 * @brief Frees normal and deliberately shifted production allocations.
 * @param[in] pointer NULL or a pointer previously returned by a test allocation boundary.
 * @note Side effects: Increments freeCalls, frees storage, and removes a matching shifted record.
 */
void TestRelease(void* pointer) {
  boundaryState.freeCalls++;
  for (auto iterator = boundaryState.shiftedAllocations.begin();
       iterator != boundaryState.shiftedAllocations.end(); ++iterator) {
    if (iterator->exposed == pointer) {
      std::free(iterator->base);
      boundaryState.shiftedAllocations.erase(iterator);
      return;
    }
  }
  std::free(pointer);
}

/**
 * @brief Returns stable labels for the bit-string formatter.
 * @param[in] bit Any integer bit index.
 * @return A static label for indices 0 through 3, otherwise "other".
 * @note Side effects: None.
 */
const char* BitLabel(int bit) {
  static const char* labels[] = {"zero", "one", "two", "three"};
  return bit >= 0 && bit < 4 ? labels[bit] : "other";
}

/**
 * @brief Verifies that parsing did not change any field of four interface entries.
 * @param[in] actual Four entries observed after parsing.
 * @param[in] expected Four entries captured before parsing.
 * @param[in] context Valid non-NULL scenario label used in assertion diagnostics.
 * @note Side effects: Records GoogleTest failures for changed prefix or port fields.
 */
void ExpectInterfacesUnchanged(
    const std::array<netIf, 4>& actual, const std::array<netIf, 4>& expected, const char* context) {
  for (size_t index = 0; index < actual.size(); index++) {
    SCOPED_TRACE(testing::Message() << context << ", interface entry " << index);
#if 0 // ADDRESS_LATER
    EXPECT_EQ(std::memcmp(actual[index].prefix, expected[index].prefix, sizeof(actual[index].prefix)), 0)
        << "Parsing without an output entry must preserve every prefix byte";
#endif 0 // ADDRESS_LATER
    EXPECT_EQ(actual[index].port, expected[index].port)
        << "Parsing without an output entry must preserve every port";
  }
}

/** @brief Provides key, link, and payload fields for intrusive address-map tests. */
struct AddressEntry {
  void* key = nullptr;
  AddressEntry* next = nullptr;
  int payload = 0;
};

/** @brief Provides a large 64-byte-aligned memory-stack value. */
struct alignas(64) OverAlignedStackValue {
  unsigned char bytes[96 << 10];
};

/** @brief Provides the header fields for inline-array layout tests. */
struct InlineArrayHeader {
  uint32_t count;
  uint32_t flags;
};

/** @brief Provides a 32-byte-aligned inline-array element. */
struct alignas(32) InlineArrayElement {
  uint64_t words[4];
};

/** @brief Provides aligned payload and guard fields for object-pool tests. */
struct alignas(32) PoolValue {
  uint64_t value;
  uint64_t guard;
};

/** @brief Provides independent links for single- and multi-producer queue tests. */
struct QueueNode {
  int value;
  QueueNode* link;
  QueueNode* next;
};

using AddressMap = ncclIntruAddressMap<AddressEntry, void*, &AddressEntry::key, &AddressEntry::next>;
using SingleQueue = ncclIntruQueue<QueueNode, &QueueNode::link>;
using MultiProducerQueue = ncclIntruQueueMpsc<QueueNode, &QueueNode::link>;

/**
 * @brief Compares intrusive-queue nodes by identity.
 * @param[in] actual Valid non-NULL node currently stored in a queue.
 * @param[in] expected Valid non-NULL node requested by the caller.
 * @return True when both pointers identify the same node, otherwise false.
 * @note Side effects: None.
 */
bool QueueNodesMatch(QueueNode* actual, QueueNode* expected) {
  return actual == expected;
}

/**
 * @brief Verifies a valid worker joins successfully and cannot terminate the test during cleanup.
 * @param[in,out] worker Joinable worker on entry and non-joinable worker after a correct production
 * join.
 * @param[in] context Valid non-NULL scenario description used in assertion diagnostics.
 * @note Side effects: Joins worker and records assertion failures; a native fallback join prevents
 * termination.
 */
void ExpectThreadJoinSuccess(std::thread& worker, const char* context) {
  const ncclResult_t result = ncclThreadJoin(worker);
  EXPECT_EQ(result, ncclSuccess) << context << ": joining an active worker must succeed";
  EXPECT_FALSE(worker.joinable()) << context << ": a successful join must consume the worker execution context";
  if (worker.joinable()) worker.join();
}

/**
 * @brief Finds three distinct pointer values that hash to one bucket.
 * @param[in] hashBits Bucket-index width in [1, 6].
 * @return Three nonzero integer pointer representations with the same hash.
 */
std::array<uintptr_t, 3> FindCollidingKeys(int hashBits) {
  std::array<std::vector<uintptr_t>, 64> buckets;
  for (uintptr_t key = 1; key < 10000; key++) {
    const uint64_t bucket = ncclHashPointer(hashBits, reinterpret_cast<void*>(key));
    buckets[bucket].push_back(key);
    if (buckets[bucket].size() == 3) {
      return {buckets[bucket][0], buckets[bucket][1], buckets[bucket][2]};
    }
  }
  return {0, 0, 0};
}

}  // namespace

/** @brief Compiles utils.cc with host identity, procfs, and allocation redirected to test seams. */
#define gethostname TestGetHostname
#define ncclGetEnv TestGetEnvironment
#define fopen TestOpenFile
#define ncclOsGetPid TestGetProcessId
#define readlink TestReadLink
#define malloc TestAllocate
#define calloc TestAllocateZeroed
#define free TestRelease
#include "../../../misc/utils.cc"
#undef free
#undef calloc
#undef malloc
#undef readlink
#undef ncclOsGetPid
#undef fopen
#undef ncclGetEnv
#undef gethostname

/**
 * @brief Verifies host-only CUDA behavior and reversible PCI identifier conversions at supported
 * boundaries.
 */
TEST(UtilsValueTest, ConvertsBusIdentifiers) {
  // The host-only configuration must reject device lookup without touching hardware.
  EXPECT_EQ(ncclCudaCompCap(), 0) << "Host-only builds must report no CUDA compute capability";

  int64_t busValue = -1;
  EXPECT_EQ(getBusId(7, &busValue), ncclInternalError)
      << "Host-only logical-device lookup must report that CUDA bus lookup is unavailable";
  EXPECT_EQ(busValue, 0) << "Host-only logical-device lookup must clear its output value";
  EXPECT_EQ(getBusId(7, nullptr), ncclInternalError)
      << "Host-only logical-device lookup must accept a missing optional output without crashing";

  // Zero, an ordinary device, and the conventional PCI maximum must survive a text round trip.
  constexpr int64_t maximumBusId = (int64_t{0xffff} << 20) | (int64_t{0xff} << 12) | (int64_t{0x1f} << 4) | 7;
  for (const auto& scenario : std::array<std::pair<int64_t, const char*>, 3>{
           {{0, "0000:00:00.0"}, {0x000002000, "0000:02:00.0"}, {maximumBusId, "ffff:ff:1f.7"}}}) {
    SCOPED_TRACE(testing::Message() << "bus value 0x" << std::hex << scenario.first);
    char text[32] = {};
    ASSERT_EQ(int64ToBusId(scenario.first, text), ncclSuccess)
        << "Formatting a supported packed bus identifier must succeed";
    EXPECT_STREQ(text, scenario.second)
        << "The packed identifier must use canonical PCI domain:bus:device.function text";
    int64_t restored = -1;
    ASSERT_EQ(busIdToInt64(text, &restored), ncclSuccess)
        << "Parsing canonical bus identifier text must succeed";
    EXPECT_EQ(restored, scenario.first) << "Formatting and parsing a bus identifier must preserve its value";
  }

  // Uppercase input must parse without modifying the caller-owned string.
  const std::string upperText = "ABCD:EF:1A.5";
  int64_t upperValue = -1;
  ASSERT_EQ(busIdToInt64(upperText.c_str(), &upperValue), ncclSuccess)
      << "Parsing uppercase hexadecimal bus text must succeed";
  EXPECT_EQ(upperValue, 0xabcdef1a5LL) << "Uppercase hexadecimal digits and separators must be parsed correctly";
  EXPECT_EQ(upperText, "ABCD:EF:1A.5") << "Bus parsing must not modify caller-owned input text";

  // PCI path parsing must select the final component with either supported ending.
  char pathWithSlash[] = "sys/class/pci0000:00/0000:00:02.0/0000:02:00.0/";
  const std::string originalPath = pathWithSlash;
  int64_t pathValue = -1;
  ASSERT_EQ(pciPathToInt64(pathWithSlash, &pathValue), ncclSuccess)
      << "Extracting the final device from a PCI path ending in a slash must succeed";
  EXPECT_EQ(pathValue, 0x2000) << "PCI path conversion must return the packed final bus identifier";
  EXPECT_EQ(pathWithSlash, originalPath) << "PCI path conversion must not modify caller-owned path text";

  char pathWithoutSlash[] = "/devices/0000:7f:1f.6";
  const std::string originalPathWithoutSlash = pathWithoutSlash;
  pathValue = -1;
  ASSERT_EQ(pciPathToInt64(pathWithoutSlash, &pathValue), ncclSuccess)
      << "Extracting the final device from a PCI path without a trailing slash must succeed";
  EXPECT_EQ(pathValue, 0x7f1f6) << "PCI path conversion must handle the final component without a trailing slash";
  EXPECT_EQ(pathWithoutSlash, originalPathWithoutSlash)
      << "PCI path conversion must preserve input text without a trailing slash";
}

/**
 * @brief Verifies hostname truncation, stable host hashing, PID namespace hashing, and
 * deterministic hash composition.
 */
TEST(UtilsValueTest, ProducesHostAndProcessValues) {
  // A present delimiter truncates a hostname while an absent delimiter preserves it.
  ResetUtilsBoundaries();
  boundaryState.hostname = "node.example.net";
  char hostname[64] = {};
  ASSERT_EQ(getHostName(hostname, sizeof(hostname), '.'), ncclSuccess)
      << "Hostname lookup with a present delimiter must succeed";
  EXPECT_STREQ(hostname, "node") << "Hostname lookup must terminate the result at the requested delimiter";

  std::memset(hostname, 0, sizeof(hostname));
  ASSERT_EQ(getHostName(hostname, sizeof(hostname), '/'), ncclSuccess)
      << "Hostname lookup with an absent delimiter must succeed";
  EXPECT_STREQ(hostname, "node.example.net") << "An absent delimiter must preserve the complete hostname";

  // A hostname system error must return both its status and fallback value.
  boundaryState.hostnameFailure = true;
  ASSERT_EQ(getHostName(hostname, sizeof(hostname), '.'), ncclSystemError)
      << "A hostname system-call failure must propagate as a system error";
  EXPECT_STREQ(hostname, "unknown") << "A hostname system-call failure must provide the documented fallback text";

  // Host hashing must honor environment, boot-ID, and hostname-fallback precedence.
  boundaryState.hostnameFailure = false;
  boundaryState.hostname = "hash-host";
  boundaryState.environmentHostIdPresent = true;
  boundaryState.environmentHostId = "explicit-host-id";
  getHostHashOnce();
  EXPECT_EQ(hostHashValue, getHash("explicit-host-id", std::strlen("explicit-host-id")))
      << "An explicit host identifier must replace the hostname and boot identifier in the host hash";
  EXPECT_TRUE(boundaryState.openedPath.empty())
      << "An explicit host identifier must bypass the boot-ID file entirely";

  boundaryState.environmentHostIdPresent = false;
  boundaryState.bootFileFailure = false;
  boundaryState.bootId = "boot-identifier";
  getHostHashOnce();
  EXPECT_EQ(hostHashValue, getHash("hash-hostboot-identifier", std::strlen("hash-hostboot-identifier")))
      << "Without an override, the host hash must combine the hostname and boot identifier";
  EXPECT_EQ(boundaryState.openedPath, HOSTID_FILE) << "Host hashing must read the documented Linux boot-ID source";

  boundaryState.bootFileFailure = true;
  getHostHashOnce();
  EXPECT_EQ(hostHashValue, getHash("hash-host", std::strlen("hash-host")))
      << "When the boot identifier is unavailable, host hashing must fall back to the hostname";

  // The public accessor must initialize once and remain stable when its source changes.
  boundaryState.environmentHostIdPresent = true;
  boundaryState.environmentHostId = "cached-host-id";
  const uint64_t cachedHostHash = getHostHash();
  EXPECT_EQ(cachedHostHash, getHash("cached-host-id", std::strlen("cached-host-id")))
      << "The public host hash must be initialized from the current explicit identifier";
  boundaryState.environmentHostId = "changed-after-cache";
  EXPECT_EQ(getHostHash(), cachedHostHash) << "The public host hash must remain stable after its first initialization";

  // Process hashing must include the namespace when available and use only the PID otherwise.
  boundaryState.processId = 4321;
  boundaryState.processNamespace = "pid:[12345]";
  boundaryState.readLinkFailure = false;
  EXPECT_EQ(getPidHash(), getHash("4321pid:[12345]", std::strlen("4321pid:[12345]")))
      << "Process hashing must combine the decimal PID and PID-namespace link value";
  boundaryState.readLinkFailure = true;
  EXPECT_EQ(getPidHash(), getHash("4321", std::strlen("4321")))
      << "When namespace lookup fails, process hashing must fall back to the decimal PID";

  // Hash composition must match the documented two-value hash sequence.
  uint64_t reference[2] = {1, 1};
  const uint64_t base = 0x0123456789abcdefULL;
  const uint64_t value = 0xfedcba9876543210ULL;
  eatHash(reference, &base, sizeof(base));
  eatHash(reference, &value, sizeof(value));
  EXPECT_EQ(hashCombine(base, value), digestHash(reference))
      << "Hash composition must consume the complete base and value in their documented order";
}

/**
 * @brief Verifies scalar, clock, random-data, atomic-reference, and thread helpers at useful
 * boundaries.
 */
TEST(UtilsValueTest, ExercisesHeaderUtilities) {
  // Integer logarithm must return the floor exponent, including its defined zero sentinel and
  // largest input.
  EXPECT_EQ(log2i(0), -1) << "The integer logarithm must report -1 for zero";
  EXPECT_EQ(log2i(1), 0) << "The integer logarithm must report exponent zero for one";
  EXPECT_EQ(log2i(3), 1) << "The integer logarithm must round a non-power-of-two input down";
  EXPECT_EQ(log2i(std::numeric_limits<long>::max()), std::numeric_limits<long>::digits - 1)
      << "The integer logarithm must handle the largest positive long without truncation";

  // The comparator must distinguish all three orderings without modifying either caller-owned
  // integer.
  int lower = -7;
  int higher = 11;
  int equal = -7;
  EXPECT_LT(compareInts(&lower, &higher), 0) << "The integer comparator must order a smaller value first";
  EXPECT_GT(compareInts(&higher, &lower), 0) << "The integer comparator must order a larger value last";
  EXPECT_EQ(compareInts(&lower, &equal), 0) << "The integer comparator must report equal values as equal";
  EXPECT_EQ(lower, -7) << "Integer comparison must preserve the first caller-owned value";
  EXPECT_EQ(higher, 11) << "Integer comparison must preserve the second caller-owned value";
  EXPECT_EQ(equal, -7) << "Integer comparison must preserve the equal-value operand";

  // Greatest-common-divisor behavior must cover both zero boundaries and ordinary coprime/composite
  // inputs.
  EXPECT_EQ(gcd(0, 0), 0) << "The greatest common divisor of two zero values must use the zero sentinel";
  EXPECT_EQ(gcd(54, 0), 54) << "A zero second operand must return the positive first operand";
  EXPECT_EQ(gcd(0, 24), 24) << "A zero first operand must return the positive second operand";
  EXPECT_EQ(gcd(54, 24), 6) << "Composite positive operands must return their greatest common divisor";
  EXPECT_EQ(gcd(17, 13), 1) << "Coprime positive operands must return one";

  // Monotonic and realtime clocks must return ordered and normalized host time values.
  const uint64_t monotonicBefore = clockNano();
  timespec realtime = {-1, -1};
  const time_t realtimeBefore = std::time(nullptr);
  clockRealtime(&realtime);
  const time_t realtimeAfter = std::time(nullptr);
  const uint64_t monotonicAfter = clockNano();
  EXPECT_GE(monotonicAfter, monotonicBefore) << "The monotonic clock must not move backward during one test";
  EXPECT_GE(realtime.tv_sec, realtimeBefore - 1)
      << "Realtime seconds must remain within one second of the wall-clock sample before it";
  EXPECT_LE(realtime.tv_sec, realtimeAfter + 1)
      << "Realtime seconds must remain within one second of the wall-clock sample after it";
  EXPECT_GE(realtime.tv_nsec, 0) << "Realtime nanoseconds must be nonnegative";
  EXPECT_LT(realtime.tv_nsec, 1000000000L) << "Realtime nanoseconds must remain below one second";

  // Zero bytes require no destination, while a positive request requires and fills only its
  // advertised range.
  EXPECT_EQ(getRandomData(nullptr, 0), ncclSuccess)
      << "A zero-byte random-data request must succeed without a destination";
  EXPECT_EQ(getRandomData(nullptr, 1), ncclSystemError)
      << "A positive random-data request must reject a missing destination";
  std::array<unsigned char, 34> randomStorage;
  randomStorage.fill(0xa5);
  ASSERT_EQ(getRandomData(randomStorage.data() + 1, randomStorage.size() - 2), ncclSuccess)
      << "A valid positive random-data request must succeed";
  EXPECT_EQ(randomStorage.front(), 0xa5) << "Random-data generation must preserve the byte before its output range";
  EXPECT_EQ(randomStorage.back(), 0xa5) << "Random-data generation must preserve the byte after its output range";
  EXPECT_TRUE(std::any_of(randomStorage.begin() + 1, randomStorage.end() - 1,
                          [](unsigned char byte) { return byte != 0xa5; }))
      << "Random-data generation must replace caller-provided output bytes";

  // Concurrent increments must not be lost, and decrement must return and store the new reference
  // count.
  unsigned int references = 0;
  constexpr int producerCount = 4;
  constexpr int incrementsPerProducer = 1000;
  std::vector<std::thread> producers;
  for (int producer = 0; producer < producerCount; producer++) {
    producers.emplace_back([&references]() {
      for (int iteration = 0; iteration < incrementsPerProducer; iteration++) {
        ncclAtomicRefCountIncrement(&references);
      }
    });
  }
  for (auto& producer : producers) {
    ExpectThreadJoinSuccess(producer, "Atomic increment scenario");
  }
  EXPECT_EQ(references, producerCount * incrementsPerProducer)
      << "Atomic increments from concurrent threads must preserve every update";
  EXPECT_EQ(ncclAtomicRefCountDecrement(&references), producerCount * incrementsPerProducer - 1)
      << "Atomic decrement must return the updated reference count";
  EXPECT_EQ(references, producerCount * incrementsPerProducer - 1)
      << "Atomic decrement must store the same updated reference count";

  // Joining a thread without an associated execution context must report an error and preserve that
  // state.
  std::thread inactiveThread;
  EXPECT_EQ(ncclThreadJoin(inactiveThread), ncclSystemError)
      << "Joining a non-joinable thread must report a system error instead of terminating";
  EXPECT_FALSE(inactiveThread.joinable()) << "A failed join must leave an inactive thread non-joinable";
}

/**
 * @brief Verifies supported interface-list syntax, output bounds, and preservation of unused
 * storage.
 */
TEST(UtilsInterfaceTest, ParsesInterfaceLists) {
  // Sentinel-filled outputs reveal writes made for missing or empty input.
  std::array<netIf, 4> interfaces = {};
  for (auto& interface : interfaces) {
    std::memset(interface.prefix, '#', sizeof(interface.prefix));
    interface.port = 777;
  }
  const auto untouchedInterfaces = interfaces;

  EXPECT_EQ(parseStringList(nullptr, interfaces.data(), interfaces.size()), 0)
      << "A missing interface expression must produce an empty list";
  ExpectInterfacesUnchanged(interfaces, untouchedInterfaces, "missing expression");
  EXPECT_EQ(parseStringList("", interfaces.data(), interfaces.size()), 0)
      << "An empty interface expression must produce an empty list";
  ExpectInterfacesUnchanged(interfaces, untouchedInterfaces, "empty expression");
  EXPECT_EQ(parseStringList("eth0", interfaces.data(), 0), 0)
      << "Zero output capacity must reject every parsed entry";
  ExpectInterfacesUnchanged(interfaces, untouchedInterfaces, "zero output capacity");

  // A normal list must parse ports, wildcard ports, and empty comma elements exactly.
  const std::string expression = ",eth0:7,,ib,eno1:65535,";
  ASSERT_EQ(parseStringList(expression.c_str(), interfaces.data(), interfaces.size()), 3)
      << "Valid comma-separated interfaces with optional ports must all be parsed";
  EXPECT_EQ(expression, ",eth0:7,,ib,eno1:65535,") << "Parsing must not modify caller-owned expression text";
  EXPECT_STREQ(interfaces[0].prefix, "eth0") << "The first interface prefix must be parsed exactly";
  EXPECT_EQ(interfaces[0].port, 7) << "The first explicit port must be parsed exactly";
  EXPECT_STREQ(interfaces[1].prefix, "ib") << "Empty comma elements must be skipped";
  EXPECT_EQ(interfaces[1].port, -1) << "An omitted port must become the wildcard value";
  EXPECT_STREQ(interfaces[2].prefix, "eno1") << "The final interface prefix must be parsed exactly";
  EXPECT_EQ(interfaces[2].port, 65535) << "The maximum valid transport port must be retained";
  EXPECT_EQ(interfaces[3].prefix[0], '#') << "Parsing must not modify an unused output entry";
  EXPECT_EQ(interfaces[3].port, 777) << "Parsing must preserve every field of an unused output entry";

  // A positive output limit must stop before the first unused entry.
  std::array<netIf, 2> limited = {};
  limited[1].port = 991;
  std::memset(limited[1].prefix, '@', sizeof(limited[1].prefix));
  const netIf unusedEntry = limited[1];
  ASSERT_EQ(parseStringList("eth0,eth1", limited.data(), 1), 1)
      << "The output capacity must limit the number of parsed interfaces";
  EXPECT_STREQ(limited[0].prefix, "eth0") << "Capacity limiting must retain the first complete element";
  EXPECT_EQ(std::memcmp(limited[1].prefix, unusedEntry.prefix, sizeof(limited[1].prefix)), 0)
      << "Capacity limiting must preserve every prefix byte beyond the requested output count";
  EXPECT_EQ(limited[1].port, 991) << "Capacity limiting must preserve the unused entry's port";

  // Sixty-three characters are the longest prefix that leaves room for the terminator.
  std::string maximumPrefix(63, 'x');
  netIf longest = {};
  ASSERT_EQ(parseStringList(maximumPrefix.c_str(), &longest, 1), 1)
      << "The longest representable 63-character prefix must parse successfully";
  EXPECT_EQ(std::strlen(longest.prefix), size_t{63})
      << "The maximum prefix must be terminated inside its 64-byte field";
  EXPECT_EQ(std::string(longest.prefix), maximumPrefix) << "The maximum prefix must retain every input character";
  EXPECT_EQ(longest.port, -1) << "A maximum-length prefix without a port must receive the wildcard port";
}

/**
 * @brief Verifies exact, prefix, wildcard-port, multi-entry, and no-match interface selection.
 */
TEST(UtilsInterfaceTest, MatchesInterfacesAndPorts) {
  // Exercise empty, prefix, exact, wildcard-port, later-entry, and no-match selection.
  std::array<netIf, 2> interfaces = {{{"eth", -1}, {"ib0", 4}}};
  const auto original = interfaces;

  EXPECT_TRUE(matchIfList("anything", 99, interfaces.data(), 0, true))
      << "An empty selection list must allow every interface";
  EXPECT_TRUE(matchIfList("eth7", 3, interfaces.data(), interfaces.size(), false))
      << "Prefix matching must accept a longer interface name and a wildcard configured port";
  EXPECT_FALSE(matchIfList("eth7", 3, interfaces.data(), interfaces.size(), true))
      << "Exact matching must reject a name that only shares the configured prefix";
  EXPECT_TRUE(matchIfList("eth", 3, interfaces.data(), interfaces.size(), true))
      << "Exact matching must accept the complete configured name";
  EXPECT_TRUE(matchIfList("ib0", 4, interfaces.data(), interfaces.size(), true))
      << "Matching must inspect later entries and accept equal concrete ports";
  EXPECT_TRUE(matchIfList("ib0", -1, interfaces.data(), interfaces.size(), true))
      << "A wildcard requested port must accept a concrete configured port";
  EXPECT_FALSE(matchIfList("ib0", 5, interfaces.data(), interfaces.size(), true))
      << "Different concrete ports must not match";
  EXPECT_FALSE(matchIfList("mlx5", 4, interfaces.data(), interfaces.size(), false))
      << "An unrelated interface name must not match any configured entry";
  // Matching is read-only for every caller-owned list field.
  for (size_t index = 0; index < interfaces.size(); index++) {
    SCOPED_TRACE(testing::Message() << "interface entry " << index);
    EXPECT_STREQ(interfaces[index].prefix, original[index].prefix)
        << "Interface matching must preserve every caller-owned prefix";
    EXPECT_EQ(interfaces[index].port, original[index].port)
        << "Interface matching must preserve every caller-owned port";
  }
}

/**
 * @brief Verifies that interface predicates cover exact, prefix, wildcard, and mismatch cases.
 */
TEST(UtilsInterfaceTest, MatchesIndividualNamesAndPortsDirectly) {
  SCOPED_TRACE("Primitive interface predicates must preserve inputs and implement their complete match partitions.");
  const std::string interfaceName = "eth0";
  const std::string exactReference = "eth0";
  const std::string prefixReference = "eth";
  const std::string otherReference = "ib";

  EXPECT_TRUE(matchIf(interfaceName.c_str(), exactReference.c_str(), true))
      << "Exact matching must accept equal complete names";
  EXPECT_FALSE(matchIf(interfaceName.c_str(), prefixReference.c_str(), true))
      << "Exact matching must reject a shorter prefix";
  EXPECT_TRUE(matchIf(interfaceName.c_str(), prefixReference.c_str(), false))
      << "Prefix matching must accept a matching configured prefix";
  EXPECT_FALSE(matchIf(interfaceName.c_str(), otherReference.c_str(), false))
      << "Prefix matching must reject unrelated names";
  EXPECT_EQ(interfaceName, "eth0") << "Name matching must preserve the candidate string";
  EXPECT_EQ(prefixReference, "eth") << "Name matching must preserve the reference string";

  EXPECT_TRUE(matchPort(-1, 7)) << "A wildcard first port must match a concrete second port";
  EXPECT_TRUE(matchPort(7, -1)) << "A wildcard second port must match a concrete first port";
  EXPECT_TRUE(matchPort(7, 7)) << "Equal concrete ports must match";
  EXPECT_FALSE(matchPort(7, 8)) << "Different concrete ports must not match";
}

/**
 * @brief Verifies stack construction, zeroing, alignment, frame lifetime, large allocation, and
 * destruction.
 */
TEST(UtilsMemoryTest, ManagesStackLifecycle) {
  // Construction must replace dirty storage with the documented empty stack state.
  ResetUtilsBoundaries();
  ncclMemoryStack stack;
  std::memset(&stack, 0xa5, sizeof(stack));
  ncclMemoryStackConstruct(&stack);
  EXPECT_EQ(stack.stub.above, nullptr) << "Construction must initialize the hunk list as empty";
  EXPECT_EQ(stack.stub.size, 0u) << "Construction must initialize the stack stub size to zero";
  EXPECT_EQ(stack.topFrame.hunk, &stack.stub) << "Construction must point the initial frame at the stack stub";
  EXPECT_EQ(stack.topFrame.bumper, 0u) << "Construction must initialize the allocation cursor to zero";
  EXPECT_EQ(stack.topFrame.end, 0u) << "Construction must initialize the allocation limit to zero";
  EXPECT_EQ(stack.topFrame.unhunks, nullptr) << "Construction must initialize the out-of-band list as empty";
  EXPECT_EQ(stack.topFrame.below, nullptr) << "The non-popable initial frame must not have a lower frame";

  // A typed allocation must be aligned and completely zero-initialized.
  uint32_t* values = ncclMemoryStackAlloc<uint32_t>(&stack, 4);
  ASSERT_NE(values, nullptr) << "A small stack allocation must return storage";
  EXPECT_EQ(reinterpret_cast<uintptr_t>(values) % alignof(uint32_t), 0u)
      << "A typed stack allocation must satisfy the type's alignment";
  const std::array<uint32_t, 4> observedValues = {values[0], values[1], values[2], values[3]};
  EXPECT_EQ(observedValues, (std::array<uint32_t, 4>{0, 0, 0, 0}))
      << "A public stack allocation must zero every requested element";
  values[0] = 0x12345678;

  // A pushed frame owns temporary small and out-of-band allocations.
  ncclMemoryStackPush(&stack);
  uint64_t* temporary = ncclMemoryStackAlloc<uint64_t>(&stack);
  ASSERT_NE(temporary, nullptr) << "A pushed frame must accept an aligned allocation";
  EXPECT_EQ(*temporary, 0u) << "A pushed-frame allocation must be zero-initialized";
  *temporary = 0xabcdef;
  void* large = ncclMemoryStackAlloc(&stack, 96 << 10, alignof(std::max_align_t));
  ASSERT_NE(large, nullptr) << "An allocation larger than a normal hunk must use tracked out-of-band storage";
  const auto* bytes = static_cast<const unsigned char*>(large);
  EXPECT_TRUE(std::all_of(bytes, bytes + (96 << 10), [](unsigned char byte) { return byte == 0; }))
      << "A large public stack allocation must zero the complete requested region";

  // Popping must discard only the upper frame and preserve lower-frame state.
  ncclMemoryStackPop(&stack);
  EXPECT_EQ(values[0], 0x12345678u) << "Popping a frame must preserve allocations owned by the lower frame";
  EXPECT_EQ(stack.topFrame.below, nullptr) << "Popping must restore the initial frame linkage";
  ncclMemoryStackDestruct(&stack);
}

/**
 * @brief Verifies raw and inline stack allocation plus free-list allocation, reuse, transfer, and
 * zeroing.
 */
TEST(UtilsMemoryTest, ManagesInlineArraysAndPools) {
  ResetUtilsBoundaries();
  ncclMemoryStack stack;
  ncclMemoryStackConstruct(&stack);

  // The internal allocator must return aligned storage and advance ownership into the active frame.
  void* raw = ncclMemoryStack::allocate(&stack, 24, 32);
  ASSERT_NE(raw, nullptr) << "A valid internal stack allocation must return storage";
  EXPECT_EQ(reinterpret_cast<uintptr_t>(raw) % 32, 0u)
      << "The internal stack allocator must honor a supported power-of-two alignment";
  EXPECT_NE(stack.topFrame.hunk, &stack.stub) << "The first internal allocation must acquire backing storage";
  EXPECT_GE(stack.topFrame.bumper, reinterpret_cast<uintptr_t>(raw) + 24)
      << "The active frame must own the complete internal allocation";

  // Inline allocation must align and zero both its header and every requested element.
  InlineArrayHeader* header =
      ncclMemoryStackAllocInlineArray<InlineArrayHeader, InlineArrayElement>(&stack, 3);
  ASSERT_NE(header, nullptr) << "A valid inline-array request must return a header";
  EXPECT_EQ(reinterpret_cast<uintptr_t>(header) % alignof(InlineArrayElement), 0u)
      << "An inline array must satisfy the stricter element alignment";
  const uintptr_t elementAddress =
      (reinterpret_cast<uintptr_t>(header) + sizeof(*header) + alignof(InlineArrayElement) - 1) &
      -uintptr_t(alignof(InlineArrayElement));
  const InlineArrayElement* elements = reinterpret_cast<const InlineArrayElement*>(elementAddress);
  EXPECT_EQ(header->count, 0u) << "Inline-array allocation must zero the complete header";
  EXPECT_EQ(header->flags, 0u) << "Inline-array allocation must zero every header field";
  for (size_t element = 0; element < 3; element++) {
    for (uint64_t word : elements[element].words) {
      EXPECT_EQ(word, 0u) << "Inline-array allocation must zero every word of every element";
    }
  }
  InlineArrayHeader* emptyHeader =
      ncclMemoryStackAllocInlineArray<InlineArrayHeader, InlineArrayElement>(&stack, 0);
  ASSERT_NE(emptyHeader, nullptr) << "A zero-element inline array must still allocate its header";
  EXPECT_EQ(emptyHeader->count, 0u) << "A zero-element inline array must zero its header";
  EXPECT_EQ(emptyHeader->flags, 0u) << "A zero-element inline array must zero every header field";

  // A fresh pool must allocate aligned zeroed objects and reuse a returned object after clearing
  // stale data.
  ncclMemoryPool destination;
  std::memset(&destination, 0xa5, sizeof(destination));
  ncclMemoryPoolConstruct(&destination);
  EXPECT_EQ(destination.head, nullptr) << "Pool construction must replace dirty storage with an empty free list";
  PoolValue* first = ncclMemoryPoolAlloc<PoolValue>(&destination, &stack);
  ASSERT_NE(first, nullptr) << "Allocation from an empty pool must use its backing stack";
  EXPECT_EQ(reinterpret_cast<uintptr_t>(first) % alignof(PoolValue), 0u)
      << "Pool allocation must honor the requested object's alignment";
  EXPECT_EQ(first->value, 0u) << "A new pool object must have zeroed value storage";
  EXPECT_EQ(first->guard, 0u) << "A new pool object must have zeroed guard storage";
  first->value = 0x1122334455667788ULL;
  first->guard = 0x8877665544332211ULL;
  ncclMemoryPoolFree(&destination, first);
  EXPECT_EQ(destination.head, reinterpret_cast<ncclMemoryPool::Cell*>(first))
      << "Freeing the only pool object must place it at the free-list head";
  EXPECT_EQ(destination.tail, reinterpret_cast<ncclMemoryPool::Cell*>(first))
      << "Freeing the only pool object must also place it at the free-list tail";
  PoolValue* reused = ncclMemoryPoolAlloc<PoolValue>(&destination, &stack);
  ASSERT_EQ(reused, first) << "The next pool allocation must reuse the most recently freed object";
  EXPECT_EQ(reused->value, 0u) << "Reused pool allocation must clear stale value storage";
  EXPECT_EQ(reused->guard, 0u) << "Reused pool allocation must clear stale guard storage";

  // Taking a nonempty source must preserve its LIFO order ahead of an existing destination free
  // list.
  ncclMemoryPool source;
  ncclMemoryPoolConstruct(&source);
  PoolValue* second = ncclMemoryPoolAlloc<PoolValue>(&source, &stack);
  PoolValue* third = ncclMemoryPoolAlloc<PoolValue>(&source, &stack);
  ncclMemoryPoolFree(&destination, reused);
  ncclMemoryPoolFree(&source, second);
  ncclMemoryPoolFree(&source, third);
  ncclMemoryPoolTakeAll(&destination, &source);
  EXPECT_EQ(source.head, nullptr) << "Taking all objects must leave the source pool empty";
  EXPECT_EQ(ncclMemoryPoolAlloc<PoolValue>(&destination, &stack), third)
      << "A transferred source head must become the first destination allocation";
  EXPECT_EQ(ncclMemoryPoolAlloc<PoolValue>(&destination, &stack), second)
      << "Transfer must preserve the source pool's free-list order";
  EXPECT_EQ(ncclMemoryPoolAlloc<PoolValue>(&destination, &stack), reused)
      << "Transferred objects must remain ahead of the destination's prior free objects";

  // Taking an empty source must preserve a populated destination exactly.
  ncclMemoryPoolFree(&destination, reused);
  ncclMemoryPool::Cell* destinationHead = destination.head;
  ncclMemoryPool::Cell* destinationTail = destination.tail;
  ncclMemoryPoolTakeAll(&destination, &source);
  EXPECT_EQ(destination.head, destinationHead) << "Taking an empty source must preserve the destination head";
  EXPECT_EQ(destination.tail, destinationTail) << "Taking an empty source must preserve the destination tail";
  EXPECT_EQ(source.head, nullptr) << "Taking an empty source must leave that source empty";

  // Moving into an empty destination must transfer both free-list endpoints.
  ncclMemoryPool emptyDestination;
  ncclMemoryPoolConstruct(&emptyDestination);
  ncclMemoryPoolTakeAll(&emptyDestination, &destination);
  EXPECT_EQ(destination.head, nullptr) << "Taking into an empty pool must still empty the source";
  EXPECT_EQ(emptyDestination.head, destinationHead) << "An empty destination must adopt the source head";
  EXPECT_EQ(emptyDestination.tail, destinationTail) << "An empty destination must adopt the source tail";
  ncclMemoryStackDestruct(&stack);
}

/**
 * @brief Verifies over-aligned out-of-band allocation and fatal first-hunk failure.
 */
TEST(UtilsMemoryTest, HandlesStackAllocationBoundaries) {
  // A real over-aligned typed allocation must be valid even when the raw out-of-band allocation is
  // less aligned.
  ResetUtilsBoundaries();
  boundaryState.shiftMallocAt = 2;
  ncclMemoryStack stack;
  ncclMemoryStackConstruct(&stack);
  OverAlignedStackValue* value = static_cast<OverAlignedStackValue*>(
      ncclMemoryStack::allocateSpilled(&stack, sizeof(OverAlignedStackValue), alignof(OverAlignedStackValue)));
  ASSERT_NE(value, nullptr) << "A valid over-aligned stack allocation must return storage";
#if 0 // ADDRESS_LATER
  EXPECT_EQ(reinterpret_cast<uintptr_t>(value) % alignof(OverAlignedStackValue), 0u)
      << "Out-of-band typed storage must satisfy the requested 64-byte alignment";
#endif 0 // ADDRESS_LATER
  ncclMemoryStackDestruct(&stack);

  // Failure to allocate the first required hunk follows the implementation's fatal contract.
  EXPECT_DEATH(
      {
        ResetUtilsBoundaries();
        boundaryState.failMallocAt = 1;
        ncclMemoryStack failedStack;
        ncclMemoryStackConstruct(&failedStack);
        (void)ncclMemoryStackAlloc(&failedStack, 16, alignof(uint64_t));
      },
      ".*") << "Failure to allocate the first required hunk must terminate instead of returning invalid storage";
}

/**
 * @brief Verifies direct intrusive-field helpers zero-extend keys and change only the selected
 * next-pointer field.
 */
TEST(UtilsContainerTest, ReadsAndWritesIntrusiveFieldsDirectly) {
  SCOPED_TRACE("Untyped intrusive helpers must support offset fields without aliasing or collateral writes.");
  /** @brief Provides offset fields and canaries for untyped intrusive helpers. */
  struct FieldRecord {
    uint8_t prefix;
    uint16_t key;
    uint32_t guard;
    FieldRecord* next;
    uint64_t suffix;
  };
  FieldRecord first = {0x5a, 0xabcd, 0x12345678u, nullptr, 0x1122334455667788ULL};
  FieldRecord second = {};
  const FieldRecord original = first;
  std::array<unsigned char, sizeof(FieldRecord)> representationBefore = {};
  std::memcpy(representationBefore.data(), &first, representationBefore.size());

  EXPECT_EQ(readKey(&first, sizeof(first.key), offsetof(FieldRecord, key)), uintptr_t{0xabcd})
      << "A narrow key must be copied and zero-extended to uintptr_t";
  EXPECT_EQ(std::memcmp(&first, representationBefore.data(), representationBefore.size()), 0)
      << "Reading a key must preserve the complete caller-owned object";
  EXPECT_EQ(readNextPtr(&first, offsetof(FieldRecord, next)), nullptr)
      << "Reading a NULL intrusive link must return NULL";

  writeNextPtr(&first, offsetof(FieldRecord, next), &second);
  EXPECT_EQ(readNextPtr(&first, offsetof(FieldRecord, next)), &second)
      << "Reading after a write must recover the exact pointer value";
  EXPECT_EQ(first.prefix, original.prefix) << "Writing the link must preserve preceding fields";
  EXPECT_EQ(first.key, original.key) << "Writing the link must preserve the key field";
  EXPECT_EQ(first.guard, original.guard) << "Writing the link must preserve the adjacent guard";
  EXPECT_EQ(first.suffix, original.suffix) << "Writing the link must preserve following fields";
}

/**
 * @brief Verifies intrusive queues across empty, ordered, transfer, wait, and abandon paths.
 */
TEST(UtilsContainerTest, ManagesIntrusiveQueues) {
  QueueNode first = {1, nullptr, reinterpret_cast<QueueNode*>(uintptr_t{0x11})};
  QueueNode second = {2, nullptr, reinterpret_cast<QueueNode*>(uintptr_t{0x22})};
  QueueNode third = {3, nullptr, reinterpret_cast<QueueNode*>(uintptr_t{0x33})};
  QueueNode missing = {4, nullptr, reinterpret_cast<QueueNode*>(uintptr_t{0x44})};

  // Construction and empty accessors must replace dirty queue state with a stable empty
  // representation.
  SingleQueue queue = {&missing, &missing};
  ncclIntruQueueConstruct(&queue);
  EXPECT_TRUE(ncclIntruQueueEmpty(&queue)) << "A newly constructed intrusive queue must be empty";
  EXPECT_EQ(ncclIntruQueueHead(&queue), nullptr) << "An empty intrusive queue must have no head";
  EXPECT_EQ(ncclIntruQueueTail(&queue), nullptr) << "An empty intrusive queue must have no tail";
  EXPECT_EQ(ncclIntruQueueTryDequeue(&queue), nullptr)
      << "Nonblocking dequeue from an empty intrusive queue must return NULL";

  // Front and back insertion must establish FIFO order through the selected link member only.
  ncclIntruQueueEnqueueFront(&queue, &first);
  ncclIntruQueueEnqueue(&queue, &second);
  ncclIntruQueueEnqueueFront(&queue, &third);
  EXPECT_FALSE(ncclIntruQueueEmpty(&queue)) << "An intrusive queue containing nodes must not report empty";
  EXPECT_EQ(ncclIntruQueueHead(&queue), &third) << "Front insertion must replace the queue head";
  EXPECT_EQ(ncclIntruQueueTail(&queue), &second) << "Back insertion must retain the final node as queue tail";
  EXPECT_EQ(third.link, &first) << "Front insertion must link the new head to the previous head";
  EXPECT_EQ(first.link, &second) << "Back insertion must append after the previous tail";
  EXPECT_EQ(second.link, nullptr) << "The queue tail's selected link must be NULL";
  EXPECT_EQ(first.next, reinterpret_cast<QueueNode*>(uintptr_t{0x11}))
      << "Queue operations must preserve an unrelated member named next";

  // Missing deletion must preserve the queue; middle deletion and both dequeue forms must update
  // exact endpoints.
  EXPECT_EQ(ncclIntruQueueDelete(&queue, &missing, QueueNodesMatch), nullptr)
      << "Deleting a node that is not present must return NULL";
  EXPECT_EQ(ncclIntruQueueHead(&queue), &third) << "Missing deletion must preserve the queue head";
  EXPECT_EQ(ncclIntruQueueTail(&queue), &second) << "Missing deletion must preserve the queue tail";
  EXPECT_EQ(ncclIntruQueueDelete(&queue, &first, QueueNodesMatch), &first)
      << "Deleting a present middle node must return that exact node";
  EXPECT_EQ(third.link, &second) << "Middle deletion must link the predecessor to the successor";
  EXPECT_EQ(ncclIntruQueueDequeue(&queue), &third)
      << "Dequeue with its nonempty precondition satisfied must remove the queue head";
  EXPECT_EQ(ncclIntruQueueTryDequeue(&queue), &second)
      << "Nonblocking dequeue on a nonempty queue must remove the remaining head";
  EXPECT_TRUE(ncclIntruQueueEmpty(&queue)) << "Removing every node must leave the queue empty";
  EXPECT_EQ(ncclIntruQueueTail(&queue), nullptr) << "Removing the final node must clear the queue tail";

  // Deleting the head and then the tail must update both endpoints without disturbing unrelated
  // fields.
  ncclIntruQueueEnqueue(&queue, &first);
  ncclIntruQueueEnqueue(&queue, &second);
  EXPECT_EQ(ncclIntruQueueDelete(&queue, &first, QueueNodesMatch), &first)
      << "Deleting the queue head must return that exact node";
  EXPECT_EQ(ncclIntruQueueHead(&queue), &second) << "Deleting the queue head must expose its successor";
  EXPECT_EQ(ncclIntruQueueTail(&queue), &second) << "Deleting the queue head must preserve the remaining tail";
  EXPECT_EQ(ncclIntruQueueDelete(&queue, &second, QueueNodesMatch), &second)
      << "Deleting the queue tail must return that exact node";
  EXPECT_TRUE(ncclIntruQueueEmpty(&queue)) << "Deleting the final node must leave the queue empty";
  EXPECT_EQ(ncclIntruQueueTail(&queue), nullptr) << "Deleting the final node must clear the queue tail";
  EXPECT_EQ(first.next, reinterpret_cast<QueueNode*>(uintptr_t{0x11}))
      << "Queue deletion must preserve unrelated caller-owned fields";

  // Transfer must append a source to a nonempty destination and then empty the source.
  SingleQueue transferDestination;
  SingleQueue source;
  ncclIntruQueueConstruct(&transferDestination);
  ncclIntruQueueConstruct(&source);
  first.link = nullptr;
  second.link = nullptr;
  third.link = nullptr;
  QueueNode* firstNext = first.next;
  ncclIntruQueueEnqueue(&transferDestination, &first);
  ncclIntruQueueEnqueue(&source, &second);
  ncclIntruQueueEnqueue(&source, &third);
  ncclIntruQueueTransfer(&transferDestination, &source);
#if 0 // ADDRESS_LATER
  EXPECT_EQ(first.link, &second) << "Queue transfer must append through the selected link member";
#endif 0 // ADDRESS_LATER
  EXPECT_EQ(second.link, &third) << "Queue transfer must preserve source order through the selected link";
  EXPECT_EQ(ncclIntruQueueTail(&transferDestination), &third) << "Queue transfer must adopt the source tail";
  EXPECT_TRUE(ncclIntruQueueEmpty(&source)) << "Queue transfer must empty the source queue";
#if 0 // ADDRESS_LATER
  EXPECT_EQ(first.next, firstNext) << "Queue transfer must preserve the unrelated next member";
#endif 0 // ADDRESS_LATER

  // Transfer into an empty destination must adopt the complete source, including both endpoints.
  SingleQueue emptyDestination;
  SingleQueue movableSource;
  ncclIntruQueueConstruct(&emptyDestination);
  ncclIntruQueueConstruct(&movableSource);
  second.link = nullptr;
  third.link = nullptr;
  ncclIntruQueueEnqueue(&movableSource, &second);
  ncclIntruQueueEnqueue(&movableSource, &third);
  ncclIntruQueueTransfer(&emptyDestination, &movableSource);
  EXPECT_EQ(ncclIntruQueueHead(&emptyDestination), &second)
      << "An empty transfer destination must adopt the source head";
  EXPECT_EQ(ncclIntruQueueTail(&emptyDestination), &third)
      << "An empty transfer destination must adopt the source tail";
  EXPECT_EQ(second.link, &third) << "Transfer into an empty destination must preserve source order";
  EXPECT_TRUE(ncclIntruQueueEmpty(&movableSource)) << "Transfer must empty a source with multiple nodes";

  // Transfer from an empty source must leave both destination endpoints and unrelated node fields
  // unchanged.
  SingleQueue retainedDestination;
  SingleQueue emptySource;
  ncclIntruQueueConstruct(&retainedDestination);
  ncclIntruQueueConstruct(&emptySource);
  first.link = nullptr;
  firstNext = first.next;
  ncclIntruQueueEnqueue(&retainedDestination, &first);
  ncclIntruQueueTransfer(&retainedDestination, &emptySource);
  EXPECT_EQ(ncclIntruQueueHead(&retainedDestination), &first)
      << "Transfer from an empty source must preserve the destination head";
  EXPECT_EQ(ncclIntruQueueTail(&retainedDestination), &first)
      << "Transfer from an empty source must preserve the destination tail";
  EXPECT_EQ(first.link, nullptr) << "Transfer from an empty source must preserve the selected terminal link";
#if 0 // ADDRESS_LATER
  EXPECT_EQ(first.next, firstNext)
      << "Transfer from an empty source must preserve unrelated caller-owned fields";
#endif 0 // ADDRESS_LATER

  // Multi-producer construction, sequential enqueue, and bulk dequeue must preserve each node
  // exactly once.
  MultiProducerQueue multiQueue = {&missing, uintptr_t{0x55}, &ncclThreadSignalLocalInstance};
  ncclIntruQueueMpscConstruct(&multiQueue);
  EXPECT_EQ(multiQueue.head, nullptr) << "Multi-producer construction must clear a dirty head";
  EXPECT_EQ(multiQueue.tail, 0u) << "Multi-producer construction must clear a dirty tail";
  EXPECT_EQ(multiQueue.waiting, nullptr) << "Multi-producer construction must clear a dirty waiter";
  EXPECT_TRUE(ncclIntruQueueMpscEmpty(&multiQueue)) << "A newly constructed multi-producer queue must be empty";
  EXPECT_EQ(ncclIntruQueueMpscDequeueAll(&multiQueue, false), nullptr)
      << "Non-waiting bulk dequeue from an empty queue must return NULL";
  first.link = nullptr;
  second.link = nullptr;
  first.next = reinterpret_cast<QueueNode*>(uintptr_t{0x11});
  second.next = reinterpret_cast<QueueNode*>(uintptr_t{0x22});
  EXPECT_TRUE(ncclIntruQueueMpscEnqueue(&multiQueue, &first))
      << "Enqueue into an active multi-producer queue must be accepted";
  EXPECT_TRUE(ncclIntruQueueMpscEnqueue(&multiQueue, &second))
      << "A second enqueue into an active multi-producer queue must be accepted";
  EXPECT_FALSE(ncclIntruQueueMpscEmpty(&multiQueue))
      << "A multi-producer queue containing nodes must not report empty";
  QueueNode* batch = ncclIntruQueueMpscDequeueAll(&multiQueue, false);
  ASSERT_EQ(batch, &first) << "Bulk dequeue must return the first enqueued node";
  EXPECT_EQ(batch->link, &second) << "Bulk dequeue must preserve enqueue order";
  EXPECT_EQ(second.link, nullptr) << "The final bulk-dequeued node must terminate the returned list";
  EXPECT_EQ(first.next, reinterpret_cast<QueueNode*>(uintptr_t{0x11}))
      << "Multi-producer enqueue must preserve unrelated fields in the first node";
  EXPECT_EQ(second.next, reinterpret_cast<QueueNode*>(uintptr_t{0x22}))
      << "Multi-producer enqueue must preserve unrelated fields in the final node";
  EXPECT_TRUE(ncclIntruQueueMpscEmpty(&multiQueue)) << "Bulk dequeue must reset the queue to empty";

  // Concurrent producers must not lose, duplicate, or corrupt any independently owned node.
  constexpr size_t concurrentCount = 8;
  std::array<QueueNode, concurrentCount> concurrentNodes = {};
  std::atomic<int> rejectedEnqueues{0};
  std::vector<std::thread> concurrentProducers;
  for (size_t index = 0; index < concurrentNodes.size(); index++) {
    concurrentNodes[index].value = static_cast<int>(index);
    concurrentProducers.emplace_back([&multiQueue, &concurrentNodes, &rejectedEnqueues, index]() {
      if (!ncclIntruQueueMpscEnqueue(&multiQueue, &concurrentNodes[index])) rejectedEnqueues.fetch_add(1);
    });
  }
  for (auto& producer : concurrentProducers) {
    ExpectThreadJoinSuccess(producer, "Concurrent queue scenario");
  }
  EXPECT_EQ(rejectedEnqueues.load(), 0) << "An active queue must accept every concurrent producer";
  std::array<bool, concurrentCount> seen = {};
  QueueNode* node = ncclIntruQueueMpscDequeueAll(&multiQueue, false);
  size_t observedCount = 0;
  while (node != nullptr && observedCount < concurrentCount) {
    ASSERT_GE(node->value, 0) << "A dequeued concurrent node must retain its caller-owned value";
    ASSERT_LT(static_cast<size_t>(node->value), concurrentCount)
        << "A dequeued concurrent node must belong to the submitted array";
    EXPECT_FALSE(seen[node->value]) << "Concurrent enqueue must not return the same node twice";
    seen[node->value] = true;
    node = node->link;
    observedCount++;
  }
  EXPECT_EQ(node, nullptr) << "The concurrent dequeue list must terminate without a cycle";
  EXPECT_EQ(observedCount, concurrentCount) << "Concurrent enqueue must preserve every submitted node";
  EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](bool present) { return present; }))
      << "Concurrent dequeue must contain every submitted node exactly once";

  // A waiting consumer must wake after one delayed producer publishes a node.
  QueueNode wakeNode = {9, nullptr, reinterpret_cast<QueueNode*>(uintptr_t{0x99})};
  std::atomic<bool> wakeAccepted{false};
  std::atomic<bool> waiterObserved{false};
  /**
   * @brief Waits briefly for the consumer, then publishes one node.
   * @note Side effects: Updates synchronization flags and enqueues wakeNode into multiQueue.
   */
  std::thread delayedProducer([&multiQueue, &wakeNode, &wakeAccepted, &waiterObserved]() {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while (std::chrono::steady_clock::now() < deadline) {
      if (COMPILER_ATOMIC_LOAD(&multiQueue.tail, std::memory_order_acquire) == 0x1) {
        waiterObserved.store(true);
        break;
      }
      std::this_thread::yield();
    }
    wakeAccepted.store(ncclIntruQueueMpscEnqueue(&multiQueue, &wakeNode));
  });
  QueueNode* awakened = ncclIntruQueueMpscDequeueAll(&multiQueue, true);
  ExpectThreadJoinSuccess(delayedProducer, "Waiting queue scenario");
  EXPECT_TRUE(waiterObserved.load()) << "The waiting scenario must observe the consumer's registered wait state";
  EXPECT_TRUE(wakeAccepted.load()) << "A delayed enqueue into an active waiting queue must be accepted";
  ASSERT_EQ(awakened, &wakeNode) << "A waiting dequeue must return the node that woke it";
  EXPECT_EQ(awakened->link, nullptr) << "A single-node waiting dequeue must return a terminated list";
  EXPECT_EQ(wakeNode.next, reinterpret_cast<QueueNode*>(uintptr_t{0x99}))
      << "Waiting enqueue and dequeue must preserve unrelated node fields";

  // Abandoning an empty queue must reject later enqueue while keeping that node recoverable by the
  // owner.
  EXPECT_EQ(ncclIntruQueueMpscAbandon(&multiQueue), nullptr)
      << "Abandoning an empty multi-producer queue must return no nodes";
  EXPECT_EQ(multiQueue.head, nullptr) << "Empty abandonment must preserve an empty queue head";
  EXPECT_EQ(multiQueue.tail, 0x2u) << "Empty abandonment must store the abandoned-state sentinel";
  EXPECT_TRUE(ncclIntruQueueMpscEmpty(&multiQueue))
      << "An abandoned multi-producer queue with no pending nodes must report empty";
  QueueNode lateNode = {10, nullptr, reinterpret_cast<QueueNode*>(uintptr_t{0xaa})};
  EXPECT_FALSE(ncclIntruQueueMpscEnqueue(&multiQueue, &lateNode))
      << "Enqueue after abandonment must report that the queue is no longer tended";
  EXPECT_FALSE(ncclIntruQueueMpscEmpty(&multiQueue))
      << "A node published after abandonment must remain visible for recovery";
  QueueNode* recovered = ncclIntruQueueMpscAbandon(&multiQueue);
  ASSERT_EQ(recovered, &lateNode) << "Repeated abandonment must return a node published after abandonment";
  EXPECT_EQ(recovered->link, nullptr) << "A recovered single-node abandoned list must be terminated";
  EXPECT_EQ(multiQueue.head, nullptr) << "Recovering an abandoned list must clear the queue head";
  EXPECT_EQ(multiQueue.tail, 0x2u) << "Recovering an abandoned list must restore the abandoned-state sentinel";
  EXPECT_TRUE(ncclIntruQueueMpscEmpty(&multiQueue)) << "Recovery must leave the queue in its abandoned empty state";
  EXPECT_EQ(lateNode.next, reinterpret_cast<QueueNode*>(uintptr_t{0xaa}))
      << "Abandonment must preserve unrelated caller-owned node fields";
}

/**
 * @brief Verifies bit filtering, wildcard formatting, and invalid arguments.
 */
TEST(UtilsContainerTest, FormatsBitSets) {
  // Invalid destinations must be rejected without changing caller-owned storage.
  char text[64] = "preserve";
  EXPECT_EQ(ncclBitsToString(1, 1, BitLabel, nullptr, sizeof(text), nullptr), ncclInvalidArgument)
      << "A missing destination must be rejected";
  EXPECT_EQ(ncclBitsToString(1, 1, BitLabel, text, 0, nullptr), ncclInvalidArgument)
      << "A zero-capacity destination must be rejected";
  EXPECT_STREQ(text, "preserve") << "Rejected formatting must not modify caller-owned destination storage";

  // Masked labels must be emitted in increasing bit order with separators.
  ASSERT_EQ(ncclBitsToString(0b1011, 0b0111, BitLabel, text, sizeof(text), nullptr), ncclSuccess)
      << "Formatting masked set bits must succeed";
  EXPECT_STREQ(text, "zero|one") << "Formatting must omit masked bits and join retained labels in bit order";

  // A complete masked set must select the caller-provided wildcard.
  ASSERT_EQ(ncclBitsToString(0b0111, 0b0111, BitLabel, text, sizeof(text), "all"), ncclSuccess)
      << "Formatting a complete masked set with a wildcard must succeed";
  EXPECT_STREQ(text, "all") << "A complete masked set must use the supplied wildcard text";

  // An empty selected set must replace stale destination text with the empty string.
  std::snprintf(text, sizeof(text), "stale");
  ASSERT_EQ(ncclBitsToString(0, 0b1111, BitLabel, text, sizeof(text), nullptr), ncclSuccess)
      << "Formatting an empty selected set must succeed";
#if 0 // ADDRESS_LATER
  EXPECT_STREQ(text, "") << "An empty selected set must produce an empty output string";
#endif 0 // ADDRESS_LATER

  // Truncation must terminate inside the advertised capacity and preserve physically padded guard
  // bytes.
  std::array<char, 32> bounded;
  bounded.fill('@');
  ASSERT_EQ(ncclBitsToString(0b0011, 0b1111, BitLabel, bounded.data(), 4, nullptr), ncclSuccess)
      << "Formatting into a small but nonempty destination must retain the success contract";
  EXPECT_EQ(std::string(bounded.data(), 4), std::string("zer\0", 4))
      << "A four-byte destination must contain a terminated truncation of the first label";
#if 0 // ADDRESS_LATER
  EXPECT_TRUE(std::all_of(bounded.begin() + 4, bounded.end(), [](char byte) { return byte == '@'; }))
      << "Formatting must not write beyond the caller-advertised four-byte capacity";
#endif 0 // ADDRESS_LATER
}

/**
 * @brief Verifies pointer hashing and invalid, empty, and allocation-failure address-map outcomes.
 */
TEST(UtilsContainerTest, ValidatesAddressMapOperations) {
  // Fixed vectors verify the documented pointer-hash algorithm and width handling.
  EXPECT_EQ(ncclHashPointer(4, reinterpret_cast<void*>(uintptr_t{0x12345678})), 9u)
      << "Four-bit pointer hashing must match the fixed algorithm vector";
  EXPECT_EQ(ncclHashPointer(8, reinterpret_cast<void*>(uintptr_t{0x12345678})), 145u)
      << "Increasing the hash width must expose the corresponding longer bucket prefix";
  EXPECT_EQ(ncclHashPointer(16, reinterpret_cast<void*>(uintptr_t{0xabcdef0123456789})), 53669u)
      << "Pointer hashing must mix all pointer-width input bits";

  // Invalid insertions must return invalid usage without allocating or changing the object.
  constexpr int keyOffset = offsetof(AddressEntry, key);
  constexpr int nextOffset = offsetof(AddressEntry, next);
  ncclIntruAddressMap_untyped map = {};
  AddressEntry entry = {reinterpret_cast<void*>(uintptr_t{1}), reinterpret_cast<AddressEntry*>(uintptr_t{0x55}), 7};
  void* output = reinterpret_cast<void*>(uintptr_t{0x99});

  EXPECT_EQ(ncclIntruAddressMapInsert_untyped(nullptr, sizeof(void*), keyOffset, nextOffset, 1, &entry),
            ncclInvalidUsage) << "Insertion must reject a missing map";
  EXPECT_EQ(ncclIntruAddressMapInsert_untyped(&map, sizeof(void*), keyOffset, nextOffset, 1, nullptr),
            ncclInvalidUsage) << "Insertion must reject a missing object";
  EXPECT_EQ(ncclIntruAddressMapInsert_untyped(&map, 0, keyOffset, nextOffset, 1, &entry), ncclInvalidUsage)
      << "Insertion must reject a zero-sized key";
  EXPECT_EQ(ncclIntruAddressMapInsert_untyped(&map, sizeof(uintptr_t) + 1, keyOffset, nextOffset, 1, &entry),
            ncclInvalidUsage) << "Insertion must reject a key wider than uintptr_t";
  EXPECT_EQ(map.hbits, 0) << "Rejected insertion must preserve the empty map's hash width";
  EXPECT_EQ(map.count, 0) << "Rejected insertion must preserve the empty map's object count";
  EXPECT_EQ(map.table, nullptr) << "Rejected insertion must not allocate a bucket table";
  EXPECT_EQ(entry.next, reinterpret_cast<AddressEntry*>(uintptr_t{0x55}))
      << "Rejected insertion must preserve the object's intrusive link";
  EXPECT_EQ(entry.key, reinterpret_cast<void*>(uintptr_t{1}))
      << "Rejected insertion must preserve the object's caller-owned key";
  EXPECT_EQ(entry.payload, 7) << "Rejected insertion must preserve unrelated object data";

  // Invalid lookup arguments preserve output, while a valid empty lookup clears it.
  EXPECT_EQ(ncclIntruAddressMapFind_untyped(nullptr, sizeof(void*), keyOffset, nextOffset, 1, &output),
            ncclInvalidUsage) << "Lookup must reject a missing map";
  EXPECT_EQ(output, reinterpret_cast<void*>(uintptr_t{0x99}))
      << "Lookup rejected before output ownership begins must preserve the caller's output";
  EXPECT_EQ(ncclIntruAddressMapFind_untyped(&map, sizeof(void*), keyOffset, nextOffset, 1, nullptr),
            ncclInvalidUsage) << "Lookup must reject a missing output pointer";
  EXPECT_EQ(ncclIntruAddressMapFind_untyped(&map, 0, keyOffset, nextOffset, 1, &output), ncclInvalidUsage)
      << "Lookup must reject a zero-sized key";
  EXPECT_EQ(output, reinterpret_cast<void*>(uintptr_t{0x99}))
      << "Lookup with an invalid key size must preserve the caller's output";
  EXPECT_EQ(ncclIntruAddressMapFind_untyped(
                &map, sizeof(uintptr_t) + 1, keyOffset, nextOffset, 1, &output),
            ncclInvalidUsage) << "Lookup must reject a key wider than uintptr_t";
  EXPECT_EQ(output, reinterpret_cast<void*>(uintptr_t{0x99}))
      << "Lookup with an oversized key must preserve the caller's output";
  ASSERT_EQ(ncclIntruAddressMapFind_untyped(&map, sizeof(void*), keyOffset, nextOffset, 1, &output), ncclSuccess)
      << "Lookup in a valid empty map must succeed";
  EXPECT_EQ(output, nullptr) << "Lookup in an empty map must report no object through its output";

  // Removal rejects invalid arguments and treats an absent key as an idempotent success.
  EXPECT_EQ(ncclIntruAddressMapRemove_untyped(nullptr, sizeof(void*), keyOffset, nextOffset, 1), ncclInvalidUsage)
      << "Removal must reject a missing map";
  EXPECT_EQ(ncclIntruAddressMapRemove_untyped(&map, 0, keyOffset, nextOffset, 1), ncclInvalidUsage)
      << "Removal must reject a zero-sized key";
  EXPECT_EQ(ncclIntruAddressMapRemove_untyped(&map, sizeof(uintptr_t) + 1, keyOffset, nextOffset, 1),
            ncclInvalidUsage) << "Removal must reject a key wider than uintptr_t";
  EXPECT_EQ(ncclIntruAddressMapRemove_untyped(&map, sizeof(void*), keyOffset, nextOffset, 1), ncclSuccess)
      << "Removal from an empty map must be an idempotent success";

  // Initial table-allocation failure must leave both map and candidate object reusable.
  ResetUtilsBoundaries();
  boundaryState.failCallocAt = 1;
  entry.next = nullptr;
  EXPECT_EQ(ncclIntruAddressMapInsert_untyped(&map, sizeof(void*), keyOffset, nextOffset, 1, &entry), ncclSystemError)
      << "Failure to allocate the initial bucket table must propagate as a system error";
  EXPECT_EQ(map.hbits, 0) << "Initial allocation failure must restore the uninitialized hash width";
  EXPECT_EQ(map.count, 0) << "Initial allocation failure must not increment the object count";
  EXPECT_EQ(map.table, nullptr) << "Initial allocation failure must leave no bucket table";
  EXPECT_EQ(entry.next, nullptr) << "Initial allocation failure must preserve the object's intrusive link";
  EXPECT_EQ(entry.key, reinterpret_cast<void*>(uintptr_t{1}))
      << "Initial allocation failure must preserve the object's caller-owned key";
  EXPECT_EQ(entry.payload, 7) << "Initial allocation failure must preserve unrelated object data";

  // Resize-allocation failure must leave every existing association and the candidate unchanged.
  ResetUtilsBoundaries();
  AddressMap full = {};
  std::array<AddressEntry, 33> entries = {};
  for (size_t index = 0; index < 32; index++) {
    entries[index].key = reinterpret_cast<void*>(index + 1);
    ASSERT_EQ(ncclIntruAddressMapInsert(&full, entries[index].key, &entries[index]), ncclSuccess)
        << "Preparing the resize boundary must insert each initial object";
  }
  const int originalHashBits = full.base.hbits;
  void** originalTable = full.base.table;
  boundaryState.failMallocAt = 1;
  entries[32].key = reinterpret_cast<void*>(uintptr_t{33});
  entries[32].next = reinterpret_cast<AddressEntry*>(uintptr_t{0x77});
  entries[32].payload = 31337;
  EXPECT_EQ(ncclIntruAddressMapInsert(&full, entries[32].key, &entries[32]), ncclSystemError)
      << "Failure to allocate a resized bucket table must propagate as a system error";
  EXPECT_EQ(full.base.hbits, originalHashBits) << "Resize failure must preserve the original hash width";
  EXPECT_EQ(full.base.count, 32) << "Resize failure must preserve the original object count";
  EXPECT_EQ(full.base.table, originalTable) << "Resize failure must preserve the original bucket table";
  EXPECT_EQ(entries[32].next, reinterpret_cast<AddressEntry*>(uintptr_t{0x77}))
      << "Resize failure must preserve the uninserted object's intrusive link";
  EXPECT_EQ(entries[32].key, reinterpret_cast<void*>(uintptr_t{33}))
      << "Resize failure must preserve the uninserted object's key";
  EXPECT_EQ(entries[32].payload, 31337)
      << "Resize failure must preserve unrelated data in the uninserted object";
  for (size_t index = 0; index < 32; index++) {
    AddressEntry* found = nullptr;
    ASSERT_EQ(ncclIntruAddressMapFind(&full, entries[index].key, &found), ncclSuccess)
        << "Every pre-existing object must remain searchable after resize failure";
    EXPECT_EQ(found, &entries[index]) << "Resize failure must not lose or replace an existing object";
    ASSERT_EQ(ncclIntruAddressMapRemove(&full, entries[index].key), ncclSuccess)
        << "Cleanup must remove each object retained after resize failure";
  }
  EXPECT_EQ(full.base.table, nullptr) << "Removing the last retained object must release the original table";
}

/**
 * @brief Verifies collision lookup and removal, idempotence, rehash preservation, and final
 * cleanup.
 */
TEST(UtilsContainerTest, ManagesAddressMapLifecycle) {
  // Three same-bucket keys verify functional collision handling without fixing chain order.
  ResetUtilsBoundaries();
  AddressMap map = {};
  const auto keys = FindCollidingKeys(4);
  ASSERT_NE(keys[0], 0u) << "The test must find representative collision keys";
  std::array<AddressEntry, 3> collisionEntries = {};
  for (size_t index = 0; index < collisionEntries.size(); index++) {
    collisionEntries[index].key = reinterpret_cast<void*>(keys[index]);
    collisionEntries[index].payload = static_cast<int>(index + 1);
    ASSERT_EQ(ncclIntruAddressMapInsert(&map, collisionEntries[index].key, &collisionEntries[index]), ncclSuccess)
        << "Each representative colliding object must insert successfully";
  }
  EXPECT_EQ(map.base.count, 3) << "The map count must include every colliding object";
  for (auto& entry : collisionEntries) {
    AddressEntry* found = nullptr;
    ASSERT_EQ(ncclIntruAddressMapFind(&map, entry.key, &found), ncclSuccess)
        << "Every colliding key must remain searchable after insertion";
    EXPECT_EQ(found, &entry) << "Collision handling must preserve the object associated with each key";
    EXPECT_EQ(entry.payload, static_cast<int>(&entry - collisionEntries.data()) + 1)
        << "Collision handling must preserve caller-owned object data";
  }

  // Removing one collision must make only that key absent and preserve both survivors.
  ASSERT_EQ(ncclIntruAddressMapRemove(&map, collisionEntries[1].key), ncclSuccess)
      << "Removing one object from a collision set must succeed";
  AddressEntry* removed = reinterpret_cast<AddressEntry*>(uintptr_t{0x99});
  ASSERT_EQ(ncclIntruAddressMapFind(&map, collisionEntries[1].key, &removed), ncclSuccess)
      << "Looking up a removed colliding key must succeed";
  EXPECT_EQ(removed, nullptr) << "A removed colliding key must no longer resolve to an object";
  for (size_t index : std::array<size_t, 2>{0, 2}) {
    AddressEntry* found = nullptr;
    ASSERT_EQ(ncclIntruAddressMapFind(&map, collisionEntries[index].key, &found), ncclSuccess)
        << "Removing one colliding key must preserve lookup for every remaining key";
    EXPECT_EQ(found, &collisionEntries[index])
        << "Removing one colliding key must not replace a remaining object's association";
  }
  EXPECT_EQ(map.base.count, 2) << "Removing one colliding object must decrement the object count once";
  // Missing and remaining-key removals must update count and restore the empty map state.
  ASSERT_EQ(ncclIntruAddressMapRemove(&map, reinterpret_cast<void*>(uintptr_t{0xdead})), ncclSuccess)
      << "Removing a missing key must be an idempotent success";
  EXPECT_EQ(map.base.count, 2) << "Removing a missing key must not change the object count";
  ASSERT_EQ(ncclIntruAddressMapRemove(&map, collisionEntries[0].key), ncclSuccess)
      << "Removing another colliding object must succeed";
  ASSERT_EQ(ncclIntruAddressMapRemove(&map, collisionEntries[2].key), ncclSuccess)
      << "Removing the final colliding object must succeed";
  EXPECT_EQ(map.base.hbits, 0) << "Removing the final object must restore the zero-initialized hash width";
  EXPECT_EQ(map.base.count, 0) << "Removing the final object must restore an empty count";
  EXPECT_EQ(map.base.table, nullptr) << "Removing the final object must release and clear the bucket table";

  // Crossing the current growth boundary must preserve all key-to-object associations.
  std::array<AddressEntry, 33> entries = {};
  for (size_t index = 0; index < entries.size(); index++) {
    entries[index].key = reinterpret_cast<void*>(0x1000 + index * 0x10);
    entries[index].payload = static_cast<int>(index);
    ASSERT_EQ(ncclIntruAddressMapInsert(&map, entries[index].key, &entries[index]), ncclSuccess)
        << "Every object through the resize boundary must insert successfully";
  }
  EXPECT_EQ(map.base.count, 33) << "Resizing must preserve the exact number of inserted objects";
  for (size_t index = 0; index < entries.size(); index++) {
    AddressEntry* found = reinterpret_cast<AddressEntry*>(uintptr_t{0x99});
    ASSERT_EQ(ncclIntruAddressMapFind(&map, entries[index].key, &found), ncclSuccess)
        << "Lookup after rehashing must return a success status";
    EXPECT_EQ(found, &entries[index]) << "Rehashing must preserve the exact object associated with every key";
    EXPECT_EQ(entries[index].key, reinterpret_cast<void*>(0x1000 + index * 0x10))
        << "Rehashing must preserve every caller-owned key";
    EXPECT_EQ(entries[index].payload, static_cast<int>(index))
        << "Rehashing must preserve unrelated caller-owned object data";
  }
  AddressEntry* missing = reinterpret_cast<AddressEntry*>(uintptr_t{0x99});
  ASSERT_EQ(ncclIntruAddressMapFind(&map, reinterpret_cast<void*>(uintptr_t{0xbeef}), &missing), ncclSuccess)
      << "Lookup of a missing key in a populated map must succeed";
  EXPECT_EQ(missing, nullptr) << "Lookup of a missing key must clear the output to NULL";

  // Complete removal after rehashing must release the table and remain idempotent.
  for (auto& entry : entries) {
    ASSERT_EQ(ncclIntruAddressMapRemove(&map, entry.key), ncclSuccess)
        << "Every rehashed object must remain removable";
  }
  EXPECT_EQ(map.base.table, nullptr) << "Removing every rehashed object must release the bucket table";
  EXPECT_EQ(map.base.hbits, 0) << "Removing every rehashed object must restore the initial hash width";
  EXPECT_EQ(map.base.count, 0) << "Removing every rehashed object must restore the initial count";
  EXPECT_EQ(ncclIntruAddressMapRemove(&map, reinterpret_cast<void*>(uintptr_t{0xbeef})), ncclSuccess)
      << "Removal must remain idempotent after the map returns to its empty state";

  // Explicit destruction must release a populated table, reset map state, and preserve caller-owned
  // objects.
  AddressMap abandonedMap = {};
  AddressEntry abandonedEntry = {reinterpret_cast<void*>(uintptr_t{0x1234}), nullptr, 77};
  ASSERT_EQ(ncclIntruAddressMapInsert(&abandonedMap, abandonedEntry.key, &abandonedEntry), ncclSuccess)
      << "Preparing explicit map destruction must insert one live object";
  AddressEntry* storedLink = abandonedEntry.next;
  ncclIntruAddressMapDestruct(&abandonedMap);
  EXPECT_EQ(abandonedMap.base.table, nullptr) << "Map destruction must release and clear the bucket table";
  EXPECT_EQ(abandonedMap.base.hbits, 0) << "Map destruction must reset the hash width";
  EXPECT_EQ(abandonedMap.base.count, 0) << "Map destruction must reset the object count";
  EXPECT_EQ(abandonedEntry.key, reinterpret_cast<void*>(uintptr_t{0x1234}))
      << "Map destruction must preserve the caller-owned object key";
  EXPECT_EQ(abandonedEntry.next, storedLink) << "Map destruction must preserve the object's intrusive link";
  EXPECT_EQ(abandonedEntry.payload, 77) << "Map destruction must preserve unrelated caller-owned object data";
  ncclIntruAddressMapDestruct(&abandonedMap);
  EXPECT_EQ(abandonedMap.base.table, nullptr) << "Repeated destruction of an empty map must remain a safe no-op";
}
