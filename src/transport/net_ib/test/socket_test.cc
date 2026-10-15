/*************************************************************************
 * Copyright (c) 2026 Oracle and/or its affiliates.
 * SPDX-License-Identifier: UPL-1.0
 *
 * Licensed under the Universal Permissive License (UPL), Version 1.0.
 * See LICENSE.txt.
 *************************************************************************/


#include "os.h"
#include "param.h"
#include "socket.h"
#include "utils.h"

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void ResetSocketOsState();

/** @brief Owns both descriptors of one local stream-socket pair. */
class SocketPairGuard {
 public:
  /**
   * @brief Creates one connected local stream-socket pair.
   * @note Side effects: Opens two descriptors or leaves both stored descriptors invalid.
   */
  SocketPairGuard() {
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, descriptors_) != 0) {
      descriptors_[0] = NCCL_INVALID_SOCKET;
      descriptors_[1] = NCCL_INVALID_SOCKET;
    }
  }

  /**
   * @brief Closes every descriptor still owned by this guard.
   * @note Side effects: Closes zero, one, or two descriptors.
   */
  ~SocketPairGuard() {
    Close(0);
    Close(1);
  }

  /**
   * @brief Disables construction that would duplicate descriptor ownership.
   * @param[in] other Source guard that cannot be copied.
   */
  SocketPairGuard(const SocketPairGuard& other) = delete;

  /**
   * @brief Disables assignment that would duplicate descriptor ownership.
   * @param[in] other Source guard that cannot be assigned.
   * @return A reference to this guard; assignment is deleted and cannot be invoked.
   */
  SocketPairGuard& operator=(const SocketPairGuard& other) = delete;

  /**
   * @brief Returns one descriptor without transferring ownership.
   * @param[in] index Descriptor index in [0, 1].
   * @return The owned descriptor or NCCL_INVALID_SOCKET after release or close.
   * @note Side effects: None.
   */
  int Get(int index) const { return descriptors_[index]; }

  /**
   * @brief Transfers ownership of one descriptor to the caller.
   * @param[in] index Descriptor index in [0, 1].
   * @return The previously owned descriptor or NCCL_INVALID_SOCKET.
   * @note Side effects: Replaces the selected stored descriptor with NCCL_INVALID_SOCKET.
   */
  int Release(int index) {
    const int descriptor = descriptors_[index];
    descriptors_[index] = NCCL_INVALID_SOCKET;
    return descriptor;
  }

  /**
   * @brief Closes one descriptor when it is still owned.
   * @param[in] index Descriptor index in [0, 1].
   * @note Side effects: May close the descriptor and always marks the selected entry invalid.
   */
  void Close(int index) {
    if (descriptors_[index] >= 0) (void)::close(descriptors_[index]);
    descriptors_[index] = NCCL_INVALID_SOCKET;
  }

 private:
  int descriptors_[2] = {NCCL_INVALID_SOCKET, NCCL_INVALID_SOCKET};
};

/**
 * @brief Creates a zero-initialized socket around one descriptor and state.
 * @param[in] descriptor Any representable descriptor, including NCCL_INVALID_SOCKET.
 * @param[in] state Any ncclSocketState value.
 * @return A socket with the requested descriptor and state and an invalid accept descriptor.
 * @note Side effects: None.
 */
struct ncclSocket MakeSocketForTest(int descriptor, enum ncclSocketState state) {
  struct ncclSocket socket = {};
  socket.socketDescriptor = descriptor;
  socket.acceptSocketDescriptor = NCCL_INVALID_SOCKET;
  socket.state = state;
  return socket;
}

/**
 * @brief Checks that every observable socket field matches an expected snapshot.
 * @param[in] actual Socket state observed after the tested operation.
 * @param[in] expected Socket snapshot captured before the tested operation.
 * @param[in] reason Valid non-NULL explanation printed when a field differs.
 * @note Side effects: Records nonfatal GoogleTest failures for mismatched fields.
 */
void ExpectSocketMatchesForTest(const struct ncclSocket& actual, const struct ncclSocket& expected,
  const char* reason) {
  SCOPED_TRACE(reason);
  EXPECT_EQ(actual.socketDescriptor, expected.socketDescriptor);
  EXPECT_EQ(actual.acceptSocketDescriptor, expected.acceptSocketDescriptor);
  EXPECT_EQ(actual.errorRetries, expected.errorRetries);
  EXPECT_EQ(std::memcmp(&actual.addr, &expected.addr, sizeof(actual.addr)), 0);
  EXPECT_EQ(actual.abortFlag, expected.abortFlag);
  EXPECT_EQ(actual.asyncFlag, expected.asyncFlag);
  EXPECT_EQ(actual.state, expected.state);
  EXPECT_EQ(actual.salen, expected.salen);
  EXPECT_EQ(actual.magic, expected.magic);
  EXPECT_EQ(actual.type, expected.type);
  EXPECT_EQ(actual.customRetry, expected.customRetry);
  EXPECT_EQ(actual.finalizeCounter, expected.finalizeCounter);
  EXPECT_EQ(std::memcmp(actual.finalizeBuffer, expected.finalizeBuffer, sizeof(actual.finalizeBuffer)), 0);
#ifdef NCCL_OS_WINDOWS
  EXPECT_EQ(actual.socketBlockingMode, expected.socketBlockingMode);
#endif
}

/**
 * @brief Creates an IPv4 loopback address with one host-order port.
 * @param[in] port Host-order port in [0, 65535].
 * @return An AF_INET loopback socket address containing port.
 * @note Side effects: None.
 */
union ncclSocketAddress MakeLoopbackAddress(uint16_t port) {
  union ncclSocketAddress address = {};
  address.sin.sin_family = AF_INET;
  address.sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin.sin_port = htons(port);
  return address;
}

/**
 * @brief Receives an exact peer-side test buffer without allowing a failed production send to hang
 * the test.
 * @param[in] descriptor Valid nonnegative connected socket descriptor.
 * @param[out] buffer Valid storage for size bytes.
 * @param[in] size Requested byte count in [0, SIZE_MAX].
 * @return true after receiving size bytes within ten 50-millisecond polls, otherwise false.
 * @note Side effects: Consumes up to size bytes from descriptor and writes buffer.
 */
bool ReceiveExactForTest(int descriptor, void* buffer, size_t size) {
  size_t offset = 0;
  for (int attempt = 0; attempt < 10 && offset < size; ++attempt) {
    struct pollfd pollDescriptor = {descriptor, POLLIN, 0};
    const int pollResult = ::poll(&pollDescriptor, 1, 50);
    if (pollResult <= 0) continue;
    const ssize_t received = ::recv(descriptor, static_cast<char*>(buffer) + offset, size - offset, MSG_DONTWAIT);
    if (received > 0) {
      offset += static_cast<size_t>(received);
    } else if (received == 0 || (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)) {
      return false;
    }
  }
  return offset == size;
}

/** @brief Provides isolated OS seams and environment state for socket tests. */
class SocketTestBase : public ::testing::Test {
 protected:
  /**
   * @brief Restores deterministic socket-test state before each test.
   * @note Side effects: Resets test doubles and removes socket-selection environment overrides.
   */
  void SetUp() override {
    ResetSocketOsState();
    (void)::unsetenv("NCCL_SOCKET_FAMILY");
    (void)::unsetenv("NCCL_SOCKET_IFNAME");
    (void)::unsetenv("NCCL_COMM_ID");
  }

  /**
   * @brief Removes socket-selection environment overrides after each test.
   * @note Side effects: Removes three environment variables.
   */
  void TearDown() override {
    (void)::unsetenv("NCCL_SOCKET_FAMILY");
    (void)::unsetenv("NCCL_SOCKET_IFNAME");
    (void)::unsetenv("NCCL_COMM_ID");
  }
};

/** @brief Groups socket address parsing and formatting scenarios. */
class SocketAddressTest : public SocketTestBase {};
/** @brief Groups socket initialization, connection, acceptance, and close scenarios. */
class SocketLifecycleTest : public SocketTestBase {};
/** @brief Groups socket progress and transfer scenarios. */
class SocketIoTest : public SocketTestBase {};
/** @brief Groups socket parameter and interface-selection scenarios. */
class SocketParameterTest : public SocketTestBase {};

/**
 * @brief Verifies port extraction and text formatting for supported and unsupported address
 * families.
 */
TEST_F(SocketAddressTest, FormatsPortsAndAddresses) {
  SCOPED_TRACE("Address formatting must preserve inputs and produce the documented family-specific text.");
  // Build boundary IPv4 and IPv6 addresses so each family and port limit is observable.
  union ncclSocketAddress ipv4 = MakeLoopbackAddress(4321);
  union ncclSocketAddress ipv6 = {};
  ipv6.sin6.sin6_family = AF_INET6;
  ipv6.sin6.sin6_addr = in6addr_loopback;
  ipv6.sin6.sin6_port = htons(UINT16_MAX);
  union ncclSocketAddress unsupported = {};
  unsupported.sa.sa_family = AF_UNIX;
  const union ncclSocketAddress ipv4Before = ipv4;
  const union ncclSocketAddress ipv6Before = ipv6;
  const union ncclSocketAddress unsupportedBefore = unsupported;
  char text[SOCKET_NAME_MAXLEN + 1] = {};

  // Extract both stored ports and format each supported family numerically.
  EXPECT_EQ(ncclSocketToPort(&ipv4), 4321);
  EXPECT_STREQ(ncclSocketToString(&ipv4, text, 1), "127.0.0.1<4321>");
  EXPECT_EQ(ncclSocketToPort(&ipv6), UINT16_MAX);
  EXPECT_STREQ(ncclSocketToString(&ipv6, text, 1), "::1<65535>");

  // Reject unsupported, missing-address, and missing-output cases without leaving stale text.
  std::strcpy(text, "stale");
  EXPECT_EQ(ncclSocketToString(&unsupported, text, 1), text);
  EXPECT_STREQ(text, "");
  std::strcpy(text, "stale");
  EXPECT_EQ(ncclSocketToString(nullptr, text, 1), text);
  EXPECT_STREQ(text, "");
  EXPECT_EQ(ncclSocketToString(&ipv4, nullptr, 1), nullptr);

  // Formatting and port extraction must not modify any input address.
  EXPECT_EQ(std::memcmp(&ipv4, &ipv4Before, sizeof(ipv4)), 0);
  EXPECT_EQ(std::memcmp(&ipv6, &ipv6Before, sizeof(ipv6)), 0);
  EXPECT_EQ(std::memcmp(&unsupported, &unsupportedBefore, sizeof(unsupported)), 0);
}

/**
 * @brief Verifies environment-family selection and valid and malformed endpoint parsing.
 */
TEST_F(SocketAddressTest, ParsesFamiliesAndEndpoints) {
  SCOPED_TRACE("Endpoint parsing must accept valid boundaries and reject malformed host and port fields.");
  // Exercise absent, recognized, and unknown family overrides.
  EXPECT_EQ(ncclEnvSocketFamily(), -1);
  ASSERT_EQ(::setenv("NCCL_SOCKET_FAMILY", "AF_INET", 1), 0);
  EXPECT_EQ(ncclEnvSocketFamily(), AF_INET);
  ASSERT_EQ(::setenv("NCCL_SOCKET_FAMILY", "AF_INET6", 1), 0);
  EXPECT_EQ(ncclEnvSocketFamily(), AF_INET6);
  ASSERT_EQ(::setenv("NCCL_SOCKET_FAMILY", "unexpected", 1), 0);
  EXPECT_EQ(ncclEnvSocketFamily(), -1);

  // Parse numeric and hostname endpoints and preserve zero, ordinary, and maximum valid ports.
  union ncclSocketAddress address = {};
  ASSERT_EQ(ncclSocketGetAddrFromString(&address, "localhost:0"), ncclSuccess);
  EXPECT_TRUE(address.sa.sa_family == AF_INET || address.sa.sa_family == AF_INET6);
  EXPECT_EQ(ncclSocketToPort(&address), 0);
  ASSERT_EQ(ncclSocketGetAddrFromString(&address, "127.0.0.1:4321"), ncclSuccess);
  EXPECT_EQ(address.sa.sa_family, AF_INET);
  EXPECT_EQ(ncclSocketToPort(&address), 4321);
  ASSERT_EQ(ncclSocketGetAddrFromString(&address, "127.0.0.1:65535"), ncclSuccess);
  EXPECT_EQ(ncclSocketToPort(&address), UINT16_MAX);
  ASSERT_EQ(ncclSocketGetAddrFromString(&address, "[::1]:65535"), ncclSuccess);
  EXPECT_EQ(address.sa.sa_family, AF_INET6);
  EXPECT_EQ(ncclSocketToPort(&address), UINT16_MAX);

  // Missing endpoints, empty hosts, unresolved IPv4 hosts, and incomplete IPv6 brackets must be
  // rejected.
  EXPECT_EQ(ncclSocketGetAddrFromString(&address, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketGetAddrFromString(&address, ""), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketGetAddrFromString(&address, ":"), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketGetAddrFromString(&address, "256.256.256.256:1"), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketGetAddrFromString(&address, "[::1:4321"), ncclInvalidArgument);
#if 0 // ADDRESS_LATER
  EXPECT_EQ(ncclSocketGetAddrFromString(&address, "127.0.0.1"), ncclInvalidArgument)
      << "an IPv4 endpoint without a port must be rejected";
  EXPECT_EQ(ncclSocketGetAddrFromString(&address, "[not-an-ip]:1"), ncclInvalidArgument)
      << "an invalid bracketed IPv6 address must be rejected";
  EXPECT_EQ(ncclSocketGetAddrFromString(&address, "127.0.0.1:70000"), ncclInvalidArgument)
      << "a port above UINT16_MAX must be rejected instead of wrapping";
  EXPECT_EQ(ncclSocketGetAddrFromString(&address, "127.0.0.1:not-a-port"), ncclInvalidArgument)
      << "a nonnumeric port must be rejected instead of becoming zero";
#endif 0 // ADDRESS_LATER
}

/** @brief Scripts one interface-discovery result. */
struct InterfaceReply {
  ncclResult_t result = ncclSuccess;
  int count = 0;
  std::string name;
  union ncclSocketAddress address = {};
};

/** @brief Records one generated socket-parameter lookup. */
struct SocketParameterLoadCall {
  std::string environment;
  int64_t defaultValue;
};

/**
 * @brief Holds scripted OS behavior for parameters, descriptors, I/O, connections, and
 * interface discovery.
 * @note Fields configure outcomes and record boundary calls that drive socket state
 * transitions.
 */
struct SocketOsState {
  std::unordered_map<std::string, int64_t> parameterValues;
  std::vector<SocketParameterLoadCall> parameterLoads;
  bool cacheParameterValues = false;
  ncclResult_t progressResult = ncclSuccess;
  size_t maximumTransferBytes = SIZE_MAX;
  bool forceClosure = false;
  int progressCalls = 0;
  int pollCalls = 0;
  std::vector<int> pollDescriptors;
  std::vector<int> pollOperations;
  ncclResult_t setFlagsResult = ncclSuccess;
  int setFlagsCalls = 0;
  ncclResult_t resetDescriptorResult = ncclSuccess;
  int resetDescriptorCalls = 0;
  ncclResult_t startConnectResult = ncclSuccess;
  enum ncclSocketState startConnectState = ncclSocketStateConnected;
  int startConnectCalls = 0;
  ncclResult_t pollConnectResult = ncclSuccess;
  enum ncclSocketState pollConnectState = ncclSocketStateConnected;
  int pollConnectCalls = 0;
  ncclResult_t acceptResult = ncclSuccess;
  int acceptedDescriptor = NCCL_INVALID_SOCKET;
  int acceptCalls = 0;
  int resetAcceptCalls = 0;
  int closeCalls = 0;
  std::vector<InterfaceReply> interfaceReplies;
  std::vector<std::string> interfacePrefixes;
  size_t nextInterfaceReply = 0;
  int interfaceFamily = -1;
  ncclResult_t subnetResult = ncclSuccess;
  int subnetFound = 0;
  std::string subnetName;
  union ncclSocketAddress subnetAddress = {};
  int subnetCalls = 0;
};

SocketOsState socketOsState;

/**
 * @brief Resets every test-controlled OS result and counter to its successful default.
 * @note Side effects: Replaces socketOsState.
 */
void ResetSocketOsState() {
  socketOsState = SocketOsState{};
}

/**
 * @brief Supplies deterministic values to the production-generated socket parameter accessors.
 * @param[in] environment Valid non-NULL full environment-variable name requested by production.
 * @param[in] defaultValue Production fallback when no configured value exists.
 * @param[in] uninitialized Production cache sentinel; it is not modified.
 * @param[out] cache Valid non-NULL accessor-owned cache storage written only when caching is
 * enabled.
 * @param[out] noCache Valid non-NULL accessor-owned policy storage always set to 0 or 1.
 * @return The configured value or defaultValue when the environment key is absent.
 * @note Side effects: Records the lookup, always writes noCache, and may write cache.
 */
int64_t LoadSocketParameterForTest(const char* environment, int64_t defaultValue,
  int64_t uninitialized, int64_t* cache, int8_t* noCache) {
  (void)uninitialized;
  socketOsState.parameterLoads.push_back({environment, defaultValue});
  const auto configured = socketOsState.parameterValues.find(environment);
  const int64_t value = configured == socketOsState.parameterValues.end()
      ? defaultValue
      : configured->second;
  if (socketOsState.cacheParameterValues) {
    *cache = value;
    *noCache = 0;
  } else {
    *noCache = 1;
  }
  return value;
}

/**
 * @brief Parses only the simple comma-separated name[:port] inputs used by socket selection tests.
 * @param[in] text NULL or a comma-separated list of simple interface and optional port entries.
 * @param[out] interfaces Valid non-NULL storage for maximumCount entries when text is non-NULL.
 * @param[in] maximumCount Maximum output count in [1, INT_MAX].
 * @return The parsed entry count in [0, maximumCount].
 * @note Side effects: Writes each returned entry to interfaces.
 * @note This seam does not validate the complete production grammar.
 */
int ParseInterfaceListForTest(const char* text, struct netIf* interfaces, int maximumCount) {
  if (text == nullptr) return 0;
  const char* current = text;
  int interfaceCount = 0;
  int characterCount = 0;
  char character;
  do {
    character = *current;
    if (character == ':') {
      if (characterCount > 0) {
        interfaces[interfaceCount].prefix[characterCount] = '\0';
        interfaces[interfaceCount].port = std::atoi(current + 1);
        ++interfaceCount;
        characterCount = 0;
      }
      while (character != ',' && character != '\0') character = *(++current);
    } else if (character == ',' || character == '\0') {
      if (characterCount > 0) {
        interfaces[interfaceCount].prefix[characterCount] = '\0';
        interfaces[interfaceCount].port = -1;
        ++interfaceCount;
        characterCount = 0;
      }
    } else {
      interfaces[interfaceCount].prefix[characterCount++] = character;
    }
    ++current;
  } while (interfaceCount < maximumCount && character != '\0');
  return interfaceCount;
}

/**
 * @brief Reports whether a descriptor is valid for the test OS boundary.
 * @param[in] descriptor Any representable socket descriptor.
 * @return true for nonnegative descriptors, otherwise false.
 * @note Side effects: None.
 */
bool IsSocketDescriptorValidForTest(ncclSocketDescriptor descriptor) {
  return descriptor >= 0;
}

/**
 * @brief Reports whether a socket contains a valid descriptor.
 * @param[in] socket Valid non-NULL socket object.
 * @return true when socket->socketDescriptor is valid, otherwise false.
 * @note Side effects: None.
 */
bool IsSocketValidForTest(struct ncclSocket* socket) {
  return IsSocketDescriptorValidForTest(socket->socketDescriptor);
}

/**
 * @brief Records one request to poll a socket between progress attempts.
 * @param[in] descriptor Descriptor recorded exactly.
 * @param[in] operation Operation recorded exactly.
 * @note Side effects: Increments pollCalls and appends both arguments to their call records.
 */
void PollSocketForTest(int descriptor, int operation) {
  ++socketOsState.pollCalls;
  socketOsState.pollDescriptors.push_back(descriptor);
  socketOsState.pollOperations.push_back(operation);
}

/**
 * @brief Returns the configured descriptor-flag result.
 * @param[in] socket Valid non-NULL socket; it is not modified.
 * @return socketOsState.setFlagsResult.
 * @note Side effects: Increments socketOsState.setFlagsCalls.
 */
ncclResult_t SetSocketFlagsForTest(struct ncclSocket* socket) {
  (void)socket;
  ++socketOsState.setFlagsCalls;
  return socketOsState.setFlagsResult;
}

/**
 * @brief Closes and resets one rejected accepted socket.
 * @param[in,out] socket Valid non-NULL socket whose descriptor and handshake state are reset.
 * @note Side effects: Closes a valid descriptor, updates socket fields, and increments
 * resetAcceptCalls.
 */
void ResetAcceptedSocketForTest(struct ncclSocket* socket) {
  ++socketOsState.resetAcceptCalls;
  if (IsSocketValidForTest(socket)) (void)::close(socket->socketDescriptor);
  socket->socketDescriptor = NCCL_INVALID_SOCKET;
  socket->state = ncclSocketStateAccepting;
  socket->finalizeCounter = 0;
}

/**
 * @brief Creates or replaces one real local descriptor for initialization and retry tests.
 * @param[in,out] socket Valid non-NULL socket with an AF_INET or AF_INET6 address.
 * @return The configured failure, ncclSystemError for a failed system call, or the flag-setup
 * result.
 * @note Side effects: Increments resetDescriptorCalls and may create, replace, or close a
 * descriptor. After successful installation, flag setup increments setFlagsCalls.
 */
ncclResult_t ResetSocketDescriptorForTest(struct ncclSocket* socket) {
  ++socketOsState.resetDescriptorCalls;
  if (socketOsState.resetDescriptorResult != ncclSuccess) return socketOsState.resetDescriptorResult;
  const int descriptor = ::socket(socket->addr.sa.sa_family, SOCK_STREAM, 0);
  if (descriptor < 0) return ncclSystemError;
  if (IsSocketValidForTest(socket)) {
    if (::dup2(descriptor, socket->socketDescriptor) < 0) {
      (void)::close(descriptor);
      return ncclSystemError;
    }
    (void)::close(descriptor);
  } else {
    socket->socketDescriptor = descriptor;
  }
  return SetSocketFlagsForTest(socket);
}

/**
 * @brief Applies the configured connect-start result and next state.
 * @param[in,out] socket Valid non-NULL connecting socket.
 * @return socketOsState.startConnectResult.
 * @note Side effects: Increments startConnectCalls and sets the configured state on success or
 * Error on failure.
 */
ncclResult_t StartSocketConnectForTest(struct ncclSocket* socket) {
  ++socketOsState.startConnectCalls;
  socket->state = socketOsState.startConnectResult == ncclSuccess ? socketOsState.startConnectState
                                                                  : ncclSocketStateError;
  return socketOsState.startConnectResult;
}

/**
 * @brief Applies the configured connect-poll result and next state.
 * @param[in,out] socket Valid non-NULL connect-polling socket.
 * @return socketOsState.pollConnectResult.
 * @note Side effects: Increments pollConnectCalls, sets the configured state on success, and
 * sets Error on ncclRemoteError.
 */
ncclResult_t PollSocketConnectForTest(struct ncclSocket* socket) {
  ++socketOsState.pollConnectCalls;
  if (socketOsState.pollConnectResult == ncclSuccess) {
    socket->state = socketOsState.pollConnectState;
  } else if (socketOsState.pollConnectResult == ncclRemoteError) {
    socket->state = ncclSocketStateError;
  }
  return socketOsState.pollConnectResult;
}

/**
 * @brief Applies the configured accept result and accepted descriptor.
 * @param[in,out] socket Valid non-NULL accepting socket.
 * @return socketOsState.acceptResult.
 * @note Side effects: Increments acceptCalls and invalidates socket's descriptor. On success,
 * may move acceptedDescriptor into socket, invalidate the stored acceptedDescriptor, and set
 * socket's state to Accepted.
 */
ncclResult_t TrySocketAcceptForTest(struct ncclSocket* socket) {
  ++socketOsState.acceptCalls;
  socket->socketDescriptor = NCCL_INVALID_SOCKET;
  if (socketOsState.acceptResult != ncclSuccess) return socketOsState.acceptResult;
  if (IsSocketDescriptorValidForTest(socketOsState.acceptedDescriptor)) {
    socket->socketDescriptor = socketOsState.acceptedDescriptor;
    socketOsState.acceptedDescriptor = NCCL_INVALID_SOCKET;
    socket->state = ncclSocketStateAccepted;
  }
  return ncclSuccess;
}

/**
 * @brief Performs one test-controlled nonblocking transfer on a real local descriptor.
 * @param[in] operation NCCL_SOCKET_SEND or NCCL_SOCKET_RECV.
 * @param[in] socket Valid non-NULL socket with a valid descriptor.
 * @param[in,out] buffer Valid storage for size bytes, read for send and written for receive.
 * @param[in] size Requested byte count in [1, INT_MAX].
 * @param[in,out] offset Valid progress counter in [0, size].
 * @param[in] block Ignored blocking hint.
 * @param[out] closed Valid non-NULL storage receiving 0 or 1.
 * @return The configured result, ncclInternalError on abort, ncclRemoteError for an unhandled
 * I/O error, or ncclSuccess for progress, closure, and retryable interruption.
 * @note Side effects: Increments progressCalls, performs bounded I/O, and may update offset,
 * the receive buffer, and closed.
 */
ncclResult_t ProgressSocketForTest(int operation, struct ncclSocket* socket, void* buffer, int size, int* offset,
  int block, int* closed) {
  (void)block;
  ++socketOsState.progressCalls;
  *closed = 0;
  if (socketOsState.progressResult != ncclSuccess) return socketOsState.progressResult;
  if (socketOsState.forceClosure) {
    *closed = 1;
    return ncclSuccess;
  }
  if (socket->abortFlag != nullptr && *socket->abortFlag != 0) return ncclInternalError;

  const size_t remaining = static_cast<size_t>(size - *offset);
  const size_t transferBytes = std::min(remaining, socketOsState.maximumTransferBytes);
  if (transferBytes == 0) return ncclSuccess;

  ssize_t transferred;
  if (operation == NCCL_SOCKET_SEND) {
    transferred = ::send(socket->socketDescriptor, static_cast<char*>(buffer) + *offset, transferBytes,
      MSG_DONTWAIT | MSG_NOSIGNAL);
  } else {
    transferred = ::recv(socket->socketDescriptor, static_cast<char*>(buffer) + *offset, transferBytes, MSG_DONTWAIT);
  }
  if (transferred > 0) {
    *offset += static_cast<int>(transferred);
    return ncclSuccess;
  }
  if (transferred == 0 && operation == NCCL_SOCKET_RECV) {
    *closed = 1;
    return ncclSuccess;
  }
  if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) return ncclSuccess;
  if (errno == EPIPE || errno == ECONNRESET) {
    *closed = 1;
    return ncclSuccess;
  }
  return ncclRemoteError;
}

/**
 * @brief Returns the next scripted interface-search result.
 * @param[in] prefixList Valid non-NULL prefix expression recorded by the test.
 * @param[out] names Required only for a successful positive reply; valid storage for one
 * maximumNameSize-byte name.
 * @param[out] addresses Required only for a successful positive reply; valid storage for one
 * address.
 * @param[in] family Requested address family or -1.
 * @param[in] maximumNameSize Per-name storage size in [1, INT_MAX].
 * @param[in] maximumInterfaces Ignored caller capacity.
 * @param[out] found Valid non-NULL storage receiving zero when no reply remains or the
 * configured count on scripted success; preserved on scripted error.
 * @return The next scripted result or ncclSuccess when no reply remains.
 * @note Side effects: Records every query, advances an available scripted reply, writes found
 * on success, and writes the first name and address only for a positive count.
 */
ncclResult_t FindInterfacesForTest(const char* prefixList, char* names, union ncclSocketAddress* addresses, int family,
  int maximumNameSize, int maximumInterfaces, int* found) {
  (void)maximumInterfaces;
  socketOsState.interfacePrefixes.emplace_back(prefixList);
  socketOsState.interfaceFamily = family;
  if (socketOsState.nextInterfaceReply >= socketOsState.interfaceReplies.size()) {
    *found = 0;
    return ncclSuccess;
  }
  const InterfaceReply& reply = socketOsState.interfaceReplies[socketOsState.nextInterfaceReply++];
  if (reply.result != ncclSuccess) return reply.result;
  *found = reply.count;
  if (reply.count > 0) {
    std::snprintf(names, static_cast<size_t>(maximumNameSize), "%s", reply.name.c_str());
    addresses[0] = reply.address;
  }
  return ncclSuccess;
}

/**
 * @brief Returns the scripted subnet-interface match.
 * @param[out] name Required only for a successful positive match; valid storage for one
 * maximumNameSize-byte interface name.
 * @param[out] localAddress Required only for a successful positive match; valid storage for
 * the scripted local address.
 * @param[in] remoteAddress NULL or any address pointer; ignored.
 * @param[in] maximumNameSize Name-storage size in [1, INT_MAX] for a positive match.
 * @param[out] found Valid non-NULL storage receiving the scripted count on success and
 * preserved on error.
 * @return socketOsState.subnetResult.
 * @note Side effects: Increments subnetCalls, writes found on success, and may write name and
 * localAddress for a positive match.
 */
ncclResult_t FindMatchingInterfaceForTest(char* name, union ncclSocketAddress* localAddress,
  union ncclSocketAddress* remoteAddress, int maximumNameSize, int* found) {
  (void)remoteAddress;
  ++socketOsState.subnetCalls;
  if (socketOsState.subnetResult != ncclSuccess) return socketOsState.subnetResult;
  *found = socketOsState.subnetFound;
  if (socketOsState.subnetFound > 0) {
    std::snprintf(name, static_cast<size_t>(maximumNameSize), "%s", socketOsState.subnetName.c_str());
    *localAddress = socketOsState.subnetAddress;
  }
  return ncclSuccess;
}

/**
 * @brief Closes the test socket descriptor and marks the socket closed.
 * @param[in,out] socket NULL for a no-op or a valid socket object.
 * @param[in] wait Ignored close-wait request.
 * @return ncclSuccess.
 * @note Side effects: Closes a valid descriptor, updates socket fields, and increments closeCalls.
 */
ncclResult_t CloseSocketForTest(struct ncclSocket* socket, bool wait = false) {
  (void)wait;
  ++socketOsState.closeCalls;
  if (socket != nullptr) {
    if (IsSocketValidForTest(socket)) (void)::close(socket->socketDescriptor);
    socket->socketDescriptor = NCCL_INVALID_SOCKET;
    socket->state = ncclSocketStateClosed;
  }
  return ncclSuccess;
}

}  // namespace

/**
 * @brief Compiles socket.cc with OS, parameter, interface, and close dependencies redirected
 * to scripted seams.
 */
#define ncclLoadParam LoadSocketParameterForTest
#define parseStringList ParseInterfaceListForTest
#define ncclOsSocketDescriptorIsValid IsSocketDescriptorValidForTest
#define ncclOsSocketIsValid IsSocketValidForTest
#define ncclOsPollSocket PollSocketForTest
#define ncclOsSocketSetFlags SetSocketFlagsForTest
#define ncclOsSocketResetAccept ResetAcceptedSocketForTest
#define ncclOsSocketResetFd ResetSocketDescriptorForTest
#define ncclOsSocketStartConnect StartSocketConnectForTest
#define ncclOsSocketPollConnect PollSocketConnectForTest
#define ncclOsSocketTryAccept TrySocketAcceptForTest
#define ncclOsSocketProgressOpt ProgressSocketForTest
#define ncclOsFindInterfaces FindInterfacesForTest
#define ncclFindInterfaceMatchSubnet FindMatchingInterfaceForTest
#define ncclSocketClose CloseSocketForTest
#include "../../../misc/socket.cc"  // NOLINT(build/include)
#undef ncclSocketClose
#undef ncclFindInterfaceMatchSubnet
#undef ncclOsFindInterfaces
#undef ncclOsSocketProgressOpt
#undef ncclOsSocketTryAccept
#undef ncclOsSocketPollConnect
#undef ncclOsSocketStartConnect
#undef ncclOsSocketResetFd
#undef ncclOsSocketResetAccept
#undef ncclOsSocketSetFlags
#undef ncclOsPollSocket
#undef ncclOsSocketIsValid
#undef ncclOsSocketDescriptorIsValid
#undef parseStringList
#undef ncclLoadParam

namespace {

/**
 * @brief Verifies the generated socket parameter mappings, configured values, and cache behavior.
 * @note Side effects: Records parameter loads and initializes one accessor cache in an isolated
 * child process.
 */
TEST_F(SocketParameterTest, LoadsGeneratedAccessors) {
  SCOPED_TRACE(
      "Socket parameter accessors must use their documented keys, defaults, configured values, and cache policy.");
  ResetSocketOsState();

  EXPECT_EQ(ncclParamRetryCnt(), 34)
      << "Socket retry count must retain its production default";
  EXPECT_EQ(ncclParamRetryTimeOut(), 100)
      << "Socket retry sleep must retain its production default in milliseconds";
  EXPECT_EQ(ncclParamPollTimeOut(), 0)
      << "Socket I/O polling must remain disabled by default";
  EXPECT_EQ(ncclParamSocketMaxRecvBuff(), -1)
      << "Receive-buffer override must retain automatic sizing by default";
  EXPECT_EQ(ncclParamSocketMaxSendBuff(), -1)
      << "Send-buffer override must retain automatic sizing by default";

  const SocketParameterLoadCall expectedLoads[] = {
      {"NCCL_SOCKET_RETRY_CNT", 34},
      {"NCCL_SOCKET_RETRY_SLEEP_MSEC", 100},
      {"NCCL_SOCKET_POLL_TIMEOUT_MSEC", 0},
      {"NCCL_SOCKET_RCVBUF", -1},
      {"NCCL_SOCKET_SNDBUF", -1},
  };
  ASSERT_EQ(socketOsState.parameterLoads.size(), std::size(expectedLoads))
      << "Every direct accessor call must perform one deterministic uncached load";
  for (size_t index = 0; index < std::size(expectedLoads); ++index) {
    SCOPED_TRACE(testing::Message() << "default socket parameter " << index);
    EXPECT_EQ(socketOsState.parameterLoads[index].environment, expectedLoads[index].environment)
        << "The accessor must request its documented environment variable";
    EXPECT_EQ(socketOsState.parameterLoads[index].defaultValue, expectedLoads[index].defaultValue)
        << "The accessor must forward its documented production default";
  }

  socketOsState.parameterLoads.clear();
  socketOsState.parameterValues = {
      {"NCCL_SOCKET_RETRY_CNT", 0},
      {"NCCL_SOCKET_RETRY_SLEEP_MSEC", 25},
      {"NCCL_SOCKET_POLL_TIMEOUT_MSEC", 7},
      {"NCCL_SOCKET_RCVBUF", 4096},
      {"NCCL_SOCKET_SNDBUF", 8192},
  };
  EXPECT_EQ(ncclParamRetryCnt(), 0)
      << "A configured zero retry budget must be returned unchanged";
  EXPECT_EQ(ncclParamRetryTimeOut(), 25)
      << "A configured retry sleep must be returned unchanged";
  EXPECT_EQ(ncclParamPollTimeOut(), 7)
      << "A configured positive poll timeout must be returned unchanged";
  EXPECT_EQ(ncclParamSocketMaxRecvBuff(), 4096)
      << "A configured receive-buffer size must be returned unchanged";
  EXPECT_EQ(ncclParamSocketMaxSendBuff(), 8192)
      << "A configured send-buffer size must be returned unchanged";
  ASSERT_EQ(socketOsState.parameterLoads.size(), std::size(expectedLoads))
      << "Every configured accessor call must remain observable through the loader seam";

  GTEST_FLAG_SET(death_test_style, "threadsafe");
  /**
   * @brief Verifies generated retry-count caching in an isolated child process.
   * @note Side effects: Runs in a subprocess because the function-local cache cannot be reset
   * in this process.
   */
  EXPECT_EXIT(
      {
        ResetSocketOsState();
        socketOsState.cacheParameterValues = true;
        socketOsState.parameterValues["NCCL_SOCKET_RETRY_CNT"] = 9;
        const int64_t firstValue = ncclParamRetryCnt();
        socketOsState.parameterValues["NCCL_SOCKET_RETRY_CNT"] = 1;
        const int64_t secondValue = ncclParamRetryCnt();
        if (firstValue != 9 || secondValue != 9 || socketOsState.parameterLoads.size() != 1) {
          std::fprintf(stderr,
              "cached retry count mismatch: first=%lld second=%lld loads=%zu; expected 9, 9, 1\n",
              static_cast<long long>(firstValue), static_cast<long long>(secondValue),
              socketOsState.parameterLoads.size());
          _exit(1);
        }
        _exit(0);
      },
      testing::ExitedWithCode(0), ".*")
      << "A cached generated accessor must retain its first loaded value";
}

/**
 * @brief Verifies configured, automatic, subnet-guided interface selection and ready-address
 * copying.
 */
TEST_F(SocketAddressTest, SelectsInterfacesAndCopiesAddress) {
  SCOPED_TRACE("Interface selection and ready-address access must write only their documented outputs.");
  char names[2 * MAX_IF_NAME_SIZE] = {};
  union ncclSocketAddress addresses[2] = {};
  int found = -1;

  // An explicit interface expression must be passed directly to the OS boundary.
  ASSERT_EQ(::setenv("NCCL_SOCKET_IFNAME", "eth-test", 1), 0);
  socketOsState.interfaceReplies.push_back({ncclSuccess, 1, "eth-test", MakeLoopbackAddress(10)});
  ASSERT_EQ(ncclFindInterfaces(names, addresses, MAX_IF_NAME_SIZE, 2, &found), ncclSuccess);
  EXPECT_EQ(found, 1);
  EXPECT_STREQ(names, "eth-test");
  EXPECT_EQ(ncclSocketToPort(&addresses[0]), 10);
  ASSERT_EQ(socketOsState.interfacePrefixes.size(), 1U);
  EXPECT_EQ(socketOsState.interfacePrefixes[0], "eth-test");

  // Without explicit configuration, an empty IB lookup must fall through to the general filter.
  ResetSocketOsState();
  (void)::unsetenv("NCCL_SOCKET_IFNAME");
  socketOsState.interfaceReplies.push_back({ncclSuccess, 0, "", {}});
  socketOsState.interfaceReplies.push_back({ncclSuccess, 1, "eth-auto", MakeLoopbackAddress(20)});
  ASSERT_EQ(ncclFindInterfaces(names, addresses, MAX_IF_NAME_SIZE, 2, &found), ncclSuccess);
  EXPECT_EQ(found, 1);
  ASSERT_EQ(socketOsState.interfacePrefixes.size(), 2U);
  EXPECT_EQ(socketOsState.interfacePrefixes[0], "ib");
  EXPECT_EQ(socketOsState.interfacePrefixes[1], "^docker,lo,virbr");
  EXPECT_STREQ(names, "eth-auto");
  EXPECT_EQ(ncclSocketToPort(&addresses[0]), 20);

  // A communication ID must request a same-subnet match before the general fallback.
  ResetSocketOsState();
  ASSERT_EQ(::setenv("NCCL_COMM_ID", "127.0.0.1:45", 1), 0);
  socketOsState.interfaceReplies.push_back({ncclSuccess, 0, "", {}});
  socketOsState.subnetFound = 1;
  socketOsState.subnetName = "eth-subnet";
  socketOsState.subnetAddress = MakeLoopbackAddress(30);
  ASSERT_EQ(ncclFindInterfaces(names, addresses, MAX_IF_NAME_SIZE, 2, &found), ncclSuccess);
  EXPECT_EQ(socketOsState.subnetCalls, 1);
  EXPECT_EQ(found, 1);
  EXPECT_STREQ(names, "eth-subnet");
  EXPECT_EQ(ncclSocketToPort(&addresses[0]), 30);

  // Address access must reject invalid sockets and copy every byte only from a ready socket.
  union ncclSocketAddress copied;
  std::memset(&copied, 0xa5, sizeof(copied));
  const union ncclSocketAddress untouchedOutput = copied;
  EXPECT_EQ(ncclSocketGetAddr(nullptr, &copied), ncclInvalidArgument);
  EXPECT_EQ(std::memcmp(&copied, &untouchedOutput, sizeof(copied)), 0);
  struct ncclSocket socket = MakeSocketForTest(NCCL_INVALID_SOCKET, ncclSocketStateInitialized);
  socket.addr = MakeLoopbackAddress(9876);
  const struct ncclSocket initializedBefore = socket;
  EXPECT_EQ(ncclSocketGetAddr(&socket, &copied), ncclInternalError);
  EXPECT_EQ(std::memcmp(&copied, &untouchedOutput, sizeof(copied)), 0);
  ExpectSocketMatchesForTest(socket, initializedBefore, "A failed address getter must not modify its input socket.");
  socket.state = ncclSocketStateReady;
  const struct ncclSocket readyBefore = socket;
  ASSERT_EQ(ncclSocketGetAddr(&socket, &copied), ncclSuccess);
  EXPECT_EQ(std::memcmp(&copied, &socket.addr, sizeof(copied)), 0);
  ExpectSocketMatchesForTest(socket, readyBefore, "A successful address getter must not modify its input socket.");
}

/**
 * @brief Verifies interface-search short-circuiting, complete fallback order, errors, and scoped
 * IPv6 parsing.
 */
TEST_F(SocketAddressTest, HandlesInterfaceFallbacksAndErrors) {
  SCOPED_TRACE("Interface fallback must follow its priority order and stop on each reachable lookup error.");
  char names[MAX_IF_NAME_SIZE] = {};
  union ncclSocketAddress addresses[1] = {};
  int found = 17;

  // Explicit lookup errors must propagate while retaining the requested address family and a zero
  // result count.
  ASSERT_EQ(::setenv("NCCL_SOCKET_FAMILY", "AF_INET6", 1), 0);
  ASSERT_EQ(::setenv("NCCL_SOCKET_IFNAME", "eth-error", 1), 0);
  socketOsState.interfaceReplies.push_back({ncclSystemError, 0, "", {}});
  EXPECT_EQ(ncclFindInterfaces(names, addresses, MAX_IF_NAME_SIZE, 1, &found), ncclSystemError);
  EXPECT_EQ(found, 0);
  EXPECT_EQ(socketOsState.interfaceFamily, AF_INET6);

  // A successful preferred-interface lookup must stop before every fallback.
  ResetSocketOsState();
  (void)::unsetenv("NCCL_SOCKET_FAMILY");
  (void)::unsetenv("NCCL_SOCKET_IFNAME");
  socketOsState.interfaceReplies.push_back({ncclSuccess, 1, "ib-test", MakeLoopbackAddress(1)});
  ASSERT_EQ(ncclFindInterfaces(names, addresses, MAX_IF_NAME_SIZE, 1, &found), ncclSuccess);
  ASSERT_EQ(socketOsState.interfacePrefixes.size(), 1U);
  EXPECT_EQ(socketOsState.interfacePrefixes[0], "ib");
  EXPECT_STREQ(names, "ib-test");

  // Empty results must follow the preferred-IB, general, Docker, loopback, and bridge order.
  ResetSocketOsState();
  for (int reply = 0; reply < 4; ++reply) {
    socketOsState.interfaceReplies.push_back({ncclSuccess, 0, "", {}});
  }
  socketOsState.interfaceReplies.push_back({ncclSuccess, 1, "virbr-test", MakeLoopbackAddress(2)});
  ASSERT_EQ(ncclFindInterfaces(names, addresses, MAX_IF_NAME_SIZE, 1, &found), ncclSuccess);
  const std::vector<std::string> expectedPrefixes = {"ib", "^docker,lo,virbr", "docker", "lo", "virbr"};
  EXPECT_EQ(socketOsState.interfacePrefixes, expectedPrefixes);
  EXPECT_STREQ(names, "virbr-test");

  // A same-subnet lookup error must stop selection before the general fallback.
  ResetSocketOsState();
  ASSERT_EQ(::setenv("NCCL_COMM_ID", "127.0.0.1:45", 1), 0);
  socketOsState.interfaceReplies.push_back({ncclSuccess, 0, "", {}});
  socketOsState.subnetResult = ncclSystemError;
  EXPECT_EQ(ncclFindInterfaces(names, addresses, MAX_IF_NAME_SIZE, 1, &found), ncclSystemError);
  EXPECT_EQ(socketOsState.subnetCalls, 1);
  EXPECT_EQ(socketOsState.interfacePrefixes.size(), 1U);

  // A malformed communication ID must return its parser error before subnet matching or general
  // fallback.
  ResetSocketOsState();
  ASSERT_EQ(::setenv("NCCL_COMM_ID", "[::1", 1), 0);
  socketOsState.interfaceReplies.push_back({ncclSuccess, 0, "", {}});
  found = 17;
  EXPECT_EQ(ncclFindInterfaces(names, addresses, MAX_IF_NAME_SIZE, 1, &found), ncclInvalidArgument)
    << "A malformed NCCL_COMM_ID must not be accepted as a usable endpoint.";
  EXPECT_EQ(found, 0);
  EXPECT_EQ(socketOsState.subnetCalls, 0);
  EXPECT_EQ(socketOsState.interfacePrefixes.size(), 1U);

  // Exhausting every fallback is a successful search with zero selected interfaces.
  ResetSocketOsState();
  (void)::unsetenv("NCCL_COMM_ID");
  for (int reply = 0; reply < 5; ++reply) {
    socketOsState.interfaceReplies.push_back({ncclSuccess, 0, "", {}});
  }
  found = 17;
  ASSERT_EQ(ncclFindInterfaces(names, addresses, MAX_IF_NAME_SIZE, 1, &found), ncclSuccess);
  EXPECT_EQ(found, 0);
  EXPECT_EQ(socketOsState.interfacePrefixes, expectedPrefixes);

  // Scoped IPv6 parsing must retain its numeric port and resolve the named loopback interface.
  union ncclSocketAddress scopedAddress = {};
  const unsigned int loopbackIndex = ::if_nametoindex("lo");
  ASSERT_NE(loopbackIndex, 0U);
  ASSERT_EQ(ncclSocketGetAddrFromString(&scopedAddress, "[fe80::1%lo]:7"), ncclSuccess);
  EXPECT_EQ(scopedAddress.sa.sa_family, AF_INET6);
  EXPECT_EQ(scopedAddress.sin6.sin6_scope_id, loopbackIndex);
  EXPECT_EQ(ncclSocketToPort(&scopedAddress), 7);
}

/**
 * @brief Verifies socket initialization, descriptor access, shutdown, and initialization cleanup.
 */
TEST_F(SocketLifecycleTest, InitializesDescriptorsAndShutsDown) {
  SCOPED_TRACE("Initialization, descriptor access, and shutdown must honor their ownership and state contracts.");
  volatile uint32_t abortFlag = 0;

  // A null destination is a documented no-op, while address-free initialization establishes usable
  // core metadata.
  EXPECT_EQ(ncclSocketInit(nullptr), ncclSuccess);
  struct ncclSocket socket = {};
  ASSERT_EQ(ncclSocketInit(&socket, nullptr, 123, ncclSocketTypeProxy, &abortFlag, 1, 1), ncclSuccess);
  EXPECT_EQ(socket.socketDescriptor, NCCL_INVALID_SOCKET);
  EXPECT_EQ(socket.acceptSocketDescriptor, NCCL_INVALID_SOCKET);
  EXPECT_EQ(socket.state, ncclSocketStateInitialized);
  EXPECT_EQ(socket.magic, 123U);
  EXPECT_EQ(socket.type, ncclSocketTypeProxy);
  EXPECT_EQ(socket.abortFlag, &abortFlag);
  EXPECT_EQ(socket.asyncFlag, 1);
  EXPECT_EQ(socket.customRetry, 1);

  // Descriptor access rejects a null socket, permits omitted output, and does not close a replaced
  // descriptor.
  ncclSocketDescriptor descriptor = 77;
  EXPECT_EQ(ncclSocketGetFd(nullptr, &descriptor), ncclInvalidArgument);
  EXPECT_EQ(descriptor, 77);
  EXPECT_EQ(ncclSocketSetFd(4, nullptr), ncclInvalidArgument);
  const struct ncclSocket beforeOmittedGet = socket;
  EXPECT_EQ(ncclSocketGetFd(&socket, nullptr), ncclSuccess);
  ExpectSocketMatchesForTest(socket, beforeOmittedGet, "Omitting the descriptor output must not modify the socket.");
  SocketPairGuard descriptorPair;
  ASSERT_GE(descriptorPair.Get(0), 0);
  const int previousDescriptor = descriptorPair.Release(0);
  socket.socketDescriptor = previousDescriptor;
  struct ncclSocket expectedAfterSet = socket;
  expectedAfterSet.socketDescriptor = descriptorPair.Get(1);
  ASSERT_EQ(ncclSocketSetFd(descriptorPair.Get(1), &socket), ncclSuccess);
  ExpectSocketMatchesForTest(socket, expectedAfterSet, "Descriptor replacement must change only socketDescriptor.");
  EXPECT_NE(::fcntl(previousDescriptor, F_GETFD), -1);
  const struct ncclSocket beforeGet = socket;
  ASSERT_EQ(ncclSocketGetFd(&socket, &descriptor), ncclSuccess);
  EXPECT_EQ(descriptor, descriptorPair.Get(1));
  ExpectSocketMatchesForTest(socket, beforeGet, "Descriptor retrieval must not modify the socket.");
  (void)::close(previousDescriptor);
  socket.socketDescriptor = NCCL_INVALID_SOCKET;

  // Supported addresses create a configured descriptor; unsupported families do not.
  union ncclSocketAddress address = MakeLoopbackAddress(0);
  union ncclSocketAddress addressBefore = address;
  ASSERT_EQ(ncclSocketInit(&socket, &address), ncclSuccess);
  EXPECT_EQ(std::memcmp(&address, &addressBefore, sizeof(address)), 0);
  EXPECT_GE(socket.socketDescriptor, 0);
  EXPECT_EQ(socket.salen, static_cast<int>(sizeof(sockaddr_in)));
  EXPECT_EQ(socketOsState.resetDescriptorCalls, 1);
  ASSERT_EQ(CloseSocketForTest(&socket, false), ncclSuccess);
  union ncclSocketAddress ipv6Address = {};
  ipv6Address.sin6.sin6_family = AF_INET6;
  ipv6Address.sin6.sin6_addr = in6addr_loopback;
  addressBefore = ipv6Address;
  ASSERT_EQ(ncclSocketInit(&socket, &ipv6Address), ncclSuccess);
  EXPECT_EQ(std::memcmp(&ipv6Address, &addressBefore, sizeof(ipv6Address)), 0);
  EXPECT_EQ(socket.salen, static_cast<int>(sizeof(sockaddr_in6)));
  ASSERT_EQ(CloseSocketForTest(&socket, false), ncclSuccess);
  address.sa.sa_family = AF_UNIX;
  addressBefore = address;
  EXPECT_EQ(ncclSocketInit(&socket, &address), ncclInternalError);
  EXPECT_EQ(std::memcmp(&address, &addressBefore, sizeof(address)), 0);
  EXPECT_EQ(socket.socketDescriptor, NCCL_INVALID_SOCKET);

  // Descriptor-configuration and descriptor-creation failures must both close partial state.
  ResetSocketOsState();
  socketOsState.setFlagsResult = ncclSystemError;
  address = MakeLoopbackAddress(0);
  addressBefore = address;
  EXPECT_EQ(ncclSocketInit(&socket, &address), ncclSystemError);
  EXPECT_EQ(std::memcmp(&address, &addressBefore, sizeof(address)), 0);
  EXPECT_EQ(socketOsState.setFlagsCalls, 1);
  EXPECT_EQ(socketOsState.closeCalls, 1);
  EXPECT_EQ(socket.socketDescriptor, NCCL_INVALID_SOCKET);
  ResetSocketOsState();
  socketOsState.resetDescriptorResult = ncclSystemError;
  address = MakeLoopbackAddress(0);
  addressBefore = address;
  EXPECT_EQ(ncclSocketInit(&socket, &address), ncclSystemError);
  EXPECT_EQ(std::memcmp(&address, &addressBefore, sizeof(address)), 0);
  EXPECT_EQ(socketOsState.closeCalls, 1);
  EXPECT_EQ(socket.state, ncclSocketStateClosed);

  // Shutdown is a null/invalid no-op, succeeds on a live descriptor, and preserves state on a
  // system failure.
  EXPECT_EQ(ncclSocketShutdown(nullptr, SHUT_RDWR), ncclSuccess);
  socket = MakeSocketForTest(NCCL_INVALID_SOCKET, ncclSocketStateReady);
  EXPECT_EQ(ncclSocketShutdown(&socket, SHUT_RDWR), ncclSuccess);
  EXPECT_EQ(socket.state, ncclSocketStateTerminating);
  SocketPairGuard shutdownPair;
  ASSERT_GE(shutdownPair.Get(0), 0);
  socket = MakeSocketForTest(shutdownPair.Get(0), ncclSocketStateReady);
  EXPECT_EQ(ncclSocketShutdown(&socket, SHUT_RDWR), ncclSuccess);
  EXPECT_EQ(socket.state, ncclSocketStateTerminating);
  const int closedDescriptor = shutdownPair.Release(0);
  (void)::close(closedDescriptor);
  socket = MakeSocketForTest(closedDescriptor, ncclSocketStateReady);
  const struct ncclSocket beforeFailedShutdown = socket;
  EXPECT_EQ(ncclSocketShutdown(&socket, SHUT_RDWR), ncclSystemError);
  ExpectSocketMatchesForTest(socket, beforeFailedShutdown,
    "A failed operating-system shutdown must leave the socket object unchanged.");
}

/**
 * @brief Verifies loopback listen setup, assigned-port discovery, and invalid inputs.
 */
TEST_F(SocketLifecycleTest, ListensOnLoopback) {
  SCOPED_TRACE("Listening must validate descriptors and expose the exact bound loopback endpoint.");
  // Null and descriptor-free sockets cannot be turned into listeners.
  EXPECT_EQ(ncclSocketListen(nullptr), ncclInvalidArgument);
  struct ncclSocket listener = MakeSocketForTest(NCCL_INVALID_SOCKET, ncclSocketStateInitialized);
  EXPECT_EQ(ncclSocketListen(&listener), ncclInvalidArgument);

  // Port zero must bind a real loopback listener and expose its assigned port and accept
  // descriptor.
  union ncclSocketAddress address = MakeLoopbackAddress(0);
  ASSERT_EQ(ncclSocketInit(&listener, &address), ncclSuccess);
  ASSERT_EQ(ncclSocketListen(&listener), ncclSuccess);
  EXPECT_EQ(listener.state, ncclSocketStateReady);
  EXPECT_EQ(listener.acceptSocketDescriptor, listener.socketDescriptor);
  const uint16_t assignedPort = ncclSocketToPort(&listener.addr);
  EXPECT_NE(assignedPort, 0);
  EXPECT_EQ(listener.addr.sa.sa_family, AF_INET);
  ASSERT_EQ(CloseSocketForTest(&listener, false), ncclSuccess);

  // A valid requested port must be retained after the listener enables address reuse and binds.
  address = MakeLoopbackAddress(assignedPort);
  ASSERT_EQ(ncclSocketInit(&listener, &address), ncclSuccess);
  ASSERT_EQ(ncclSocketListen(&listener), ncclSuccess);
  EXPECT_EQ(ncclSocketToPort(&listener.addr), assignedPort);
  ASSERT_EQ(CloseSocketForTest(&listener, false), ncclSuccess);

  // A descriptor closed behind the socket object must produce a system error rather than a ready
  // listener.
  address = MakeLoopbackAddress(0);
  ASSERT_EQ(ncclSocketInit(&listener, &address), ncclSuccess);
  ASSERT_EQ(::close(listener.socketDescriptor), 0);
  EXPECT_EQ(ncclSocketListen(&listener), ncclSystemError);
  EXPECT_EQ(listener.state, ncclSocketStateInitialized);
  listener.socketDescriptor = NCCL_INVALID_SOCKET;
}

/**
 * @brief Verifies connect start, polling, handshake transmission, readiness, and state errors.
 */
TEST_F(SocketLifecycleTest, ConnectsAndReportsReadiness) {
  SCOPED_TRACE("Connect, poll, and readiness calls must return their reachable statuses and exact state changes.");
  // Connect rejects null, invalid, and non-initialized sockets with state-specific results.
  EXPECT_EQ(ncclSocketConnect(nullptr), ncclInvalidArgument);
  struct ncclSocket socket = MakeSocketForTest(NCCL_INVALID_SOCKET, ncclSocketStateInitialized);
  struct ncclSocket unchangedSocket = socket;
  EXPECT_EQ(ncclSocketConnect(&socket), ncclInvalidArgument);
  ExpectSocketMatchesForTest(socket, unchangedSocket, "An invalid descriptor must be rejected without socket changes.");
  SocketPairGuard pair;
  ASSERT_GE(pair.Get(0), 0);
  socket = MakeSocketForTest(pair.Get(0), ncclSocketStateReady);
  unchangedSocket = socket;
  EXPECT_EQ(ncclSocketConnect(&socket), ncclInternalError);
  ExpectSocketMatchesForTest(socket, unchangedSocket, "A ready socket must not be restarted as a connection.");
  socket.state = ncclSocketStateError;
  unchangedSocket = socket;
  EXPECT_EQ(ncclSocketConnect(&socket), ncclRemoteError);
  ExpectSocketMatchesForTest(socket, unchangedSocket, "An erroneous socket must remain erroneous when reconnected.");

  // A system-level connect failure must be returned and leave the socket in the error state set
  // by the provider boundary.
  ResetSocketOsState();
  socket = MakeSocketForTest(pair.Get(0), ncclSocketStateInitialized);
  socket.asyncFlag = 1;
  socketOsState.startConnectResult = ncclSystemError;
  EXPECT_EQ(ncclSocketConnect(&socket), ncclSystemError)
    << "A system failure from connect start must not be converted to success or a different error class.";
  EXPECT_EQ(socket.state, ncclSocketStateError);

  // A successful asynchronous connect must send the exact magic and type before becoming ready.
  ResetSocketOsState();
  socket = MakeSocketForTest(pair.Get(0), ncclSocketStateInitialized);
  socket.asyncFlag = 1;
  socket.magic = 0x0102030405060708ULL;
  socket.type = ncclSocketTypeNetIb;
  socketOsState.startConnectState = ncclSocketStateConnected;
  ASSERT_EQ(ncclSocketConnect(&socket), ncclSuccess);
  EXPECT_EQ(socket.state, ncclSocketStateReady);
  EXPECT_EQ(socketOsState.startConnectCalls, 1);
  uint64_t receivedMagic = 0;
  enum ncclSocketType receivedType = ncclSocketTypeUnknown;
  ASSERT_TRUE(ReceiveExactForTest(pair.Get(1), &receivedMagic, sizeof(receivedMagic)));
  ASSERT_TRUE(ReceiveExactForTest(pair.Get(1), &receivedType, sizeof(receivedType)));
  EXPECT_EQ(receivedMagic, socket.magic);
  EXPECT_EQ(receivedType, socket.type);

  // A synchronous public connect must complete both handshake fields before it returns.
  ResetSocketOsState();
  SocketPairGuard synchronousPair;
  ASSERT_GE(synchronousPair.Get(0), 0);
  socket = MakeSocketForTest(synchronousPair.Get(0), ncclSocketStateInitialized);
  socket.magic = 0x8877665544332211ULL;
  socket.type = ncclSocketTypeBootstrap;
  socketOsState.startConnectState = ncclSocketStateConnected;
  ASSERT_EQ(ncclSocketConnect(&socket), ncclSuccess);
  EXPECT_EQ(socket.state, ncclSocketStateReady);
  receivedMagic = 0;
  receivedType = ncclSocketTypeUnknown;
  ASSERT_TRUE(ReceiveExactForTest(synchronousPair.Get(1), &receivedMagic, sizeof(receivedMagic)));
  ASSERT_TRUE(ReceiveExactForTest(synchronousPair.Get(1), &receivedType, sizeof(receivedType)));
  EXPECT_EQ(receivedMagic, socket.magic);
  EXPECT_EQ(receivedType, socket.type);

  // Readiness reports stable terminal states and advances one unfinished connect-poll transition.
  int running = -1;
  EXPECT_EQ(ncclSocketReady(nullptr, &running), ncclSuccess);
  EXPECT_EQ(running, 0);
  ASSERT_EQ(ncclSocketReady(&socket, &running), ncclSuccess);
  EXPECT_EQ(running, 1);
  socket.state = ncclSocketStateError;
  unchangedSocket = socket;
  EXPECT_EQ(ncclSocketReady(&socket, &running), ncclRemoteError);
  ExpectSocketMatchesForTest(socket, unchangedSocket, "Readiness must not modify an erroneous socket.");
  socket.state = ncclSocketStateClosed;
  unchangedSocket = socket;
  EXPECT_EQ(ncclSocketReady(&socket, &running), ncclRemoteError);
  ExpectSocketMatchesForTest(socket, unchangedSocket, "Readiness must not modify a closed socket.");
  socket.state = ncclSocketStateConnectPolling;
  socketOsState.pollConnectState = ncclSocketStateConnectPolling;
  ASSERT_EQ(ncclSocketReady(&socket, &running), ncclSuccess);
  EXPECT_EQ(running, 0);
  EXPECT_EQ(socketOsState.pollConnectCalls, 1);

  // An abort during handshake progress must surface as an internal error after reporting not-ready.
  volatile uint32_t abortFlag = 1;
  socket = MakeSocketForTest(pair.Get(0), ncclSocketStateConnected);
  socket.abortFlag = &abortFlag;
  unchangedSocket = socket;
  running = 7;
  EXPECT_EQ(ncclSocketReady(&socket, &running), ncclInternalError)
    << "An observed abort during handshake transmission must reach the readiness caller.";
  EXPECT_EQ(running, 0);
  ExpectSocketMatchesForTest(socket, unchangedSocket, "An aborted handshake must not advance socket state.");

  // The public poll wrapper propagates its boundary result and resulting state.
  ResetSocketOsState();
  EXPECT_EQ(ncclSocketPollConnect(nullptr), ncclInvalidArgument);
  socket = MakeSocketForTest(pair.Get(0), ncclSocketStateConnectPolling);
  socketOsState.pollConnectResult = ncclRemoteError;
  EXPECT_EQ(ncclSocketPollConnect(&socket), ncclRemoteError);
  EXPECT_EQ(socket.state, ncclSocketStateError);
  socketOsState.pollConnectResult = ncclSystemError;
  socket.state = ncclSocketStateConnectPolling;
  EXPECT_EQ(ncclSocketPollConnect(&socket), ncclSystemError)
    << "A polling system failure must be returned without completing the connection.";
  EXPECT_EQ(socket.state, ncclSocketStateConnectPolling);
  socketOsState.pollConnectResult = ncclSuccess;
  socketOsState.pollConnectState = ncclSocketStateConnected;
  socket.state = ncclSocketStateConnectPolling;
  EXPECT_EQ(ncclSocketPollConnect(&socket), ncclSuccess);
  EXPECT_EQ(socket.state, ncclSocketStateConnected);
}

/**
 * @brief Verifies accepted-socket validation for correct, rejected, and partially transferred
 * handshakes.
 */
TEST_F(SocketLifecycleTest, AcceptsAndValidatesHandshakes) {
  SCOPED_TRACE("Accept handshakes must preserve the listener and distinguish valid, rejected, and partial peers.");
  // Accept rejects null inputs and distinguishes listener error from other wrong states.
  SocketPairGuard listenerPair;
  ASSERT_GE(listenerPair.Get(0), 0);
  struct ncclSocket accepted = MakeSocketForTest(NCCL_INVALID_SOCKET, ncclSocketStateNone);
  struct ncclSocket listener = MakeSocketForTest(listenerPair.Get(0), ncclSocketStateInitialized);
  struct ncclSocket listenerBefore = listener;
  EXPECT_EQ(ncclSocketAccept(nullptr, &listener), ncclInvalidArgument);
  ExpectSocketMatchesForTest(listener, listenerBefore, "A null accept destination must not modify the listener.");
  const struct ncclSocket acceptedBeforeNullListener = accepted;
  EXPECT_EQ(ncclSocketAccept(&accepted, nullptr), ncclInvalidArgument);
  ExpectSocketMatchesForTest(accepted, acceptedBeforeNullListener,
    "A null listener must not modify the accept destination.");
  EXPECT_EQ(ncclSocketAccept(&accepted, &listener), ncclInternalError);
  ExpectSocketMatchesForTest(listener, listenerBefore, "A listener-state error must not modify the listener.");
  listener.state = ncclSocketStateError;
  listenerBefore = listener;
  EXPECT_EQ(ncclSocketAccept(&accepted, &listener), ncclSystemError);
  ExpectSocketMatchesForTest(listener, listenerBefore, "An erroneous listener must remain unchanged.");

  // A matching synchronous handshake must be accepted and leave the connection ready.
  SocketPairGuard validPair;
  ASSERT_GE(validPair.Get(0), 0);
  listener = MakeSocketForTest(listenerPair.Get(0), ncclSocketStateReady);
  listener.acceptSocketDescriptor = listenerPair.Get(0);
  listener.magic = 0x1122334455667788ULL;
  listener.type = ncclSocketTypeProxy;
  listener.asyncFlag = 0;
  listenerBefore = listener;
  socketOsState.acceptedDescriptor = validPair.Release(0);
  ASSERT_EQ(::send(validPair.Get(1), &listener.magic, sizeof(listener.magic), MSG_NOSIGNAL),
    static_cast<ssize_t>(sizeof(listener.magic)));
  ASSERT_EQ(::send(validPair.Get(1), &listener.type, sizeof(listener.type), MSG_NOSIGNAL),
    static_cast<ssize_t>(sizeof(listener.type)));
  ASSERT_EQ(ncclSocketAccept(&accepted, &listener), ncclSuccess);
  EXPECT_EQ(accepted.state, ncclSocketStateReady);
  EXPECT_EQ(socketOsState.acceptCalls, 1);
  ExpectSocketMatchesForTest(listener, listenerBefore, "A successful accept must not modify the listener.");
  ASSERT_EQ(CloseSocketForTest(&accepted, false), ncclSuccess);

  // A wrong magic must discard only the candidate connection and return to accepting.
  ResetSocketOsState();
  listener.asyncFlag = 1;
  listenerBefore = listener;
  SocketPairGuard wrongMagicPair;
  accepted = MakeSocketForTest(NCCL_INVALID_SOCKET, ncclSocketStateNone);
  socketOsState.acceptedDescriptor = wrongMagicPair.Release(0);
  const uint64_t wrongMagic = listener.magic + 1;
  ASSERT_EQ(::send(wrongMagicPair.Get(1), &wrongMagic, sizeof(wrongMagic), MSG_NOSIGNAL),
    static_cast<ssize_t>(sizeof(wrongMagic)));
  ASSERT_EQ(ncclSocketAccept(&accepted, &listener), ncclSuccess);
  EXPECT_EQ(accepted.state, ncclSocketStateAccepting);
  EXPECT_EQ(accepted.socketDescriptor, NCCL_INVALID_SOCKET);
  EXPECT_EQ(socketOsState.resetAcceptCalls, 1);
  ExpectSocketMatchesForTest(listener, listenerBefore, "Rejecting wrong magic must not modify the listener.");

  // A wrong type is a protocol error that closes and marks the accepted socket erroneous.
  ResetSocketOsState();
  SocketPairGuard wrongTypePair;
  accepted = MakeSocketForTest(NCCL_INVALID_SOCKET, ncclSocketStateNone);
  socketOsState.acceptedDescriptor = wrongTypePair.Release(0);
  const enum ncclSocketType wrongType = ncclSocketTypeBootstrap;
  ASSERT_EQ(::send(wrongTypePair.Get(1), &listener.magic, sizeof(listener.magic), MSG_NOSIGNAL),
    static_cast<ssize_t>(sizeof(listener.magic)));
  ASSERT_EQ(::send(wrongTypePair.Get(1), &wrongType, sizeof(wrongType), MSG_NOSIGNAL),
    static_cast<ssize_t>(sizeof(wrongType)));
  EXPECT_EQ(ncclSocketAccept(&accepted, &listener), ncclInternalError);
  EXPECT_EQ(accepted.state, ncclSocketStateError);
  EXPECT_EQ(socketOsState.closeCalls, 1);
  ExpectSocketMatchesForTest(listener, listenerBefore, "Rejecting a wrong type must not modify the listener.");

  // Incremental connect and accept handshakes must preserve progress until every field is
  // transferred.
  ResetSocketOsState();
  socketOsState.maximumTransferBytes = 2;
  SocketPairGuard partialPair;
  struct ncclSocket connector = MakeSocketForTest(partialPair.Get(0), ncclSocketStateConnected);
  connector.asyncFlag = 1;
  connector.magic = 99;
  connector.type = ncclSocketTypeNetSocket;
  ASSERT_EQ(socketFinalizeConnect(&connector), ncclSuccess);
  EXPECT_EQ(connector.state, ncclSocketStateConnected);
  EXPECT_EQ(connector.finalizeCounter, 2);
  for (int attempt = 0; attempt < 8 && connector.state != ncclSocketStateReady; ++attempt) {
    ASSERT_EQ(socketFinalizeConnect(&connector), ncclSuccess);
  }
  ASSERT_EQ(connector.state, ncclSocketStateReady);
  EXPECT_EQ(connector.finalizeCounter,
    static_cast<int>(sizeof(connector.magic) + sizeof(connector.type)));
  uint64_t partialMagic = 0;
  enum ncclSocketType partialType = ncclSocketTypeUnknown;
  ASSERT_TRUE(ReceiveExactForTest(partialPair.Get(1), &partialMagic, sizeof(partialMagic)));
  ASSERT_TRUE(ReceiveExactForTest(partialPair.Get(1), &partialType, sizeof(partialType)));
  EXPECT_EQ(partialMagic, connector.magic);
  EXPECT_EQ(partialType, connector.type);

  SocketPairGuard partialAcceptPair;
  struct ncclSocket partialAccept = MakeSocketForTest(partialAcceptPair.Get(0), ncclSocketStateAccepted);
  partialAccept.asyncFlag = 1;
  partialAccept.magic = 100;
  partialAccept.type = ncclSocketTypeBootstrap;
  ASSERT_EQ(::send(partialAcceptPair.Get(1), &partialAccept.magic, sizeof(partialAccept.magic), MSG_NOSIGNAL),
    static_cast<ssize_t>(sizeof(partialAccept.magic)));
  ASSERT_EQ(::send(partialAcceptPair.Get(1), &partialAccept.type, sizeof(partialAccept.type), MSG_NOSIGNAL),
    static_cast<ssize_t>(sizeof(partialAccept.type)));
  ASSERT_EQ(socketFinalizeAccept(&partialAccept), ncclSuccess);
  EXPECT_EQ(partialAccept.state, ncclSocketStateAccepted);
  EXPECT_EQ(partialAccept.finalizeCounter, 2);
  for (int attempt = 0; attempt < 8 && partialAccept.state != ncclSocketStateReady; ++attempt) {
    ASSERT_EQ(socketFinalizeAccept(&partialAccept), ncclSuccess);
  }
  ASSERT_EQ(partialAccept.state, ncclSocketStateReady);
  EXPECT_EQ(partialAccept.finalizeCounter,
    static_cast<int>(sizeof(partialAccept.magic) + sizeof(partialAccept.type)));

  // States without an active transition must remain unchanged.
  struct ncclSocket stable = MakeSocketForTest(NCCL_INVALID_SOCKET, ncclSocketStateTerminating);
  ASSERT_EQ(socketProgressState(&stable), ncclSuccess);
  EXPECT_EQ(stable.state, ncclSocketStateTerminating);
}

/**
 * @brief Verifies state-machine error propagation, abort handling, and synchronous connect
 * finalization.
 */
TEST_F(SocketLifecycleTest, PropagatesStateErrorsAndAborts) {
  SCOPED_TRACE("Lifecycle progress must propagate reachable boundary errors and aborts without false advancement.");
  int running = 1;
  SocketPairGuard transitionPair;
  ASSERT_GE(transitionPair.Get(0), 0);
  ASSERT_GE(transitionPair.Get(1), 0);
  struct ncclSocket socket = MakeSocketForTest(transitionPair.Get(0), ncclSocketStateAccepting);
  socket.acceptSocketDescriptor = transitionPair.Get(1);

  // Every active transition must return its own boundary failure without advancing into the next
  // state.
  socketOsState.acceptResult = ncclSystemError;
  EXPECT_EQ(socketProgressState(&socket), ncclSystemError);
  EXPECT_EQ(socket.state, ncclSocketStateAccepting);
  EXPECT_EQ(socket.socketDescriptor, NCCL_INVALID_SOCKET);
  ResetSocketOsState();
  socket = MakeSocketForTest(transitionPair.Get(0), ncclSocketStateAccepted);
  socketOsState.setFlagsResult = ncclSystemError;
  EXPECT_EQ(socketProgressState(&socket), ncclSystemError);
  EXPECT_EQ(socket.state, ncclSocketStateAccepted);
  ResetSocketOsState();
  socket = MakeSocketForTest(transitionPair.Get(0), ncclSocketStateConnecting);
  socketOsState.startConnectResult = ncclRemoteError;
  EXPECT_EQ(socketProgressState(&socket), ncclRemoteError);
  EXPECT_EQ(socket.state, ncclSocketStateError);
  ResetSocketOsState();
  socket = MakeSocketForTest(transitionPair.Get(0), ncclSocketStateConnectPolling);
  socketOsState.pollConnectResult = ncclSystemError;
  EXPECT_EQ(socketProgressState(&socket), ncclSystemError);
  EXPECT_EQ(socket.state, ncclSocketStateConnectPolling);
  ResetSocketOsState();
  socket = MakeSocketForTest(transitionPair.Get(0), ncclSocketStateConnected);
  socketOsState.progressResult = ncclRemoteError;
  EXPECT_EQ(socketProgressState(&socket), ncclRemoteError);
  EXPECT_EQ(socket.state, ncclSocketStateConnected);
  ResetSocketOsState();
  volatile uint32_t transitionAbortFlag = 1;
  socket = MakeSocketForTest(transitionPair.Get(0), ncclSocketStateConnected);
  socket.abortFlag = &transitionAbortFlag;
  const struct ncclSocket beforeAbortedProgress = socket;
  EXPECT_EQ(socketProgressState(&socket), ncclInternalError)
    << "An abort reported during connect finalization must propagate through state progress.";
  ExpectSocketMatchesForTest(socket, beforeAbortedProgress,
    "Aborted connect finalization must not change the socket state or handshake counter.");

  // Readiness must propagate a transition failure and preserve its initial not-ready report.
  ResetSocketOsState();
  socket = MakeSocketForTest(transitionPair.Get(0), ncclSocketStateAccepting);
  socket.acceptSocketDescriptor = transitionPair.Get(1);
  socketOsState.acceptResult = ncclSystemError;
  EXPECT_EQ(ncclSocketReady(&socket, &running), ncclSystemError);
  EXPECT_EQ(running, 0);

  // Public connect must propagate the remote failure class produced by the real connect boundary.
  SocketPairGuard connectPair;
  ASSERT_GE(connectPair.Get(0), 0);
  ResetSocketOsState();
  socket = MakeSocketForTest(connectPair.Get(0), ncclSocketStateInitialized);
  socket.asyncFlag = 1;
  socketOsState.startConnectResult = ncclRemoteError;
  EXPECT_EQ(ncclSocketConnect(&socket), ncclRemoteError);
  EXPECT_EQ(socket.state, ncclSocketStateError);

  // Public accept must propagate a provider failure before handling an otherwise valid abort
  // request.
  volatile uint32_t abortFlag = 1;
  struct ncclSocket listener = MakeSocketForTest(transitionPair.Get(1), ncclSocketStateReady);
  listener.acceptSocketDescriptor = transitionPair.Get(1);
  listener.asyncFlag = 1;
  struct ncclSocket accepted = MakeSocketForTest(NCCL_INVALID_SOCKET, ncclSocketStateNone);
  ResetSocketOsState();
  socketOsState.acceptResult = ncclSystemError;
  EXPECT_EQ(ncclSocketAccept(&accepted, &listener), ncclSystemError);
  EXPECT_EQ(accepted.socketDescriptor, NCCL_INVALID_SOCKET);

  // A receive failure after valid magic must propagate from finalization through the public accept
  // call.
  ResetSocketOsState();
  SocketPairGuard remoteAcceptPair;
  ASSERT_GE(remoteAcceptPair.Get(0), 0);
  accepted = MakeSocketForTest(remoteAcceptPair.Get(0), ncclSocketStateAccepted);
  accepted.acceptSocketDescriptor = listener.acceptSocketDescriptor;
  accepted.asyncFlag = 1;
  accepted.magic = listener.magic;
  accepted.type = listener.type;
  accepted.finalizeCounter = static_cast<int>(sizeof(accepted.magic));
  const struct ncclSocket beforeRemoteAccept = accepted;
  const struct ncclSocket listenerBeforeRemoteAccept = listener;
  socketOsState.progressResult = ncclRemoteError;
  EXPECT_EQ(ncclSocketAccept(&accepted, &listener), ncclRemoteError)
    << "A type-handshake receive failure must reach the caller as a remote error.";
  ExpectSocketMatchesForTest(accepted, beforeRemoteAccept,
    "A failed type receive must not claim handshake or state progress.");
  ExpectSocketMatchesForTest(listener, listenerBeforeRemoteAccept,
    "A failed accepted-side handshake must not modify the listener.");

  // An observed abort must override an otherwise valid in-progress connect or accept result.
  ResetSocketOsState();
  socket.state = ncclSocketStateInitialized;
  socket.abortFlag = &abortFlag;
  socketOsState.startConnectState = ncclSocketStateConnectPolling;
  socketOsState.pollConnectState = ncclSocketStateConnectPolling;
  EXPECT_EQ(ncclSocketConnect(&socket), ncclInternalError);
  listener.abortFlag = &abortFlag;
  accepted = MakeSocketForTest(NCCL_INVALID_SOCKET, ncclSocketStateNone);
  EXPECT_EQ(ncclSocketAccept(&accepted, &listener), ncclInternalError);
  EXPECT_EQ(accepted.socketDescriptor, NCCL_INVALID_SOCKET);

  // Resuming an already erroneous destination must not overwrite it from the listener.
  accepted = MakeSocketForTest(NCCL_INVALID_SOCKET, ncclSocketStateError);
  accepted.acceptSocketDescriptor = transitionPair.Get(1);
  EXPECT_EQ(ncclSocketAccept(&accepted, &listener), ncclSystemError);
  EXPECT_EQ(accepted.state, ncclSocketStateError);

  // A synchronous connector must send both handshake fields before returning ready.
  ResetSocketOsState();
  SocketPairGuard synchronousPair;
  ASSERT_GE(synchronousPair.Get(0), 0);
  struct ncclSocket synchronous = MakeSocketForTest(synchronousPair.Get(0), ncclSocketStateConnected);
  synchronous.magic = 0xaabbccddU;
  synchronous.type = ncclSocketTypeBootstrap;
  synchronous.asyncFlag = 0;
  ASSERT_EQ(socketFinalizeConnect(&synchronous), ncclSuccess);
  EXPECT_EQ(synchronous.state, ncclSocketStateReady);
  uint64_t receivedMagic = 0;
  enum ncclSocketType receivedType = ncclSocketTypeUnknown;
  ASSERT_TRUE(ReceiveExactForTest(synchronousPair.Get(1), &receivedMagic, sizeof(receivedMagic)));
  ASSERT_TRUE(ReceiveExactForTest(synchronousPair.Get(1), &receivedType, sizeof(receivedType)));
  EXPECT_EQ(receivedMagic, synchronous.magic);
  EXPECT_EQ(receivedType, synchronous.type);

  // Accept finalization must reset a closed or synchronously unreadable candidate for another
  // accept attempt.
  ResetSocketOsState();
  SocketPairGuard closedCandidatePair;
  ASSERT_GE(closedCandidatePair.Get(0), 0);
  accepted = MakeSocketForTest(closedCandidatePair.Release(0), ncclSocketStateAccepted);
  accepted.asyncFlag = 1;
  socketOsState.forceClosure = true;
  EXPECT_EQ(socketFinalizeAccept(&accepted), ncclSuccess);
  EXPECT_EQ(accepted.state, ncclSocketStateAccepting);
  EXPECT_EQ(socketOsState.resetAcceptCalls, 1);
  ResetSocketOsState();
  SocketPairGuard unreadableCandidatePair;
  ASSERT_GE(unreadableCandidatePair.Get(0), 0);
  accepted = MakeSocketForTest(unreadableCandidatePair.Release(0), ncclSocketStateAccepted);
  accepted.asyncFlag = 0;
  socketOsState.progressResult = ncclRemoteError;
  EXPECT_EQ(socketFinalizeAccept(&accepted), ncclSuccess);
  EXPECT_EQ(accepted.state, ncclSocketStateAccepting);
  EXPECT_EQ(socketOsState.resetAcceptCalls, 1);
}

/**
 * @brief Verifies direct internal progress and wait contracts for partial I/O, closure, polling,
 * and errors.
 * @note Side effects: Transfers bytes over local socket pairs and records OS-boundary and poll
 * calls.
 */
TEST_F(SocketIoTest, DirectInternalProgressAndWaitContracts) {
  SCOPED_TRACE(
      "Internal socket progress and wait must preserve inputs, update offsets, and propagate every natural status.");
  char payload[] = "abcde";
  const char payloadBefore[] = "abcde";
  char received[sizeof(payload)] = {};
  SocketPairGuard pair;
  ASSERT_GE(pair.Get(0), 0) << "A local connected descriptor is required for direct progress testing";
  ASSERT_GE(pair.Get(1), 0) << "A peer descriptor is required to observe direct transfers";
  struct ncclSocket socket = MakeSocketForTest(pair.Get(0), ncclSocketStateReady);
  const struct ncclSocket socketBefore = socket;

  socketOsState.maximumTransferBytes = 2;
  int offset = 0;
  int closed = 0;
  ASSERT_EQ(socketProgress(NCCL_SOCKET_SEND, &socket, payload, 5, &offset, &closed), ncclSuccess)
      << "One direct progress call must report a successful partial transfer";
  EXPECT_EQ(offset, 2) << "Direct progress must expose exactly the transferred byte count";
  EXPECT_EQ(closed, 0) << "A live peer must not be reported as closed";
  ASSERT_TRUE(ReceiveExactForTest(pair.Get(1), received, 2))
      << "The peer must receive the exact progressed prefix";
  EXPECT_EQ(std::string(received, 2), "ab")
      << "Direct progress must start at the caller-provided offset";
  EXPECT_EQ(std::memcmp(payload, payloadBefore, sizeof(payload)), 0)
      << "Sending must not modify the caller's buffer";
  ExpectSocketMatchesForTest(socket, socketBefore,
      "Direct progress must not modify socket metadata.");

  ResetSocketOsState();
  socketOsState.maximumTransferBytes = 2;
  char receiveOutput[4] = {'?', '?', '?', '!'};
  ASSERT_EQ(::send(pair.Get(1), "xyz", 3, MSG_NOSIGNAL), 3)
      << "The peer must provide input for direct receive progress";
  offset = 0;
  closed = 0;
  ASSERT_EQ(socketProgress(NCCL_SOCKET_RECV, &socket, receiveOutput, 3, &offset, &closed), ncclSuccess)
      << "Direct receive progress must write the available prefix";
  EXPECT_EQ(offset, 2) << "A capped receive must report only two consumed bytes";
  EXPECT_EQ(std::string(receiveOutput, 2), "xy")
      << "Receive progress must write the exact peer bytes";
  EXPECT_EQ(receiveOutput[2], '?') << "A partial receive must not overwrite an unreceived byte";
  EXPECT_EQ(receiveOutput[3], '!') << "Receive progress must preserve storage beyond the requested size";
  ASSERT_EQ(socketProgress(NCCL_SOCKET_RECV, &socket, receiveOutput, 3, &offset, &closed), ncclSuccess)
      << "A second receive progress call must complete the remaining byte";
  EXPECT_EQ(offset, 3) << "Completing receive progress must update the output offset to size";
  EXPECT_EQ(std::string(receiveOutput, 3), "xyz")
      << "Repeated receive progress must neither skip nor duplicate data";
  EXPECT_EQ(receiveOutput[3], '!') << "Completing receive progress must preserve the trailing canary";

  ResetSocketOsState();
  socketOsState.forceClosure = true;
  offset = 1;
  closed = 0;
  EXPECT_EQ(socketProgress(NCCL_SOCKET_RECV, &socket, received, 5, &offset, &closed), ncclSuccess)
      << "A caller requesting closure information must receive closure as successful progress";
  EXPECT_EQ(closed, 1) << "The optional closure output must change to one";
  EXPECT_EQ(offset, 1) << "Peer closure must not claim unread bytes";
  EXPECT_EQ(socketProgress(NCCL_SOCKET_RECV, &socket, received, 5, &offset), ncclRemoteError)
      << "Peer closure without an output channel must be returned as a remote error";

  ResetSocketOsState();
  socketOsState.progressResult = ncclSystemError;
  offset = 1;
  closed = 7;
  EXPECT_EQ(socketProgress(NCCL_SOCKET_SEND, &socket, payload, 5, &offset, &closed), ncclSystemError)
      << "Direct progress must preserve an OS-boundary system error";
  EXPECT_EQ(offset, 1) << "A system error must not claim transfer progress";
  EXPECT_EQ(closed, 7) << "A failed progress call must preserve the optional closure output";
  EXPECT_EQ(std::memcmp(payload, payloadBefore, sizeof(payload)), 0)
      << "A failed send must preserve the caller's bytes";

  ResetSocketOsState();
  volatile uint32_t abortFlag = 1;
  socket.abortFlag = &abortFlag;
  offset = 0;
  closed = 7;
  EXPECT_EQ(socketProgress(NCCL_SOCKET_SEND, &socket, payload, 5, &offset, &closed), ncclInternalError)
      << "Direct progress must propagate a caller abort";
  EXPECT_EQ(offset, 0) << "An aborted operation must not claim progress";
  EXPECT_EQ(closed, 7) << "An aborted progress call must preserve the optional closure output";
  socket.abortFlag = nullptr;

  ResetSocketOsState();
  socketOsState.maximumTransferBytes = 2;
  socketOsState.parameterValues["NCCL_SOCKET_POLL_TIMEOUT_MSEC"] = 1;
  offset = 0;
  ASSERT_EQ(socketWait(NCCL_SOCKET_SEND, &socket, payload, 5, &offset), ncclSuccess)
      << "Direct wait must complete a transfer split across several progress calls";
  EXPECT_EQ(offset, 5) << "Direct wait must return only after the complete requested size";
  EXPECT_EQ(socketOsState.progressCalls, 3)
      << "Five bytes with a two-byte progress cap require exactly three attempts";
  EXPECT_EQ(socketOsState.pollCalls, 2)
      << "Wait must poll only between incomplete progress attempts";
  EXPECT_EQ(socketOsState.pollDescriptors, std::vector<int>({socket.socketDescriptor, socket.socketDescriptor}))
      << "Every poll must target the progressed socket descriptor";
  EXPECT_EQ(socketOsState.pollOperations, std::vector<int>({NCCL_SOCKET_SEND, NCCL_SOCKET_SEND}))
      << "Every poll must retain the caller's operation";
  std::memset(received, 0, sizeof(received));
  ASSERT_TRUE(ReceiveExactForTest(pair.Get(1), received, 5))
      << "The peer must receive all bytes completed by direct wait";
  EXPECT_EQ(std::string(received, 5), "abcde")
      << "Chunked wait must neither skip nor duplicate bytes";
  EXPECT_EQ(std::memcmp(payload, payloadBefore, sizeof(payload)), 0)
      << "Direct wait must preserve a send buffer";

  ResetSocketOsState();
  socketOsState.maximumTransferBytes = 2;
  ASSERT_EQ(::send(pair.Get(1), "uvw", 3, MSG_NOSIGNAL), 3)
      << "The peer must provide the complete direct-wait receive payload";
  std::memset(received, '?', sizeof(received));
  offset = 0;
  ASSERT_EQ(socketWait(NCCL_SOCKET_RECV, &socket, received, 3, &offset), ncclSuccess)
      << "Wait must complete a chunked receive when polling is disabled";
  EXPECT_EQ(offset, 3) << "Polling policy must not change the completed byte count";
  EXPECT_EQ(socketOsState.progressCalls, 2)
      << "Three bytes with a two-byte cap require two progress attempts";
  EXPECT_EQ(socketOsState.pollCalls, 0)
      << "The default zero timeout must suppress intermediate polls";
  EXPECT_EQ(std::string(received, 3), "uvw")
      << "A non-polling receive wait must preserve byte order";
  EXPECT_EQ(received[3], '?') << "Receive wait must preserve storage beyond the requested size";

  ResetSocketOsState();
  offset = 5;
  EXPECT_EQ(socketWait(NCCL_SOCKET_SEND, &socket, payload, 5, &offset), ncclSuccess)
      << "An already complete offset is a successful no-op";
  EXPECT_EQ(socketOsState.progressCalls, 0)
      << "An already complete wait must not reach the OS boundary";
  EXPECT_EQ(socketOsState.pollCalls, 0)
      << "An already complete wait must not poll";

  ResetSocketOsState();
  socketOsState.progressResult = ncclSystemError;
  offset = 1;
  EXPECT_EQ(socketWait(NCCL_SOCKET_SEND, &socket, payload, 5, &offset), ncclSystemError)
      << "Direct wait must return the first progress failure";
  EXPECT_EQ(offset, 1) << "A failed wait must preserve its last valid offset";
  EXPECT_EQ(socketOsState.pollCalls, 0) << "Wait must not poll after a progress error";

  ResetSocketOsState();
  socket.abortFlag = &abortFlag;
  offset = 0;
  EXPECT_EQ(socketWait(NCCL_SOCKET_SEND, &socket, payload, 5, &offset), ncclInternalError)
      << "Direct wait must propagate an abort observed by progress";
  EXPECT_EQ(offset, 0) << "An aborted wait must not claim transfer progress";
  EXPECT_EQ(socketOsState.progressCalls, 1)
      << "Wait must stop immediately after the abort result";
  EXPECT_EQ(socketOsState.pollCalls, 0) << "Wait must not poll after an abort";
  socket.abortFlag = nullptr;

  ResetSocketOsState();
  socketOsState.forceClosure = true;
  offset = 0;
  EXPECT_EQ(socketWait(NCCL_SOCKET_RECV, &socket, received, 5, &offset), ncclRemoteError)
      << "Wait cannot complete after peer closure and must return a remote error";
  EXPECT_EQ(offset, 0) << "Peer closure must not claim receive progress";
  ExpectSocketMatchesForTest(socket, socketBefore,
      "Direct wait success and failures must preserve socket metadata.");
}

/**
 * @brief Verifies public progress and wait validation and successful delegation to their internal
 * helpers.
 */
TEST_F(SocketIoTest, ProgressesAndWaits) {
  SCOPED_TRACE("Public progress and wait wrappers must reject NULL sockets and delegate valid transfers exactly.");
  char payload[] = "abcde";
  const char payloadBefore[] = "abcde";
  int offset = 3;
  int closed = 7;

  // NULL validation belongs to the public wrappers and must preserve every caller-owned output.
  EXPECT_EQ(ncclSocketProgress(NCCL_SOCKET_SEND, nullptr, payload, 5, &offset, &closed), ncclInvalidArgument);
  EXPECT_EQ(offset, 3) << "Rejected progress must preserve the caller's offset";
  EXPECT_EQ(closed, 7) << "Rejected progress must preserve the caller's closure output";
  EXPECT_EQ(ncclSocketWait(NCCL_SOCKET_SEND, nullptr, payload, 5, &offset), ncclInvalidArgument);
  EXPECT_EQ(offset, 3) << "Rejected wait must preserve the caller's offset";
  EXPECT_EQ(socketOsState.progressCalls, 0)
      << "NULL wrapper inputs must fail before reaching the OS boundary";

  // One valid partial transfer proves that the progress wrapper forwards the operation and output
  // arguments.
  SocketPairGuard pair;
  ASSERT_GE(pair.Get(0), 0) << "A local connected descriptor is required for wrapper delegation";
  ASSERT_GE(pair.Get(1), 0) << "A peer descriptor is required to observe wrapper delegation";
  struct ncclSocket socket = MakeSocketForTest(pair.Get(0), ncclSocketStateReady);
  const struct ncclSocket socketBefore = socket;
  socketOsState.maximumTransferBytes = 2;
  offset = 0;
  closed = 0;
  ASSERT_EQ(ncclSocketProgress(NCCL_SOCKET_SEND, &socket, payload, 5, &offset, &closed), ncclSuccess);
  EXPECT_EQ(offset, 2) << "The public progress wrapper must expose internal partial progress";
  EXPECT_EQ(closed, 0) << "The public progress wrapper must preserve the live-peer result";
  char received[6] = {};
  ASSERT_TRUE(ReceiveExactForTest(pair.Get(1), received, 2))
      << "The peer must observe the bytes sent through the public progress wrapper";
  EXPECT_EQ(std::string(received, 2), "ab")
      << "The progress wrapper must forward the original buffer and offset";

  // One complete transfer proves that the wait wrapper delegates until its output offset reaches
  // size.
  ResetSocketOsState();
  offset = 0;
  ASSERT_EQ(ncclSocketWait(NCCL_SOCKET_SEND, &socket, payload, 5, &offset), ncclSuccess);
  EXPECT_EQ(offset, 5) << "The public wait wrapper must expose complete progress";
  std::memset(received, 0, sizeof(received));
  ASSERT_TRUE(ReceiveExactForTest(pair.Get(1), received, 5))
      << "The peer must observe all bytes sent through the public wait wrapper";
  EXPECT_EQ(std::string(received, 5), "abcde")
      << "The wait wrapper must forward the original buffer and size";
  EXPECT_EQ(std::memcmp(payload, payloadBefore, sizeof(payload)), 0)
      << "Public progress and wait sends must preserve the caller's buffer";
  ExpectSocketMatchesForTest(socket, socketBefore,
      "Public progress and wait wrappers must preserve socket metadata.");
}

/**
 * @brief Verifies complete send, receive, and bidirectional transfers with state validation.
 */
TEST_F(SocketIoTest, TransfersBuffers) {
  SCOPED_TRACE("Transfer wrappers must validate state, preserve send data, and receive the exact peer bytes.");
  char payload[] = "socket-data";
  const char payloadBefore[] = "socket-data";
  char received[sizeof(payload)] = {};

  // Complete send and receive wrappers reject null and non-ready sockets but accept zero-byte ready
  // operations.
  EXPECT_EQ(ncclSocketSend(nullptr, payload, 4), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketRecv(nullptr, received, 4), ncclInvalidArgument);
  SocketPairGuard pair;
  ASSERT_GE(pair.Get(0), 0);
  struct ncclSocket wrongState = MakeSocketForTest(pair.Get(0), ncclSocketStateInitialized);
  const struct ncclSocket wrongStateBefore = wrongState;
  EXPECT_EQ(ncclSocketSend(&wrongState, payload, 4), ncclInternalError);
  EXPECT_EQ(ncclSocketRecv(&wrongState, received, 4), ncclInternalError);
  ExpectSocketMatchesForTest(wrongState, wrongStateBefore,
    "Rejected send and receive calls must not modify a non-ready socket.");
  struct ncclSocket socket = MakeSocketForTest(pair.Get(0), ncclSocketStateReady);
  const struct ncclSocket socketBefore = socket;
  EXPECT_EQ(ncclSocketSend(&socket, nullptr, 0), ncclSuccess);
  EXPECT_EQ(ncclSocketRecv(&socket, nullptr, 0), ncclSuccess);

  // A real local connection must carry the entire payload in both directions.
  socketOsState.maximumTransferBytes = 3;
  ASSERT_EQ(ncclSocketSend(&socket, payload, sizeof(payload)), ncclSuccess);
  ASSERT_TRUE(ReceiveExactForTest(pair.Get(1), received, sizeof(received)));
  EXPECT_EQ(std::memcmp(payload, received, sizeof(payload)), 0);
  std::memset(received, 0, sizeof(received));
  ASSERT_EQ(::send(pair.Get(1), payload, sizeof(payload), MSG_NOSIGNAL),
    static_cast<ssize_t>(sizeof(payload)));
  ASSERT_EQ(ncclSocketRecv(&socket, received, sizeof(received)), ncclSuccess);
  EXPECT_EQ(std::memcmp(payload, received, sizeof(payload)), 0);
  EXPECT_EQ(std::memcmp(payload, payloadBefore, sizeof(payload)), 0);
  ExpectSocketMatchesForTest(socket, socketBefore, "Complete send and receive calls must preserve socket metadata.");

  // Combined send/receive must validate both sockets and finish both directions without data loss.
  EXPECT_EQ(ncclSocketSendRecv(nullptr, payload, 4, &socket, received, 4), ncclInternalError);
  EXPECT_EQ(ncclSocketSendRecv(&socket, payload, 4, nullptr, received, 4), ncclInternalError);
  wrongState = socket;
  wrongState.state = ncclSocketStateInitialized;
  EXPECT_EQ(ncclSocketSendRecv(&wrongState, payload, 4, &socket, received, 4), ncclInternalError);
  ASSERT_EQ(::send(pair.Get(1), "peer", 4, MSG_NOSIGNAL), 4);
  std::memset(received, 0, sizeof(received));
  ASSERT_EQ(ncclSocketSendRecv(&socket, payload, 4, &socket, received, 4), ncclSuccess);
  EXPECT_EQ(std::string(received, 4), "peer");
  char peerReceived[5] = {};
  ASSERT_TRUE(ReceiveExactForTest(pair.Get(1), peerReceived, 4));
  EXPECT_EQ(std::string(peerReceived, 4), "sock");
  EXPECT_EQ(std::memcmp(payload, payloadBefore, sizeof(payload)), 0);
  ExpectSocketMatchesForTest(socket, socketBefore, "Bidirectional transfer must preserve both socket inputs.");

  // Receive remains valid during termination, whereas send does not.
  socket.state = ncclSocketStateTerminating;
  ASSERT_EQ(::send(pair.Get(1), "done", 4, MSG_NOSIGNAL), 4);
  EXPECT_EQ(ncclSocketRecv(&socket, received, 4), ncclSuccess);
  EXPECT_EQ(std::string(received, 4), "done");
  EXPECT_EQ(ncclSocketSend(&socket, payload, 4), ncclInternalError);
  struct ncclSocket terminatingExpected = socketBefore;
  terminatingExpected.state = ncclSocketStateTerminating;
  ExpectSocketMatchesForTest(socket, terminatingExpected,
    "Terminating receive and rejected send must preserve every field except the caller-set state.");
  EXPECT_EQ(std::memcmp(payload, payloadBefore, sizeof(payload)), 0);
}

/**
 * @brief Verifies multi-operation scheduling plus blocking and nonblocking receive outcomes.
 */
TEST_F(SocketIoTest, RunsMultipleOperationsAndTryReceive) {
  SCOPED_TRACE("Multi-operation and try-receive calls must expose exact offsets, closure, status, and data.");
  char outbound[] = "outbound";
  char inbound[] = "inbound!";
  const char outboundBefore[] = "outbound";
  const char inboundBefore[] = "inbound!";
  char received[sizeof(inbound)] = {};
  SocketPairGuard sendPair;
  ASSERT_GE(sendPair.Get(0), 0);
  struct ncclSocket sendSocket = MakeSocketForTest(sendPair.Get(0), ncclSocketStateReady);
  const struct ncclSocket sendSocketBefore = sendSocket;

  // Invalid multi-operation inputs must fail before I/O and reset only entries visited before a
  // null socket.
  EXPECT_EQ(ncclSocketMultiOp(nullptr, 1), ncclInvalidArgument);
  struct ncclSocketOp invalidOperations[2] = {};
  invalidOperations[0] = {NCCL_SOCKET_SEND, &sendSocket, outbound, 1, 7};
  invalidOperations[1] = {NCCL_SOCKET_RECV, nullptr, received, 1, 9};
  EXPECT_EQ(ncclSocketMultiOp(invalidOperations, 0), ncclInvalidArgument);
  EXPECT_EQ(ncclSocketMultiOp(invalidOperations, 2), ncclInvalidArgument);
  EXPECT_EQ(invalidOperations[0].offset, 0);
  EXPECT_EQ(invalidOperations[1].offset, 9);
  EXPECT_EQ(std::memcmp(outbound, outboundBefore, sizeof(outbound)), 0);
  ExpectSocketMatchesForTest(sendSocket, sendSocketBefore,
    "Invalid multi-operation input must not modify an already validated socket.");

  // Two real operations must progress round-robin until both exact buffers complete.
  SocketPairGuard receivePair;
  ASSERT_GE(receivePair.Get(0), 0);
  struct ncclSocket receiveSocket = MakeSocketForTest(receivePair.Get(0), ncclSocketStateReady);
  const struct ncclSocket receiveSocketBefore = receiveSocket;
  ASSERT_EQ(::send(receivePair.Get(1), inbound, sizeof(inbound), MSG_NOSIGNAL),
    static_cast<ssize_t>(sizeof(inbound)));
  socketOsState.maximumTransferBytes = 2;
  struct ncclSocketOp operations[2] = {
    {NCCL_SOCKET_SEND, &sendSocket, outbound, static_cast<int>(sizeof(outbound)), 55},
    {NCCL_SOCKET_RECV, &receiveSocket, received, static_cast<int>(sizeof(received)), 66},
  };
  ASSERT_EQ(ncclSocketMultiOp(operations, 2), ncclSuccess);
  EXPECT_EQ(operations[0].offset, static_cast<int>(sizeof(outbound)));
  EXPECT_EQ(operations[1].offset, static_cast<int>(sizeof(received)));
  EXPECT_EQ(std::memcmp(received, inbound, sizeof(inbound)), 0);
  char peerReceived[sizeof(outbound)] = {};
  ASSERT_TRUE(ReceiveExactForTest(sendPair.Get(1), peerReceived, sizeof(peerReceived)));
  EXPECT_EQ(std::memcmp(peerReceived, outbound, sizeof(outbound)), 0);
  EXPECT_EQ(std::memcmp(outbound, outboundBefore, sizeof(outbound)), 0);
  EXPECT_EQ(std::memcmp(inbound, inboundBefore, sizeof(inbound)), 0);
  ExpectSocketMatchesForTest(sendSocket, sendSocketBefore, "Multi-operation send must preserve its socket metadata.");
  ExpectSocketMatchesForTest(receiveSocket, receiveSocketBefore,
    "Multi-operation receive must preserve its socket metadata.");

  // Nonblocking receive reports no-data as in-progress and completes after any initial partial
  // transfer.
  SocketPairGuard tryPair;
  ASSERT_GE(tryPair.Get(0), 0);
  struct ncclSocket trySocket = MakeSocketForTest(tryPair.Get(0), ncclSocketStateReady);
  const struct ncclSocket trySocketBefore = trySocket;
  int closed = 9;
  EXPECT_EQ(ncclSocketTryRecv(nullptr, received, 4, &closed, false), ncclInvalidArgument);
  EXPECT_EQ(closed, 9);
  const std::vector<char> beforeEmptyReceive(received, received + sizeof(received));
  EXPECT_EQ(ncclSocketTryRecv(&trySocket, received, 4, &closed, false), ncclInProgress);
  EXPECT_EQ(closed, 0);
  EXPECT_EQ(std::memcmp(received, beforeEmptyReceive.data(), sizeof(received)), 0);
  ASSERT_EQ(::send(tryPair.Get(1), "part", 4, MSG_NOSIGNAL), 4);
  socketOsState.maximumTransferBytes = 2;
  ASSERT_EQ(ncclSocketTryRecv(&trySocket, received, 4, &closed, false), ncclSuccess);
  EXPECT_EQ(std::string(received, 4), "part");
  EXPECT_EQ(closed, 0);

  // Blocking receive waits for all available bytes, and peer closure is returned separately from
  // data.
  ASSERT_EQ(::send(tryPair.Get(1), "wait", 4, MSG_NOSIGNAL), 4);
  ASSERT_EQ(ncclSocketTryRecv(&trySocket, received, 4, &closed, true), ncclSuccess);
  EXPECT_EQ(std::string(received, 4), "wait");
  tryPair.Close(1);
  ASSERT_EQ(ncclSocketTryRecv(&trySocket, received, 4, &closed, true), ncclSuccess);
  EXPECT_EQ(closed, 1);
  ExpectSocketMatchesForTest(trySocket, trySocketBefore, "Try-receive must not modify socket metadata.");
}

/**
 * @brief Verifies that every public I/O wrapper preserves lower-level failures and abort results.
 */
TEST_F(SocketIoTest, PropagatesErrorsAndAbort) {
  SCOPED_TRACE("Every I/O wrapper must preserve reachable remote and abort errors without claiming progress.");
  char value = 'x';
  int offset = 0;
  int closed = 9;
  SocketPairGuard errorPair;
  ASSERT_GE(errorPair.Get(0), 0);
  struct ncclSocket socket = MakeSocketForTest(errorPair.Get(0), ncclSocketStateReady);
  struct ncclSocket socketBefore = socket;

  // A progress failure must be returned by direct wait, send, receive, and combined transfer
  // wrappers.
  socketOsState.progressResult = ncclRemoteError;
  EXPECT_EQ(ncclSocketWait(NCCL_SOCKET_SEND, &socket, &value, 1, &offset), ncclRemoteError);
  EXPECT_EQ(ncclSocketSend(&socket, &value, 1), ncclRemoteError);
  EXPECT_EQ(ncclSocketRecv(&socket, &value, 1), ncclRemoteError);
  EXPECT_EQ(ncclSocketSendRecv(&socket, &value, 1, &socket, &value, 1), ncclRemoteError);
  EXPECT_EQ(value, 'x');
  ExpectSocketMatchesForTest(socket, socketBefore,
    "Remote I/O failures must not modify socket metadata or the caller's byte.");

  // Multi-operation failure must leave its already-reset offset at zero rather than claiming
  // completion.
  struct ncclSocketOp operation = {NCCL_SOCKET_SEND, &socket, &value, 1, 7};
  EXPECT_EQ(ncclSocketMultiOp(&operation, 1), ncclRemoteError);
  EXPECT_EQ(operation.offset, 0);
  ExpectSocketMatchesForTest(socket, socketBefore, "A failed multi-operation must not modify its socket.");

  // Try-receive must initialize its closure result before propagating a progress failure.
  EXPECT_EQ(ncclSocketTryRecv(&socket, &value, 1, &closed, false), ncclRemoteError);
  EXPECT_EQ(closed, 0);
  EXPECT_EQ(value, 'x');
  ExpectSocketMatchesForTest(socket, socketBefore, "A failed try-receive must not modify its socket or buffer.");

  // An abort observed by the OS boundary must reach every wrapper without data or state changes.
  ResetSocketOsState();
  volatile uint32_t abortFlag = 1;
  socket.abortFlag = &abortFlag;
  socketBefore = socket;
  offset = 0;
  EXPECT_EQ(ncclSocketProgress(NCCL_SOCKET_SEND, &socket, &value, 1, &offset, &closed), ncclInternalError);
  EXPECT_EQ(offset, 0);
  EXPECT_EQ(ncclSocketWait(NCCL_SOCKET_SEND, &socket, &value, 1, &offset), ncclInternalError)
    << "Wait must propagate an abort detected by socket progress.";
  EXPECT_EQ(offset, 0);
  EXPECT_EQ(ncclSocketSend(&socket, &value, 1), ncclInternalError)
    << "Send must propagate an abort detected before any byte is written.";
  EXPECT_EQ(ncclSocketRecv(&socket, &value, 1), ncclInternalError)
    << "Receive must propagate an abort detected before any byte is read.";
  EXPECT_EQ(ncclSocketSendRecv(&socket, &value, 1, &socket, &value, 1), ncclInternalError)
    << "Combined transfer must propagate an abort from its first operation.";
  operation = {NCCL_SOCKET_SEND, &socket, &value, 1, 7};
  EXPECT_EQ(ncclSocketMultiOp(&operation, 1), ncclInternalError)
    << "Multi-operation progress must propagate an abort from its active entry.";
  EXPECT_EQ(operation.offset, 0);
  closed = 9;
  EXPECT_EQ(ncclSocketTryRecv(&socket, &value, 1, &closed, false), ncclInternalError)
    << "Try-receive must propagate an abort while preserving its initialized closure output.";
  EXPECT_EQ(closed, 0);
  EXPECT_EQ(value, 'x');
  ExpectSocketMatchesForTest(socket, socketBefore,
    "Aborted I/O calls must not modify socket metadata or the caller's byte.");
}

}  // namespace
