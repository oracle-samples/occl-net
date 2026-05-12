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

#include "common.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr auto kPollTimeout = std::chrono::seconds(30);
constexpr auto kPollInterval = std::chrono::microseconds(100);

void TestLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {
}

bool RuntimeEnabled() {
  const char* singleNodeValue = std::getenv("OCCL_NET_IB_SINGLE_NODE_TESTS");
  if (singleNodeValue != nullptr && std::strcmp(singleNodeValue, "1") == 0) return true;

  const char* value = std::getenv("OCCL_NET_IB_GPU_LOCAL_E2E");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

int DeviceIndexFromEnv(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return fallback;

  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
    ADD_FAILURE() << name << " must be a non-negative integer, got '" << value << "'";
    return fallback;
  }
  return static_cast<int>(parsed);
}

testing::AssertionResult ResultOk(ncclResult_t result, const char* call) {
  if (result == ncclSuccess) return testing::AssertionSuccess();
  return testing::AssertionFailure() << call << " returned " << result;
}

class CommPair {
 public:
  CommPair() = default;
  CommPair(const CommPair&) = delete;
  CommPair& operator=(const CommPair&) = delete;

  ~CommPair() { Close(); }

  void Close() {
    if (send_ != nullptr) {
      (void)ncclIbCloseSend(send_);
      send_ = nullptr;
    }
    if (recv_ != nullptr) {
      (void)ncclIbCloseRecv(recv_);
      recv_ = nullptr;
    }
    if (listen_ != nullptr) {
      (void)ncclIbCloseListen(listen_);
      listen_ = nullptr;
    }
  }

  testing::AssertionResult CloseChecked() {
    if (send_ != nullptr) {
      ncclResult_t result = ncclIbCloseSend(send_);
      send_ = nullptr;
      if (result != ncclSuccess) return ResultOk(result, "ncclIbCloseSend");
    }
    if (recv_ != nullptr) {
      ncclResult_t result = ncclIbCloseRecv(recv_);
      recv_ = nullptr;
      if (result != ncclSuccess) return ResultOk(result, "ncclIbCloseRecv");
    }
    if (listen_ != nullptr) {
      ncclResult_t result = ncclIbCloseListen(listen_);
      listen_ = nullptr;
      if (result != ncclSuccess) return ResultOk(result, "ncclIbCloseListen");
    }
    return testing::AssertionSuccess();
  }

  void* listen_ = nullptr;
  void* send_ = nullptr;
  void* recv_ = nullptr;
};

class MemoryHandle {
 public:
  MemoryHandle() = default;
  MemoryHandle(const MemoryHandle&) = delete;
  MemoryHandle& operator=(const MemoryHandle&) = delete;

  MemoryHandle(MemoryHandle&& other) noexcept {
    comm_ = other.comm_;
    handle_ = other.handle_;
    other.comm_ = nullptr;
    other.handle_ = nullptr;
  }

  MemoryHandle& operator=(MemoryHandle&& other) noexcept {
    if (this == &other) return *this;
    Deregister();
    comm_ = other.comm_;
    handle_ = other.handle_;
    other.comm_ = nullptr;
    other.handle_ = nullptr;
    return *this;
  }

  ~MemoryHandle() { Deregister(); }

  testing::AssertionResult Register(void* comm, void* data, size_t size, int type) {
    comm_ = comm;
    ncclResult_t result = ncclIbRegMr(comm_, data, size, type, &handle_);
    return ResultOk(result, "ncclIbRegMr");
  }

  testing::AssertionResult DeregisterChecked() {
    if (handle_ == nullptr) return testing::AssertionSuccess();
    ncclResult_t result = ncclIbDeregMr(comm_, handle_);
    handle_ = nullptr;
    comm_ = nullptr;
    return ResultOk(result, "ncclIbDeregMr");
  }

  void* get() const { return handle_; }

 private:
  void Deregister() {
    if (handle_ != nullptr) {
      (void)ncclIbDeregMr(comm_, handle_);
      handle_ = nullptr;
      comm_ = nullptr;
    }
  }

  void* comm_ = nullptr;
  void* handle_ = nullptr;
};

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  testing::AssertionResult Allocate(size_t size) {
    cudaError_t result = cudaMalloc(&ptr_, size);
    if (result == cudaSuccess) return testing::AssertionSuccess();
    return testing::AssertionFailure() << "cudaMalloc failed: " << cudaGetErrorString(result);
  }

  void* get() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
};

class NetIbSingleNodeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!RuntimeEnabled()) {
      GTEST_SKIP() << "Set OCCL_NET_IB_SINGLE_NODE_TESTS=1 to run single-node net-IB tests.";
    }

    ncclNetCommConfig_t config = {};
    config.trafficClass = 0;
    ASSERT_EQ(ncclIbInit(&context_, 0, &config, TestLog, nullptr), ncclSuccess);

    ASSERT_EQ(ncclIbDevices(&device_count_), ncclSuccess);
    if (device_count_ <= 0) {
      GTEST_SKIP() << "No net-IB virtual devices are available.";
    }

    listen_device_ = DeviceIndexFromEnv("OCCL_NET_IB_E2E_LISTEN_DEV", 0);
    connect_device_ = DeviceIndexFromEnv("OCCL_NET_IB_E2E_CONNECT_DEV", listen_device_);
    ASSERT_GE(listen_device_, 0);
    ASSERT_GE(connect_device_, 0);
    ASSERT_LT(listen_device_, device_count_);
    ASSERT_LT(connect_device_, device_count_);
  }

  void TearDown() override {
    if (context_ != nullptr) {
      EXPECT_EQ(ncclIbFinalize(context_), ncclSuccess);
      context_ = nullptr;
    }
  }

  testing::AssertionResult OpenPair(CommPair* pair) {
    std::array<unsigned char, NCCL_NET_HANDLE_MAXSIZE> handle = {};
    ncclResult_t result = ncclIbListen(context_, listen_device_, handle.data(), &pair->listen_);
    if (result != ncclSuccess) return ResultOk(result, "ncclIbListen");
    if (pair->listen_ == nullptr) return testing::AssertionFailure() << "ncclIbListen returned a null comm";

    ncclNetDeviceHandle_t* send_device = nullptr;
    ncclNetDeviceHandle_t* recv_device = nullptr;
    const auto deadline = std::chrono::steady_clock::now() + kPollTimeout;
    while (std::chrono::steady_clock::now() < deadline && (pair->send_ == nullptr || pair->recv_ == nullptr)) {
      if (pair->send_ == nullptr) {
        result = ncclIbConnect(context_, connect_device_, handle.data(), &pair->send_, &send_device);
        if (result != ncclSuccess) return ResultOk(result, "ncclIbConnect");
      }
      if (pair->recv_ == nullptr) {
        result = ncclIbAccept(pair->listen_, &pair->recv_, &recv_device);
        if (result != ncclSuccess) return ResultOk(result, "ncclIbAccept");
      }
      if (pair->send_ == nullptr || pair->recv_ == nullptr) {
        std::this_thread::sleep_for(kPollInterval);
      }
    }

    if (pair->send_ == nullptr || pair->recv_ == nullptr) {
      return testing::AssertionFailure() << "Timed out waiting for net-IB connect/accept";
    }
    return testing::AssertionSuccess();
  }

  testing::AssertionResult PollRequest(void* request, int* sizes) {
    if (request == nullptr) return testing::AssertionFailure() << "Cannot poll a null request";

    const auto deadline = std::chrono::steady_clock::now() + kPollTimeout;
    while (std::chrono::steady_clock::now() < deadline) {
      int done = 0;
      ncclResult_t result = ncclIbTest(request, &done, sizes);
      if (result != ncclSuccess) return ResultOk(result, "ncclIbTest");
      if (done) return testing::AssertionSuccess();
      std::this_thread::sleep_for(kPollInterval);
    }
    return testing::AssertionFailure() << "Timed out polling net-IB request";
  }

  testing::AssertionResult PostSend(void* send_comm, void* data, size_t size, int tag, void* handle, void** request) {
    const auto deadline = std::chrono::steady_clock::now() + kPollTimeout;
    while (std::chrono::steady_clock::now() < deadline) {
      ncclResult_t result = ncclIbIsend(send_comm, data, size, tag, handle, nullptr, request);
      if (result != ncclSuccess) return ResultOk(result, "ncclIbIsend");
      if (*request != nullptr) return testing::AssertionSuccess();
      std::this_thread::sleep_for(kPollInterval);
    }
    return testing::AssertionFailure() << "Timed out waiting for ncclIbIsend to return a request";
  }

  testing::AssertionResult RunHostTransfer(const std::vector<size_t>& sizes, const std::vector<int>& tags) {
    if (sizes.size() != tags.size()) {
      return testing::AssertionFailure() << "Transfer sizes and tags must have matching lengths";
    }

    CommPair pair;
    auto open_result = OpenPair(&pair);
    if (!open_result) return open_result;

    std::vector<std::vector<unsigned char>> send_buffers;
    std::vector<std::vector<unsigned char>> recv_buffers;
    std::vector<MemoryHandle> send_handles;
    std::vector<MemoryHandle> recv_handles;
    std::vector<void*> recv_data;
    std::vector<size_t> recv_sizes;
    std::vector<void*> recv_handle_values;
    std::vector<void*> profiler_handles;
    const int count = static_cast<int>(sizes.size());

    send_buffers.reserve(count);
    recv_buffers.reserve(count);
    send_handles.reserve(count);
    recv_handles.reserve(count);
    recv_data.reserve(count);
    recv_sizes.reserve(count);
    recv_handle_values.reserve(count);
    profiler_handles.assign(count, nullptr);

    for (int i = 0; i < count; i++) {
      send_buffers.emplace_back(sizes[i], static_cast<unsigned char>(0x30 + i));
      recv_buffers.emplace_back(sizes[i], 0);
      send_handles.emplace_back();
      recv_handles.emplace_back();

      auto recv_reg = recv_handles.back().Register(pair.recv_, recv_buffers.back().data(), recv_buffers.back().size(), NCCL_PTR_HOST);
      if (!recv_reg) return recv_reg;
      auto send_reg = send_handles.back().Register(pair.send_, send_buffers.back().data(), send_buffers.back().size(), NCCL_PTR_HOST);
      if (!send_reg) return send_reg;

      recv_data.push_back(recv_buffers.back().data());
      recv_sizes.push_back(recv_buffers.back().size());
      recv_handle_values.push_back(recv_handles.back().get());
    }

    void* recv_request = nullptr;
    ncclResult_t result = ncclIbIrecv(pair.recv_, count, recv_data.data(), recv_sizes.data(),
                                      const_cast<int*>(tags.data()), recv_handle_values.data(),
                                      profiler_handles.data(), &recv_request);
    if (result != ncclSuccess) return ResultOk(result, "ncclIbIrecv");
    if (recv_request == nullptr) return testing::AssertionFailure() << "ncclIbIrecv returned a null request";

    std::vector<void*> send_requests(count, nullptr);
    for (int i = 0; i < count; i++) {
      auto post_result = PostSend(pair.send_, send_buffers[i].data(), send_buffers[i].size(), tags[i],
                                  send_handles[i].get(), &send_requests[i]);
      if (!post_result) return post_result;
    }

    for (int i = 0; i < count; i++) {
      int sent_size = 0;
      auto poll_result = PollRequest(send_requests[i], &sent_size);
      if (!poll_result) return poll_result;
      if (sent_size != static_cast<int>(sizes[i])) {
        return testing::AssertionFailure() << "Unexpected send size at index " << i << ": " << sent_size;
      }
    }

    std::vector<int> received_sizes(count, 0);
    auto recv_poll_result = PollRequest(recv_request, received_sizes.data());
    if (!recv_poll_result) return recv_poll_result;

    for (int i = 0; i < count; i++) {
      if (received_sizes[i] != static_cast<int>(sizes[i])) {
        return testing::AssertionFailure() << "Unexpected recv size at index " << i << ": " << received_sizes[i];
      }
      if (std::memcmp(send_buffers[i].data(), recv_buffers[i].data(), sizes[i]) != 0) {
        return testing::AssertionFailure() << "Payload mismatch at index " << i;
      }
    }

    for (int i = count - 1; i >= 0; i--) {
      auto recv_dereg = recv_handles[i].DeregisterChecked();
      if (!recv_dereg) return recv_dereg;
      auto send_dereg = send_handles[i].DeregisterChecked();
      if (!send_dereg) return send_dereg;
    }

    return pair.CloseChecked();
  }

  void* context_ = nullptr;
  int device_count_ = 0;
  int listen_device_ = 0;
  int connect_device_ = 0;
};

TEST_F(NetIbSingleNodeTest, InitDevicesPropertiesFinalize) {
  ASSERT_GT(device_count_, 0);

  for (int dev = 0; dev < device_count_; dev++) {
    ncclNetProperties_t props = {};
    ASSERT_EQ(ncclIbGetProperties(dev, &props), ncclSuccess);
    EXPECT_NE(props.name, nullptr);
    EXPECT_GT(props.maxRecvs, 0);
  }
}

TEST_F(NetIbSingleNodeTest, ListenConnectAcceptAndClose) {
  CommPair pair;
  ASSERT_TRUE(OpenPair(&pair));
  ASSERT_NE(pair.send_, nullptr);
  ASSERT_NE(pair.recv_, nullptr);
  ASSERT_TRUE(pair.CloseChecked());
}

TEST_F(NetIbSingleNodeTest, RegisterOverlappingHostBuffers) {
  CommPair pair;
  ASSERT_TRUE(OpenPair(&pair));

  constexpr size_t buffer_size = 8192;
  std::vector<unsigned char> buffer(buffer_size, 0);
  MemoryHandle first;
  MemoryHandle second;
  MemoryHandle third;

  ASSERT_TRUE(first.Register(pair.send_, buffer.data() + 512, 2048, NCCL_PTR_HOST));
  ASSERT_TRUE(second.Register(pair.send_, buffer.data() + 1024, 512, NCCL_PTR_HOST));
  ASSERT_TRUE(third.Register(pair.send_, buffer.data() + 2048, 2048, NCCL_PTR_HOST));

  EXPECT_TRUE(second.DeregisterChecked());
  EXPECT_TRUE(first.DeregisterChecked());
  EXPECT_TRUE(third.DeregisterChecked());
  EXPECT_TRUE(pair.CloseChecked());
}

TEST_F(NetIbSingleNodeTest, HostRoundTripOneBuffer) {
  ASSERT_TRUE(RunHostTransfer({1024}, {123}));
}

TEST_F(NetIbSingleNodeTest, HostRoundTripTwoBuffersDifferentTags) {
  ASSERT_TRUE(RunHostTransfer({512, 1536}, {231, 232}));
}

TEST_F(NetIbSingleNodeTest, HostFlushAfterReceive) {
  CommPair pair;
  ASSERT_TRUE(OpenPair(&pair));

  constexpr size_t length = 2048;
  std::vector<unsigned char> send_buffer(length, 0xab);
  std::vector<unsigned char> recv_buffer(length, 0);
  MemoryHandle send_handle;
  MemoryHandle recv_handle;
  ASSERT_TRUE(recv_handle.Register(pair.recv_, recv_buffer.data(), recv_buffer.size(), NCCL_PTR_HOST));
  ASSERT_TRUE(send_handle.Register(pair.send_, send_buffer.data(), send_buffer.size(), NCCL_PTR_HOST));

  void* recv_data[1] = {recv_buffer.data()};
  size_t recv_sizes[1] = {recv_buffer.size()};
  int flush_sizes[1] = {static_cast<int>(recv_buffer.size())};
  int tags[1] = {321};
  void* recv_handles[1] = {recv_handle.get()};
  void* profiler_handles[1] = {nullptr};
  void* recv_request = nullptr;
  ASSERT_EQ(ncclIbIrecv(pair.recv_, 1, recv_data, recv_sizes, tags, recv_handles, profiler_handles, &recv_request), ncclSuccess);
  ASSERT_NE(recv_request, nullptr);

  void* send_request = nullptr;
  ASSERT_TRUE(PostSend(pair.send_, send_buffer.data(), send_buffer.size(), tags[0], send_handle.get(), &send_request));

  int sent_size = 0;
  int received_size = 0;
  ASSERT_TRUE(PollRequest(send_request, &sent_size));
  ASSERT_TRUE(PollRequest(recv_request, &received_size));
  ASSERT_EQ(sent_size, received_size);
  ASSERT_EQ(std::memcmp(send_buffer.data(), recv_buffer.data(), recv_buffer.size()), 0);

  void* flush_request = nullptr;
  ASSERT_EQ(ncclIbIflush(pair.recv_, 1, recv_data, flush_sizes, recv_handles, &flush_request), ncclSuccess);
  if (flush_request == nullptr) {
    GTEST_SKIP() << "Flush is not enabled for the selected net-IB device.";
  }
  ASSERT_TRUE(PollRequest(flush_request, nullptr));

  EXPECT_TRUE(recv_handle.DeregisterChecked());
  EXPECT_TRUE(send_handle.DeregisterChecked());
  EXPECT_TRUE(pair.CloseChecked());
}

TEST_F(NetIbSingleNodeTest, CudaRoundTripSmoke) {
  int cuda_device_count = 0;
  cudaError_t cuda_result = cudaGetDeviceCount(&cuda_device_count);
  if (cuda_result != cudaSuccess || cuda_device_count <= 0) {
    GTEST_SKIP() << "CUDA runtime did not report an available GPU: " << cudaGetErrorString(cuda_result);
  }

  ncclNetProperties_t props = {};
  ASSERT_EQ(ncclIbGetProperties(connect_device_, &props), ncclSuccess);
  if ((props.ptrSupport & NCCL_PTR_CUDA) == 0) {
    GTEST_SKIP() << "Selected net-IB device does not report CUDA pointer support.";
  }

  CommPair pair;
  ASSERT_TRUE(OpenPair(&pair));

  constexpr size_t length = 1024;
  std::vector<unsigned char> host_send(length, 0x5a);
  std::vector<unsigned char> host_recv(length, 0);
  DeviceBuffer device_send;
  DeviceBuffer device_recv;
  ASSERT_TRUE(device_send.Allocate(length));
  ASSERT_TRUE(device_recv.Allocate(length));
  ASSERT_EQ(cudaMemcpy(device_send.get(), host_send.data(), length, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(device_recv.get(), 0, length), cudaSuccess);

  MemoryHandle send_handle;
  MemoryHandle recv_handle;
  auto recv_reg = recv_handle.Register(pair.recv_, device_recv.get(), length, NCCL_PTR_CUDA);
  auto send_reg = send_handle.Register(pair.send_, device_send.get(), length, NCCL_PTR_CUDA);
  if (!recv_reg || !send_reg) {
    (void)recv_handle.DeregisterChecked();
    (void)send_handle.DeregisterChecked();
    GTEST_SKIP() << "CUDA memory registration is not supported by the selected net-IB path.";
  }

  void* recv_data[1] = {device_recv.get()};
  size_t recv_sizes[1] = {length};
  int tags[1] = {411};
  void* recv_handles[1] = {recv_handle.get()};
  void* profiler_handles[1] = {nullptr};
  void* recv_request = nullptr;
  ASSERT_EQ(ncclIbIrecv(pair.recv_, 1, recv_data, recv_sizes, tags, recv_handles, profiler_handles, &recv_request), ncclSuccess);
  ASSERT_NE(recv_request, nullptr);

  void* send_request = nullptr;
  ASSERT_TRUE(PostSend(pair.send_, device_send.get(), length, tags[0], send_handle.get(), &send_request));

  int sent_size = 0;
  int received_size = 0;
  ASSERT_TRUE(PollRequest(send_request, &sent_size));
  ASSERT_TRUE(PollRequest(recv_request, &received_size));
  ASSERT_EQ(sent_size, received_size);
  ASSERT_EQ(cudaMemcpy(host_recv.data(), device_recv.get(), length, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(std::memcmp(host_send.data(), host_recv.data(), length), 0);

  EXPECT_TRUE(recv_handle.DeregisterChecked());
  EXPECT_TRUE(send_handle.DeregisterChecked());
  EXPECT_TRUE(pair.CloseChecked());
}

}  // namespace
