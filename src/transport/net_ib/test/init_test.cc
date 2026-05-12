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

#include "test_net_ib_state.h"

#include <gtest/gtest.h>

int64_t ncclIbArThreshold = 8192;

int64_t ncclParamIbPrepostReceiveWorkRequests() {
  return -2;
}

int64_t ncclParamIbReceiverSideMatchingScheme() {
  return -2;
}

ncclResult_t ncclIbPortRecoveryThreadStart() {
  return ncclSuccess;
}

ncclResult_t ncclIbPortRecoveryThreadStop() {
  return ncclSuccess;
}

ncclResult_t ncclIbGdrSupport() {
  return ncclSystemError;
}

ncclResult_t ncclIbDmaBufSupport(int) {
  return ncclSystemError;
}

#include "../init.cc"  // NOLINT(build/include)

namespace {

void ResetDevices() {
  ncclNIbDevs = 0;
  ncclNMergedIbDevs = 0;
  std::memset(ncclIbMergedDevs, 0, sizeof(ncclIbMergedDevs));
  for (auto& dev : ncclIbDevs) {
    dev.speed = 0;
    dev.link = 0;
    dev.portNum = 0;
    dev.devName[0] = '\0';
  }
}

}  // namespace

TEST(NetIbInitTest, WidthAndSpeedDecodeKnownValues) {
  EXPECT_EQ(ncclIbWidth(1), 1);
  EXPECT_EQ(ncclIbWidth(2), 4);
  EXPECT_EQ(ncclIbWidth(4), 8);
  EXPECT_EQ(ncclIbWidth(8), 12);
  EXPECT_EQ(ncclIbWidth(16), 2);
  EXPECT_EQ(ncclIbWidth(0), 2);

  EXPECT_EQ(ncclIbSpeed(1), 2500);
  EXPECT_EQ(ncclIbSpeed(2), 5000);
  EXPECT_EQ(ncclIbSpeed(4), 10000);
  EXPECT_EQ(ncclIbSpeed(8), 10000);
  EXPECT_EQ(ncclIbSpeed(16), 14000);
  EXPECT_EQ(ncclIbSpeed(32), 25000);
  EXPECT_EQ(ncclIbSpeed(64), 50000);
  EXPECT_EQ(ncclIbSpeed(128), 100000);
  EXPECT_EQ(ncclIbSpeed(0), 200000);
}

TEST(NetIbInitTest, MakeVirtualDeviceRejectsInvalidInputs) {
  ResetDevices();
  ncclNIbDevs = 1;
  int dev = -1;
  ncclNetVDeviceProps_t props = {};

  props.ndevs = 0;
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&dev, &props), ncclInvalidUsage);

  props.ndevs = NCCL_IB_MAX_DEVS_PER_NIC + 1;
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&dev, &props), ncclInvalidUsage);

  ncclNMergedIbDevs = MAX_IB_VDEVS;
  props = {};
  props.ndevs = 1;
  props.devs[0] = 0;
  EXPECT_EQ(ncclIbMakeVDeviceInternal(&dev, &props), ncclInvalidUsage);
}

TEST(NetIbInitTest, MakeVirtualDeviceStoresSortedPhysicalDevices) {
  ResetDevices();
  ncclNIbDevs = 3;
  ncclIbDevs[0].speed = 100;
  ncclIbDevs[1].speed = 200;
  ncclIbDevs[2].speed = 300;

  int dev = -1;
  ncclNetVDeviceProps_t props = {};
  props.ndevs = 2;
  props.devs[0] = 2;
  props.devs[1] = 0;

  ASSERT_EQ(ncclIbMakeVDeviceInternal(&dev, &props), ncclSuccess);

  EXPECT_EQ(dev, 0);
  EXPECT_EQ(ncclNMergedIbDevs, 1);
  EXPECT_EQ(ncclIbMergedDevs[0].vProps.ndevs, 2);
  EXPECT_EQ(ncclIbMergedDevs[0].vProps.devs[0], 2);
  EXPECT_EQ(ncclIbMergedDevs[0].vProps.devs[1], 0);
  EXPECT_EQ(ncclIbMergedDevs[0].speed, 400);
}

TEST(NetIbInitTest, PropertiesRejectInvalidMergedDevice) {
  ResetDevices();
  ncclNMergedIbDevs = 1;
  ncclNetProperties_t props = {};

  EXPECT_EQ(ncclIbGetProperties(1, &props), ncclInvalidUsage);
}
