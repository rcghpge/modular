//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CPU_DEVICE_H
#define TARGET_LIB_CPU_DEVICE_H

#include "HAL/M_driver_common.h"

struct M_driver_device {
  explicit M_driver_device(M_driver_driver_t driver) : driver(driver) {}

  M_driver_driver_t driver;
};

#endif // TARGET_LIB_CPU_DEVICE_H
