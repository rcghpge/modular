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

#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "Support/MDialect/MAttrs.h"

#include "mlir/IR/AttrTypeSubElements.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_CANONICALIZEFUNCDEVICEIDS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Must be 0, not an out-of-range sentinel: some ops (e.g.
// `MGP::BufferConstantOp`) infer `DeviceRefAttr::getDefault` (id 0) as their
// result device and would fail re-verification otherwise. The emitted Mojo
// reads only the device label, never the id.
constexpr uint64_t kCanonicalDeviceId = 0;

struct CanonicalizeFuncDeviceIds
    : public impl::CanonicalizeFuncDeviceIdsBase<CanonicalizeFuncDeviceIds> {
  using CanonicalizeFuncDeviceIdsBase::CanonicalizeFuncDeviceIdsBase;

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::MLIRContext *ctx = moduleOp.getContext();

    // Device ids live on `#M.device_ref` in `!mgp.tensor` types and in a few
    // op attributes, so replace in both.
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([ctx](DeviceRefAttr deviceRef) -> DeviceRefAttr {
      if (deviceRef.getId() == kCanonicalDeviceId)
        return deviceRef;
      return DeviceRefAttr::get(ctx, deviceRef.getLabel(), kCanonicalDeviceId);
    });

    for (FuncOp kernelStub :
         llvm::make_early_inc_range(moduleOp.getOps<FuncOp>()))
      replacer.recursivelyReplaceElementsIn(kernelStub, /*replaceAttrs=*/true,
                                            /*replaceLocs=*/false,
                                            /*replaceTypes=*/true);
  }
};

} // namespace

} // namespace M::MOGG
