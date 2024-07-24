//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTensorInterface.h"
#include "GenericML/MGPTransforms/MGPTransforms.h"
#include "GenericML/MGPTransforms/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace M::MGP {

// Generate support code for VerifyModelArgumentsPass.
#define GEN_PASS_DEF_VERIFYMODELARGUMENTSPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// VerifyModelArgumentsPass
//===----------------------------------------------------------------------===//

static std::string shapeToString(ArrayRef<int64_t> shape) {
  const auto dimToString = [](int64_t dim) {
    static constexpr std::string_view kDynamicDimensionLabel = "any";
    if (dim == ShapedType::kDynamic)
      return std::string(kDynamicDimensionLabel);

    return std::to_string(dim);
  };
  std::string str;
  llvm::raw_string_ostream stream(str);
  for (uint64_t i = 0; i < shape.size() - 1; ++i)
    stream << dimToString(shape[i]) << ",";
  stream << dimToString(shape.back());

  return str;
}

class VerifyModelArgumentsPass
    : public impl::VerifyModelArgumentsPassBase<VerifyModelArgumentsPass> {
  using VerifyModelArgumentsPassBase::VerifyModelArgumentsPassBase;

  void runOnOperation() override {
    auto &ctx = getContext();
    OpBuilder builder(&ctx);
    ModelOp model = getOperation();
    BufferBuilderContext builderContext =
        TensorBufferDialectInterface::getAllBufferBuilders(model);
    Block *executeBlock = model.getExecuteBlock();
    ExecuteState executeState(&ctx, model);
    Location loc = model->getLoc();

    builder.setInsertionPointToStart(executeBlock);
    for (BlockArgument arg : executeState.getExtraArgs()) {
      auto tensorArg = dyn_cast<DeviceTensorType>(arg.getType());
      if (!tensorArg)
        continue;

      if (!tensorArg.hasRank())
        continue;

      if (tensorArg.getNumDynamicDims() == tensorArg.getRank())
        continue;

      BufferBuilder *bufferBuilder =
          builderContext.getBufferBuilder(tensorArg.getDevice());
      auto expectedTensorSpec = tensorArg.getTensorSpec();
      auto runtimeTensorSpec = bufferBuilder->tensorExtractTensorSpec(
          builder, loc, executeState.getContext(), arg, /*isShapeOnly=*/true);
      auto areEqual =
          builder
              .create<CpuTensorSpecEqualStaticOp>(
                  loc, runtimeTensorSpec, expectedTensorSpec.getRawDims())
              .getResult();

      std::string message;
      llvm::raw_string_ostream stream(message);
      stream << llvm::format(
          "Tensor type mismatch for argument %ld: Expected shape: [%s]",
          arg.getArgNumber() - model.getExecuteImplicitArguments().size(),
          shapeToString(expectedTensorSpec.getShape()).c_str());
      builder.create<AssertOp>(loc, areEqual,
                               StringAttr::get(&ctx, stream.str()));
    }
  }
};

} // namespace
} // namespace M::MGP
