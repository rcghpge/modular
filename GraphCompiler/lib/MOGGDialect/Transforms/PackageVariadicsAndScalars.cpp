//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Materializes the scalar and tuple (variadic) operands a `mogg._call.opaque`'s
// kernel expects. `MAPToMOGG` leaves every operand as a plain `!mogg.tensor_`;
// this pass re-runs operand inference against the callee `mogg.kernel.decl`
// and, per its classification, wraps a scalar operand in `mogg._scalar.create`
// and packs each variadic run into a `mogg._tuple.create`.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/OperandInference.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_PACKAGEVARIADICSANDSCALARS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Packs one operand list (a call's inputs or outputs) into the scalar / tuple
// forms the kernel expects, per `mapping`: a scalar kernel argument gets its
// `!mogg.tensor_` operand wrapped in `mogg._scalar.create`; a variadic argument
// gets its run of consecutive `!mogg.tensor_` operands (same kernel-argument
// position) packed into one `mogg._tuple.create`. `changed` is set if anything
// was rewritten.
SmallVector<Value>
packOperands(OpBuilder &builder, Location loc, ValueRange operands,
             const KernelDeclIndexToOperandAdaptorMapping &mapping,
             bool &changed) {
  MLIRContext *ctx = builder.getContext();
  SmallVector<Value> packed;
  for (size_t i = 0, e = operands.size(); i < e;) {
    auto it = mapping.find(i);
    if (it == mapping.end()) {
      packed.push_back(operands[i]);
      ++i;
      continue;
    }
    const KernelDeclOperandAdaptor &arg = it->second;

    if (arg.isScalarType()) {
      packed.push_back(ScalarCreateOp::create(
          builder, loc, ScalarType::get(ctx), operands[i]));
      changed = true;
      ++i;
      continue;
    }

    if (arg.isVariadicTensorType()) {
      unsigned pos = arg.sourceDescriptor->position;
      SmallVector<Value> run;
      while (i < e) {
        auto runIt = mapping.find(i);
        if (runIt == mapping.end() || !runIt->second.isVariadicTensorType() ||
            runIt->second.sourceDescriptor->position != pos)
          break;
        run.push_back(operands[i]);
        ++i;
      }
      packed.push_back(
          TupleCreateOp::create(builder, loc, TupleType::get(ctx), run));
      changed = true;
      continue;
    }

    packed.push_back(operands[i]);
    ++i;
  }
  return packed;
}

// Packed operands, and whether anything needed it.
struct PackedCall {
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
  bool changed = false;
};

// Wraps each operand `fn` wants as a scalar or a tuple.
FailureOr<PackedCall> packAgainstSignature(Operation *call,
                                           const KernelDeclFunctionAdaptor &fn,
                                           StringRef callee, ValueRange inputs,
                                           ValueRange outputs,
                                           bool hasSameVariadicOperandSize) {
  ErrorOr<KernelDeclOperandMapping> mapping = mapToKernelImplementation(
      inputs, outputs, fn, callee, hasSameVariadicOperandSize,
      /*hasExtraOutputShapeArgument=*/false);
  if (mapping.isError())
    return call->emitOpError("PackageVariadicsAndScalars: ")
           << mapping.getError();

  OpBuilder builder(call);
  PackedCall packed;
  packed.inputs = packOperands(builder, call->getLoc(), inputs,
                               mapping->operandMapping, packed.changed);
  packed.outputs = packOperands(builder, call->getLoc(), outputs,
                                mapping->resultMapping, packed.changed);
  return packed;
}

LogicalResult packCall(CallOpaqueOp call, const MojoRegistry &registry) {
  KernelDeclOp decl = registry.lookupKernelBySymName(call.getCallee());
  if (!decl)
    return call.emitOpError(
               "PackageVariadicsAndScalars: no kernel decl named '")
           << call.getCallee() << "'";

  KernelDeclAdaptor adaptor(decl);
  if (!adaptor.executeFunction)
    return success();

  FailureOr<PackedCall> packed = packAgainstSignature(
      call, *adaptor.executeFunction, call.getCallee(), call.getInputs(),
      call.getOutputs(), call.getHasSameVariadicOperandSize());
  if (failed(packed))
    return failure();
  if (!packed->changed)
    return success();

  OpBuilder builder(call);
  CallOpaqueOp::create(builder, call.getLoc(), call.getCalleeAttr(),
                       packed->inputs, packed->outputs, call.getDeviceContext(),
                       call.getKernelParamAttr());
  call.erase();
  return success();
}

// Same as `packCall`, against the `mogg.shape_func.decl` signature: no
// outputs, but `range_shape` still wants `Scalar[dtype]`, not a
// `!mogg.tensor_`.
LogicalResult packShapeCall(CallShapeV2Op call, const MojoRegistry &registry) {
  ShapeFunctionDeclOp decl =
      registry.lookupShapeFunctionBySymName(call.getCallee());
  if (!decl)
    return call.emitOpError(
               "PackageVariadicsAndScalars: no shape_func.decl named '")
           << call.getCallee() << "'";

  // No output operand to map.
  KernelDeclFunctionAdaptor fn(decl);
  fn.outputResult = std::nullopt;

  FailureOr<PackedCall> packed = packAgainstSignature(
      call, fn, call.getCallee(), call.getOperands(), /*outputs=*/{},
      /*hasSameVariadicOperandSize=*/false);
  if (failed(packed))
    return failure();
  if (!packed->changed)
    return success();

  OpBuilder builder(call);
  auto replacement = CallShapeV2Op::create(
      builder, call.getLoc(), call.getResult().getType(), call.getCalleeAttr(),
      call.getIsThrowsAttr(), packed->inputs, call.getKernelParamAttr());
  call.getResult().replaceAllUsesWith(replacement.getResult());
  call.erase();
  return success();
}

struct PackageVariadicsAndScalarsPass
    : public MOGG::impl::PackageVariadicsAndScalarsBase<
          PackageVariadicsAndScalarsPass> {
  using Base::Base;

  void runOnOperation() override {
    MojoRegistry registry(getOperation());
    // Post-order: packing erases the call.
    WalkResult result = getOperation().walk([&](Operation *op) {
      LogicalResult packed = success();
      if (auto call = dyn_cast<CallOpaqueOp>(op))
        packed = packCall(call, registry);
      else if (auto shapeCall = dyn_cast<CallShapeV2Op>(op))
        packed = packShapeCall(shapeCall, registry);
      return failed(packed) ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace

} // namespace M::MOGG
