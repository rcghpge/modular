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

#include "GraphCompiler/Conversion/PackageVariadicsAndScalars/PackageVariadicsAndScalars.h"
#include "GraphCompiler/Conversion/OperandInference.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_PACKAGEVARIADICSANDSCALARS
#include "GraphCompiler/Conversion/PackageVariadicsAndScalars/PackageVariadicsAndScalars.h.inc"

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

LogicalResult packCall(CallOpaqueOp call, const MojoRegistry &registry) {
  KernelDeclOp decl = registry.lookupKernelBySymName(call.getCallee());
  if (!decl)
    return call.emitOpError(
               "PackageVariadicsAndScalars: no kernel decl named '")
           << call.getCallee() << "'";

  KernelDeclAdaptor adaptor(decl);
  if (!adaptor.executeFunction)
    return success();

  ErrorOr<KernelDeclOperandMapping> mapping = mapToKernelImplementation(
      call.getInputs(), call.getOutputs(), *adaptor.executeFunction,
      call.getCallee(), /*hasSameVariadicOperandSize=*/false,
      /*hasExtraOutputShapeArgument=*/false);
  if (mapping.isError())
    return call.emitOpError("PackageVariadicsAndScalars: ")
           << mapping.getError();

  OpBuilder builder(call);
  bool changed = false;
  SmallVector<Value> inputs =
      packOperands(builder, call.getLoc(), call.getInputs(),
                   mapping->operandMapping, changed);
  SmallVector<Value> outputs =
      packOperands(builder, call.getLoc(), call.getOutputs(),
                   mapping->resultMapping, changed);
  if (!changed)
    return success();

  CallOpaqueOp::create(builder, call.getLoc(), call.getCalleeAttr(), inputs,
                       outputs, call.getDeviceContext(),
                       call.getKernelParamAttr());
  call.erase();
  return success();
}

struct PackageVariadicsAndScalarsPass
    : public MOGG::impl::PackageVariadicsAndScalarsBase<
          PackageVariadicsAndScalarsPass> {
  using Base::Base;

  void runOnOperation() override {
    MojoRegistry registry(getOperation());
    SmallVector<CallOpaqueOp> calls;
    getOperation().walk([&](CallOpaqueOp c) {
      calls.push_back(c);
      return WalkResult::skip();
    });
    for (CallOpaqueOp call : calls)
      if (failed(packCall(call, registry)))
        return signalPassFailure();
  }
};

} // namespace

} // namespace M::MOGG
