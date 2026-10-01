//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Conforms each `mogg` call's operands to the callee's declared signature.
// `MAPToMOGG` leaves every operand as a plain `!mogg.tensor`; this pass
// re-runs operand inference against the callee `mogg.kernel.decl` /
// `mogg.shape_func.decl` and, per its classification, wraps a scalar operand in
// `mogg.scalar.create`, packs each variadic run into a `mogg.tuple.create`,
// and wraps a trait-bound argument in a `mogg.trait.create`. Doing this here
// keeps each `::emitMojo` a local, 1:1 lowering: the call emit forwards
// whatever handle the IR gives it.
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

#define GEN_PASS_DEF_CONFORMCALLOPERANDS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Packs one operand list (a call's inputs or outputs) into the scalar / tuple
// forms the kernel expects, per `mapping`: a scalar kernel argument gets its
// `!mogg.tensor` operand wrapped in `mogg.scalar.create`; a variadic argument
// gets its run of consecutive `!mogg.tensor` operands (same kernel-argument
// position) packed into one `mogg.tuple.create`. `changed` is set if anything
// was rewritten.
SmallVector<Value>
packOperands(OpBuilder &builder, Location loc, ValueRange operands,
             const KernelDeclIndexToOperandAdaptorMapping &mapping,
             M::DeviceRefAttr device, bool &changed) {
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
      // Record the ABI the callee's variadic parameter wants, so emit picks the
      // aggregate without inspecting the run's producing ops. The variant's
      // own IOSpec (not store mutability -- a fused output is not a
      // `MutableInputTensor`) distinguishes the three cases.
      VariadicABI abi = VariadicABI::Plain;
      if (const auto *v = std::get_if<KernelDeclVariadicTensorAdaptor>(
              &arg.underlyingType)) {
        if (v->ioSpec == IOSpec::FusedInputTensor)
          abi = VariadicABI::FusedInput;
        else if (v->ioSpec == IOSpec::FusedOutputTensor)
          abi = VariadicABI::FusedOutput;
      }
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
          TupleCreateOp::create(builder, loc, TupleType::get(ctx), run, abi));
      changed = true;
      continue;
    }

    // A NON-fused trait-bound tensor argument (`Tensor` / `DenseTensor` /
    // `TileTensorable`) has its raw `!mogg.tensor` re-presented as the
    // conforming trait argument by `mogg.trait.create`, which carries the
    // distilled contract and the execution device (for the emit-side layout /
    // alignment). A FUSED trait argument is handled at its producer instead:
    // its fusion functor is a local of the prologue/epilogue emit, so the
    // contract is stamped onto the `mogg.tensor.create.prologue`/`.epilogue`
    // at MAP lowering (see `consumingFusedContract`), and this pass leaves the
    // operand alone.
    if (std::optional<TensorArgContract> contract = arg.getTraitContract();
        contract && !contract->fused) {
      // Keep the operand's own kind: a mutable-input trait argument comes in as
      // a `!mogg.buffer`, everything else as a `!mogg.tensor`.
      packed.push_back(TraitCreateOp::create(
          builder, loc, operands[i].getType(), operands[i],
          builder.getStringAttr(device.getLabel()),
          tensorArgContractToAttr(ctx, *contract)));
      changed = true;
      ++i;
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

// Wraps each operand `fn` wants as a scalar or a tuple. `allowMutToImmut` lets
// a mutable operand back a read-only kernel input (see below); only kernel
// calls opt in.
FailureOr<PackedCall>
packAgainstSignature(Operation *call, const KernelDeclFunctionAdaptor &fn,
                     StringRef callee, ValueRange inputs, ValueRange outputs,
                     M::DeviceRefAttr device, bool hasSameVariadicOperandSize,
                     bool resultLeavesByValue, bool allowMutToImmut = false) {
  ErrorOr<KernelDeclOperandMapping> mapping = mapToKernelImplementation(
      inputs, outputs, fn, callee, hasSameVariadicOperandSize,
      /*hasExtraOutputShapeArgument=*/false, resultLeavesByValue,
      allowMutToImmut);
  if (mapping.isError())
    return call->emitOpError("ConformCallOperands: ") << mapping.getError();

  OpBuilder builder(call);

  // `MAPToMOGG` reconstructs a fused mutable-load operand as a mutable
  // `mogg.tensor.create` (a `!mogg.buffer`). When it backs a read-only kernel
  // input, the mapping above accepts the mut->immut widening, but codegen would
  // still emit a mutable slice the kernel's immutable parameter rejects.
  // Recreate that producer immutable so the emitted slice matches; DCE drops
  // the now-unused mutable original.
  SmallVector<Value> reconciledInputs(inputs.begin(), inputs.end());
  bool reconciled = false;
  if (allowMutToImmut) {
    for (const auto &[idx, kernelArg] : mapping->operandMapping) {
      if (idx >= reconciledInputs.size() || kernelArg.isMutableTensorType())
        continue;
      Value operand = reconciledInputs[idx];
      if (!isa<BufferType>(operand.getType()))
        continue;
      auto producer = operand.getDefiningOp<TensorCreateInterface>();
      if (!producer || !producer.getMut())
        continue;
      builder.setInsertionPointAfter(producer);
      reconciledInputs[idx] =
          producer.cloneWithMut(builder, /*mut=*/false)->getResult(0);
      reconciled = true;
    }
    builder.setInsertionPoint(call);
  }

  PackedCall packed;
  packed.changed = reconciled;
  packed.inputs = packOperands(builder, call->getLoc(), reconciledInputs,
                               mapping->operandMapping, device, packed.changed);
  packed.outputs = packOperands(builder, call->getLoc(), outputs,
                                mapping->resultMapping, device, packed.changed);
  return packed;
}

// The execution device recorded on the enclosing `mogg.kernel`, which every
// `mogg` call lowered by `MAPToMOGG` sits inside. `mogg.trait.create` needs it
// to pick the preferred alignment and device-passable flavor at emit, locally,
// the same way `mogg.tensor.create` does.
static M::DeviceRefAttr enclosingKernelDevice(Operation *call) {
  if (auto kernel = call->getParentOfType<KernelOp>())
    return MO::getDeviceOrDefault(kernel);
  return {};
}

LogicalResult packCall(CallOpaqueOp call, const MojoRegistry &registry) {
  KernelDeclOp decl = registry.lookupKernelBySymName(call.getCallee());
  if (!decl)
    return call.emitOpError("ConformCallOperands: no kernel decl named '")
           << call.getCallee() << "'";

  KernelDeclAdaptor adaptor(decl);
  if (!adaptor.executeFunction)
    return success();

  // A kernel returning an `!mo.opaque` returns it by value, so the call carries
  // no output operand standing in for it.
  FailureOr<PackedCall> packed = packAgainstSignature(
      call, *adaptor.executeFunction, call.getCallee(), call.getInputs(),
      call.getOutputs(), enclosingKernelDevice(call),
      call.getHasSameVariadicOperandSize(),
      /*resultLeavesByValue=*/call.getOpaqueResult() != nullptr,
      /*allowMutToImmut=*/true);
  if (failed(packed))
    return failure();
  bool changed = packed->changed;

  OpBuilder builder(call);

  // A distributed kernel whose execute signature takes a `DeviceContextArray`
  // gets its per-device `!mogg.context` operands packed into one
  // tuple-of-contexts -- exactly as a variadic tensor argument is packed --
  // which emit renders as the `DeviceContextArray`. An ordinary single-context
  // kernel passes its one context through untouched.
  bool variadicContext = llvm::any_of(adaptor.executeFunction->inputArguments,
                                      [](const KernelDeclOperandAdaptor &arg) {
                                        return arg.isVariadicContextType();
                                      });
  SmallVector<Value> contexts(call.getDeviceContexts());
  if (variadicContext && !contexts.empty()) {
    auto tuple = TupleCreateOp::create(
        builder, call.getLoc(), TupleType::get(call.getContext()), contexts);
    contexts.assign({tuple.getTuple()});
    changed = true;
  }

  if (!changed)
    return success();

  // Emit reads the fusion flags rather than the operand's producer, which
  // packing has just replaced. `hasSameVariadicOperandSize` is not carried
  // over: this pass is what consumes it.
  auto repacked = CallOpaqueOp::create(
      builder, call.getLoc(), call.getResultTypes(), call.getCalleeAttr(),
      packed->inputs, packed->outputs, contexts, call.getKernelParamAttr(),
      builder.getBoolAttr(call.getHasEpilogueFusion()),
      builder.getBoolAttr(call.getHasPrologueFusion()));
  // A by-value opaque result carries the kernel's return out to the enclosing
  // `mogg.output`, so the repacked call inherits its uses.
  call.replaceAllUsesWith(repacked.getResults());
  call.erase();
  return success();
}

// Same as `packCall`, against the `mogg.shape_func.decl` signature: no
// outputs, but `range_shape` still wants `Scalar[dtype]`, not a
// `!mogg.tensor`.
LogicalResult packShapeCall(CallShapeOp call, const MojoRegistry &registry) {
  ShapeFunctionDeclOp decl =
      registry.lookupShapeFunctionBySymName(call.getCallee());
  if (!decl)
    return call.emitOpError("ConformCallOperands: no shape_func.decl named '")
           << call.getCallee() << "'";

  // A shape function returns its shape by value, so there is no output operand
  // to map.
  FailureOr<PackedCall> packed = packAgainstSignature(
      call, KernelDeclFunctionAdaptor(decl), call.getCallee(),
      call.getOperands(), /*outputs=*/{}, enclosingKernelDevice(call),
      /*hasSameVariadicOperandSize=*/false, /*resultLeavesByValue=*/true);
  if (failed(packed))
    return failure();
  if (!packed->changed)
    return success();

  OpBuilder builder(call);
  auto replacement = CallShapeOp::create(
      builder, call.getLoc(), call.getResult().getType(), call.getCalleeAttr(),
      call.getIsThrowsAttr(), packed->inputs, call.getKernelParamAttr());
  call.getResult().replaceAllUsesWith(replacement.getResult());
  call.erase();
  return success();
}

struct ConformCallOperandsPass
    : public MOGG::impl::ConformCallOperandsBase<ConformCallOperandsPass> {
  using Base::Base;

  void runOnOperation() override {
    MojoRegistry registry(getOperation());
    // Post-order: packing erases the call.
    WalkResult result = getOperation().walk([&](Operation *op) {
      LogicalResult packed = success();
      if (auto call = dyn_cast<CallOpaqueOp>(op))
        packed = packCall(call, registry);
      else if (auto shapeCall = dyn_cast<CallShapeOp>(op))
        packed = packShapeCall(shapeCall, registry);
      return failed(packed) ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace

} // namespace M::MOGG
