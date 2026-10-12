//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Conforms each `mogg` call's operands to the callee's declared signature.
// `MAPToMOGG` leaves plain `!mogg.tensor` operands; this pass re-runs operand
// inference against the callee decl and wraps scalars in `mogg.scalar.create`,
// variadic runs in `mogg.tuple.create`, and trait-bound args in
// `mogg.trait.create`, so each `::emitMojo` stays a local, 1:1 lowering.
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

// Packs one operand list (a call's inputs or outputs) per `mapping`: a scalar
// kernel argument's `!mogg.tensor` is wrapped in `mogg.scalar.create`; a legacy
// variadic argument's consecutive `!mogg.tensor` operands go into one
// `mogg.tuple.create`; a trait argument, including each element of a
// `TensorTuple`, gets a `mogg.trait.create`, and a `TensorTuple`'s run is then
// packed as well. Sets `changed` on any rewrite.
SmallVector<Value>
packOperands(OpBuilder &builder, Location loc, ValueRange operands,
             const KernelDeclIndexToOperandAdaptorMapping &mapping,
             M::DeviceRefAttr device, bool &changed) {
  MLIRContext *ctx = builder.getContext();

  // A NON-fused trait-bound tensor argument (`Tensor` / `DenseTensor` /
  // `TileTensorable`) has its raw `!mogg.tensor` re-presented as the
  // conforming trait argument by `mogg.trait.create`, which carries the
  // distilled contract and the execution device (for the emit-side layout /
  // alignment). A FUSED trait argument is handled at its producer instead: its
  // fusion functor is a local of the prologue/epilogue emit, so the contract is
  // stamped onto the `mogg.tensor.create.prologue`/`.epilogue` at MAP lowering
  // (see `consumingFusedContract`), and this pass leaves the operand alone.
  auto asTraitArg = [&](Value operand,
                        const TensorArgContract &contract) -> Value {
    if (contract.fused)
      return operand;
    changed = true;
    // Keep the operand's own kind: a mutable-input trait argument comes in as
    // a `!mogg.buffer`, everything else as a `!mogg.tensor`.
    return TraitCreateOp::create(builder, loc, operand.getType(), operand,
                                 builder.getStringAttr(device.getLabel()),
                                 tensorArgContractToAttr(ctx, contract));
  };

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
      // Record the ABI the callee's variadic parameter wants, so emit picks the
      // aggregate without inspecting the run's producing ops. For the legacy
      // lane, the variant's own IOSpec (not store mutability -- a fused output
      // is not a `MutableInputTensor`) distinguishes the fused cases; a
      // slot-backed (`StableVariadicTensors`) parameter packs its
      // `StableTensor` elements.
      VariadicABI abi = VariadicABI::Plain;
      if (arg.isTraitTensorTupleType()) {
        abi = VariadicABI::TensorTuple;
        for (Value &element : run)
          element = asTraitArg(element, *arg.getTraitContract());
      } else if (const auto *v = std::get_if<KernelDeclVariadicTensorAdaptor>(
                     &arg.underlyingType)) {
        if (v->stable)
          abi = VariadicABI::Stable;
        else if (v->ioSpec == IOSpec::FusedInputTensor)
          abi = VariadicABI::FusedInput;
        else if (v->ioSpec == IOSpec::FusedOutputTensor)
          abi = VariadicABI::FusedOutput;
      }
      packed.push_back(
          TupleCreateOp::create(builder, loc, TupleType::get(ctx), run, abi));
      changed = true;
      continue;
    }

    if (std::optional<TensorArgContract> contract = arg.getTraitContract()) {
      packed.push_back(asTraitArg(operands[i], *contract));
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

  // `MAPToMOGG` rebuilds a fused mutable-load operand as a mutable
  // `mogg.tensor.create`. Backing a read-only input, the mapping accepts
  // mut->immut, but codegen would emit a mutable slice the immutable parameter
  // rejects, so recreate the producer immutable (DCE drops the original).
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

// The enclosing `mogg.routine`'s execution device, which `mogg.trait.create`
// needs at emit to pick alignment and device-passable flavor, as
// `mogg.tensor.create` does.
static M::DeviceRefAttr enclosingKernelDevice(Operation *call) {
  if (auto kernel = call->getParentOfType<RoutineOp>())
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

  // A distributed kernel taking a `DeviceContextArray` gets its per-device
  // `!mogg.context` operands packed into one tuple, as for a variadic tensor
  // argument.
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
