//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOTOPRIMITIVES_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOTOPRIMITIVES_H

#include "GenericML/GraphCompiler/Conversion/MGPSupport.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "Support/AlignedAlloc.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace M::MGP {

/// Generate the code for declaring the pass.
#define GEN_PASS_DECL_MOTOPRIMITIVES
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

/// Generate the code for registering the pass.
#define GEN_PASS_REGISTRATION
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

/// This class holds the state needed to orchestrate the lowering of one
/// mo.graph operation to a mgp.model. It is shared between the lowering itself
/// (MOToPrimitives) and ops that it supports (implementers of
/// LowerableToMGPInterface). The state is gradually built as ops are lowered.
class MOGraphLoweringState {
public:
  MOGraphLoweringState(M::MO::GraphOp graph, TypeConverter &typeConverter,
                       llvm::StringRef compiledFrameworkLabel,
                       size_t minimumCpuAlignment)
      : graph(graph), typeConverter(typeConverter),
        common(compiledFrameworkLabel, minimumCpuAlignment),
        init(graph.getContext()), execute(graph.getContext()) {}

  /// Common state used by both the init and execute phases.
  struct CommonState {
    explicit CommonState(llvm::StringRef compiledFrameworkLabel,
                         size_t minimumCpuAlignment)
        : compiledFrameworkLabel(compiledFrameworkLabel),
          minimumCpuAlignment(minimumCpuAlignment) {}

    /// The top-level mgp.model operation.
    MGP::ModelOp modelOp;

    /// The name of the target framework, e.g. "mgp", "tfl", etc.
    llvm::StringRef compiledFrameworkLabel;

    /// The minimum alignment for all CPU hosted buffers (constants and
    /// allocated)
    size_t minimumCpuAlignment;

    CommonState(const CommonState &) = delete;
    void operator=(const CommonState &) = delete;
  };

  /// State used when building an invokable for the 'init' phase.
  struct InitState {
    InitState(MLIRContext *ctx) : builder(ctx) {}

    /// This is the current chain value being added to.
    Value chain;

    /// This op creates and returns the overall MGP context.
    MGP::ContextCreateOp mgpContext;

    /// This operation is the context for all compiled code in the graph.
    /// The current implementation stores the context as a state slot within the
    /// mgp context.
    mlir::Operation *jitContext;

    /// The slot inside the MGP context which contains the jit context.
    uint32_t jitContextSlot;

    /// This is a builder appending operations to the end of the invokable.
    OpBuilder builder;

    InitState(const InitState &) = delete;
    void operator=(const InitState &) = delete;
  };

  /// State used when building an invokable for the 'execute' phase.
  struct ExecuteState {
    ExecuteState(MLIRContext *ctx) : builder(ctx) {}

    /// The original input chain to the invokable, ready when the argument
    /// tensors are ready.
    Value invokableInputChain;

    /// The "main thread" chain in the execute phase.
    Value chain;

    /// The MGP context, passed in as the second argument.
    Value mgpContext;

    /// Keeps track of what mo.tensor SSA values are lowered to.
    DenseMap<Value, Value> loweredTensors;

    /// Tracks what parameters are lowered to.
    DenseMap<StringAttr, Value> loweredParams;

    /// The next available constant identifier. Used to assign names
    /// to dense_resource attributes capturing MO constants.
    size_t nextConstId = 0;

    /// This is a builder appending operations to the end of the invokable.
    OpBuilder builder;

    ExecuteState(const ExecuteState &) = delete;
    void operator=(const ExecuteState &) = delete;
  };

  /// Return the top-level mgp.model operation.
  MGP::ModelOp getModelOp() { return common.modelOp; }

  /// Set the top-level mgp.model operation.
  void setModelOp(MGP::ModelOp modelOp) { common.modelOp = modelOp; }

  llvm::StringRef getCompiledFrameworkLabel() {
    return common.compiledFrameworkLabel;
  }

  size_t getMinimumCpuAlignment() { return common.minimumCpuAlignment; }

  /// Return the graph being lowered.
  MO::GraphOp &getGraph() { return graph; }

  /// Return type converter for lowering known MO types into MGP ones.
  TypeConverter &getTypeConverter() { return typeConverter; }

  /// Return the current chain value being added to.
  Value getInitChain() { return init.chain; }

  /// Set the current chain value being added to.
  void setInitChain(Value chain) { init.chain = chain; }

  /// Return the builder for appending operations to the end of the invokable.
  OpBuilder &getInitOpBuilder() { return init.builder; }

  /// Set op to create and return the overall MGP context.
  void setInitMgpContext(MGP::ContextCreateOp mgpContext) {
    init.mgpContext = mgpContext;
  }

  /// return op to create and return the overall MGP context.
  MGP::ContextCreateOp &getInitMgpContext() { return init.mgpContext; }

  /// Return the builder for appending operations to the end of the invokable.
  OpBuilder &getExecuteOpBuilder() { return execute.builder; }

  /// Set the original input chain to the invokable, ready when the argument
  /// tensors are ready.
  void setExecuteInvokableInputChain(Value invokableInputChain) {
    execute.invokableInputChain = invokableInputChain;
  }

  /// Return the original input chain to the invokable, ready when the argument
  /// tensors are ready.
  Value getExecuteInvokableInputChain() { return execute.invokableInputChain; }

  /// Return the MGP context.
  void setExecuteMgpContext(Value mgpContext) {
    execute.mgpContext = mgpContext;
  }

  /// Return the MGP context.
  Value getExecuteMgpContext() { return execute.mgpContext; }

  /// Set the "main thread" chain in the execute phase.
  void setExecuteChain(Value chain) { execute.chain = chain; }

  /// Return the "main thread" chain in the execute phase.
  Value getExecuteChain() { return execute.chain; }

  /// Return what mo.tensor the SSA value is lowered to.
  Value lookupExecuteLoweredTensor(Value key) {
    return execute.loweredTensors.lookup(key);
  }

  /// Register what mo.tensor the SSA value (key) is lowered to.
  void emplaceExecuteLoweredTensor(Value key, Value value) {
    execute.loweredTensors.try_emplace(key, value);
  }

  /// Return what the parameter with the given name is lowered to.
  Value lookupExecuteLoweredParam(StringAttr name) {
    return execute.loweredParams.lookup(name);
  }

  /// Register what the parameter with the given name is lowered to.
  void emplaceExecuteLoweredParam(StringAttr name, Value value) {
    execute.loweredParams.try_emplace(name, value);
  }

  /// Return the SSA values corresponding to the parametric dimensions in the
  /// given shape (in the order they appear in the shape).
  SmallVector<Value> lookupExecuteDynDims(MO::ShapeAttr shapeAttr);

  /// Return and increment the next available constant identifier.
  size_t getExecuteNextConstId() { return execute.nextConstId++; }

  /// Set the jit context. Should only be accessed during init and referred to
  /// by the context slot in execute.
  void setJitCtx(mlir::Operation *op) { init.jitContext = op; }

  /// Get the jit context. Should only be accessed during init and referred to
  /// by the context slot in execute.
  mlir::Operation *getJitCtx() { return init.jitContext; }

  /// The JIT context is stored in one of the "slots" of the context. This sets
  /// the index of that slot.
  void setJitCtxSlot(uint32_t slot) { init.jitContextSlot = slot; }

  /// The JIT context is stored in one of the "slots" of the context. This
  /// returns the index of the slot which contains it.
  uint32_t getJitCtxSlot() { return init.jitContextSlot; }

private:
  //===-----------------------------------------------------------------===//
  // Lowering state
  //===-----------------------------------------------------------------===//
  /// The graph being lowered.
  MO::GraphOp graph;
  /// Type converter for lowering known MO types into MGP ones.
  TypeConverter &typeConverter;

  CommonState common;
  InitState init;
  ExecuteState execute;
};

} // namespace M::MGP

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTOPRIMITIVES_H
