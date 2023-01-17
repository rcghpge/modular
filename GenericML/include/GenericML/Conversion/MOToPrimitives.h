//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_CONVERSION_MOTOPRIMITIVES_H
#define GENERICML_CONVERSION_MOTOPRIMITIVES_H

#include "GenericML/Conversion/Support.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MODialect/MOOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace M::MGP {

/// Generate the code for declaring the pass.
#define GEN_PASS_DECL_MOTOPRIMITIVES
#include "GenericML/Conversion/MOToPrimitives.h.inc"

/// Generate the code for registering the pass.
#define GEN_PASS_REGISTRATION
#include "GenericML/Conversion/MOToPrimitives.h.inc"

/// This class holds the state needed to orchestrate the lowering of one
/// mo.graph operation to a mgp.model. It is shared between the lowering itself
/// (MOToPrimitives) and ops that it supports (implementers of
/// MOLoweringInterface). The state is gradually built as ops are lowered.
class MOGraphLoweringState {
public:
  MOGraphLoweringState(M::MO::GraphOp graph, TypeConverter &typeConverter,
                       llvm::StringRef compiledFrameworkLabel)
      : graph(graph), typeConverter(typeConverter),
        common(compiledFrameworkLabel), init(graph.getContext()),
        execute(graph.getContext()) {}

  /// Common state used by both the init and execute phases.
  struct CommonState {
    explicit CommonState(llvm::StringRef compiledFrameworkLabel)
        : compiledFrameworkLabel(compiledFrameworkLabel) {}

    /// The top-level mgp.model operation.
    MGP::ModelOp modelOp;

    /// The name of the target framework, e.g. "mgp", "tfl", etc.
    llvm::StringRef compiledFrameworkLabel;

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

    /// The next available constant identifier. Used to assign names
    /// to dense_resource attributes capturing MO constants.
    size_t nextConstId = 0;

    /// This is a builder appending operations to the end of the invokable.
    OpBuilder builder;

    ExecuteState(const ExecuteState &) = delete;
    void operator=(const ExecuteState &) = delete;
  };

  MGP::ModelOp getModelOp() { return common.modelOp; }

  llvm::StringRef getCompiledFrameworkLabel() {
    return common.compiledFrameworkLabel;
  }

  /// Return the graph being lowered.
  MO::GraphOp &getGraph() { return graph; }

  /// Return type converter for lowering known MO types into MGP ones.
  TypeConverter &getTypeConverter() { return typeConverter; }

  /// Set the top-level mgp.model operation.
  void setCommonModelOp(MGP::ModelOp modelOp) { common.modelOp = modelOp; }

  /// Return the top-level mgp.model operation.
  MGP::ModelOp &getCommonModelOp() { return common.modelOp; }

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

  /// Return and increment the next available constant identifier.
  size_t getExecuteNextConstId() { return execute.nextConstId++; }

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

#endif // GENERICML_CONVERSION_MOTOPRIMITIVES_H
