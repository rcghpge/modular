//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Helper for lowering MO graphs into device-dependent buffer-semantic form.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_INTERFACES_LOWERINGSTATE_H
#define GENERICML_INTERFACES_LOWERINGSTATE_H

#include "GenericML/GraphCompiler/Conversion/MGPSupport.h"
#include "GenericML/GraphCompiler/Conversion/ParameterLowering.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GraphRT/MIPDialect/MIPOps.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpDefinition.h"
#include <unordered_map>

namespace M::MGP {

//===----------------------------------------------------------------------===//
// MOGraphLoweringState
//===----------------------------------------------------------------------===//

/// This class holds the state needed to orchestrate the lowering of one
/// mo.graph operation to a mgp.model. It is shared between the lowering itself
/// (MOToPrimitives) and ops that it supports (implementers of
/// LowerableToMGPInterface). The state is gradually built as ops are lowered.
class MOGraphLoweringState {
public:
  MOGraphLoweringState(M::MO::GraphOp graph, TypeConverter &typeConverter,
                       llvm::StringRef compiledFrameworkLabel,
                       size_t minimumCpuAlignment,
                       DeviceSpecCollectionAttr deviceSpecCollectionAttr)
      : graph(graph), typeConverter(typeConverter),
        common(compiledFrameworkLabel, minimumCpuAlignment,
               deviceSpecCollectionAttr),
        init(graph.getContext()), execute(graph.getContext()) {}

  /// Common state used by both the init and execute phases.
  struct CommonState {
    explicit CommonState(llvm::StringRef compiledFrameworkLabel,
                         size_t minimumCpuAlignment,
                         DeviceSpecCollectionAttr deviceSpecCollectionAttr)
        : compiledFrameworkLabel(compiledFrameworkLabel),
          minimumCpuAlignment(minimumCpuAlignment),
          deviceSpecCollectionAttr(deviceSpecCollectionAttr) {}

    /// The top-level mgp.model operation.
    MGP::ModelOp modelOp;

    /// The name of the target framework, e.g. "mgp", "tfl", etc.
    llvm::StringRef compiledFrameworkLabel;

    /// The minimum alignment for all CPU hosted buffers (constants and
    /// allocated)
    size_t minimumCpuAlignment;

    /// Devices expected to be available at runtime.
    DeviceSpecCollectionAttr deviceSpecCollectionAttr;

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

    /// This operation creates the context for all compiled kernels in the
    /// graph within a slot of the !mgp.context.
    MGP::CpuKernelCreateContextOp compiledKernelsContext;

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

    /// Keeps track of what runtime SSA values the graph SSA values lower to.
    DenseMap<Value, Value> loweredValues;

    /// The next available constant identifier. Used to assign names
    /// to dense_resource attributes capturing MO constants.
    size_t nextConstId = 0;

    /// This is a builder appending operations to the end of the invokable.
    OpBuilder builder;

    ExecuteState(const ExecuteState &) = delete;
    void operator=(const ExecuteState &) = delete;
  };

  // Base class for clients that want to inject their own custom state.
  struct ClientState {
    virtual ~ClientState() = default;
  };

  // Check if the client state has been initialized.
  bool hasClientState() const { return bool(clientState); }

  // Initializes the client state, only to be called once.
  template <typename T>
  void initClientState(T &&initState) {
    assert(!hasClientState());
    clientState = std::make_unique<T>(std::forward<T>(initState));
  }

  // Get a reference to the stored client state.
  template <typename T>
  T &getClientState() {
    return static_cast<T &>(*clientState);
  }

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

  /// Returns information about the devices we expect at runtime.
  DeviceSpecCollectionAttr getDeviceSpecCollection() const {
    return common.deviceSpecCollectionAttr;
  }

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

  /// Return the MGP context.
  void setExecuteMgpContext(Value mgpContext) {
    execute.mgpContext = mgpContext;
  }

  /// Return the MGP context.
  Value getExecuteMgpContext() { return execute.mgpContext; }

  // Global side effect chain, currently reserved exclusively for asserts.
  // Consumers of tensor specs must block on this chain. This is not intended
  // for general side effects.
  // TODO: Remove the need for this by wiring dependencies on asserts locally.
  void setExecuteChain(Value chain) { execute.chain = chain; }

  /// Return the global side effect chain in the execute phase. Consumers of
  /// tensor specs must block on this chain.
  // TODO: Remove the need for this by wiring dependencies on asserts locally.
  Value getExecuteChain() { return execute.chain; }

  /// Return the runtime SSA value that the given graph SSA value is lowered to.
  /// The key is an arbitrary SSA value in the mo.graph, most often (but not
  /// necessarily) a `mo.tensor`.
  Value lookupExecuteLoweredValue(Value key) {
    return execute.loweredValues.lookup(key);
  }

  using ParamPlaceholderOp = ParameterLoweringState::PlaceholderOp;

  /// Register what runtime SSA value the given graph SSA value is lowered to.
  /// The key is an arbitrary SSA value in the mo.graph, most often (but not
  /// necessarily) a `mo.tensor`.
  void emplaceExecuteLoweredValue(Value key, Value value) {
    // If we are mapping to a placeholder, we remember it.
    if (auto placeholder = value.getDefiningOp<ParamPlaceholderOp>())
      placeholderMap.find(placeholder)->getSecond().emplace_back(key);

    execute.loweredValues.try_emplace(key, value);
  }

  /// Return and increment the next available constant identifier.
  size_t getExecuteNextConstId() { return execute.nextConstId++; }

  /// Set the compiled kernels context. Should only be accessed during init
  /// and referred to by the context slot in execute.
  void setCompiledKernelsContext(MGP::CpuKernelCreateContextOp op) {
    init.compiledKernelsContext = op;
  }

  /// Get the compiled kernels context. Should only be accessed during init
  /// and referred to by the context slot in execute.
  MGP::CpuKernelCreateContextOp getCompiledKernelsContext() {
    return init.compiledKernelsContext;
  }

  /// Helper to extract the dynamic dimensions for an MGP tensor.
  MGP::CpuTensorSpecDimsDynamicOp getDynamicDims(Value mgpTensor, Location loc);

  /// Helper to extract the 'shape' of an MGP tensor.
  MGP::CpuTensorExtractShapeOp getShape(Value mgpTensor, Location loc);

  /// Helper to create a new runtime tensor from the given buffer, chain, and
  /// tensor type.
  template <typename TensorTy>
  MGP::CpuTensorCreateOp
  createNewMgpTensor(Value chain, TypedValue<MGP::CpuBufferType> buffer,
                     TensorTy resTy, Location loc,
                     ValueRange runtimeDims = {}) {
    OpBuilder &exeBuilder = getExecuteOpBuilder();
    MGP::CpuTensorSpecType outSpecTy = MGP::createCpuTensorSpec(resTy);
    auto outTensorTy = exeBuilder.getType<MGP::CpuTensorType>(outSpecTy);
    auto specOp = exeBuilder.create<MGP::CpuTensorSpecCreateOp>(loc, outSpecTy,
                                                                runtimeDims);
    return exeBuilder.create<MGP::CpuTensorCreateOp>(loc, outTensorTy, chain,
                                                     buffer, specOp);
  }

  /// The given MO inArg (a !mo.tensor) has been lowered to the given MGP
  /// outArg (an !mgp.tensor). If inArg is subject to any dynamic shapes
  /// (rank or dimensions), update the parameter map to capture the MGP values
  /// they are represented by, either indexes or !mgp.cpu.tensor_specs. Returns
  /// true if any parameters were mapped.
  ErrorOr<bool> lowerArg(BlockArgument &inArg, BlockArgument &outArg);

  /// Returns a lowered value for the parameter expression attribute.
  /// Parameter references not seen before are lowered to a placeholder,
  /// parameter operators are lowered recursively, and all other attributes
  /// are delegated to the provided hook (using the execute state's builder
  /// and the provided location).
  ErrorOr<Value> lowerExpression(Location loc, TypedAttr attr);

  /// Lower a parameter declaration, resolving the placeholder if necessary. A
  /// declaration can only be lowered once, and an error is returned
  /// otherwise.
  ErrorOrSuccess lowerParamDecl(KGEN::ParamDeclAttr declAttr, Value value);

  /// Return the SSA values corresponding to the parametric dimensions in the
  /// given shape (in the order they appear in the shape).
  ErrorOr<SmallVector<Value>> lowerShapeAttrDynDims(MOSH::ShapeAttr shapeAttr,
                                                    Location loc);

  /// Track this compiled kernel as corresponding to this index in the
  /// compiled kernels context.
  void trackCompiledKernel(StringRef ref, size_t index) {
    compiledKernelsToIndex[ref.str()] = index;
  }

  /// Get the index of the kernel in the compiled kernels context.
  size_t getCompiledKernelIndex(StringRef ref) {
    auto itr = compiledKernelsToIndex.find(ref.str());
    assert(itr != compiledKernelsToIndex.end() &&
           "All kernels must be accounted for through lowering the load "
           "kernels.");
    return itr->second;
  }

private:
  //===-----------------------------------------------------------------===//
  // Parameter lowering helpers.
  //===-----------------------------------------------------------------===//

  /// Maps KGEN::POC opcodes to mip.index.* op names.
  static ErrorOr<StringRef> pocToMopIndexOp(KGEN::POC poc);

  /// Lowers an operator attribute to the appropriate mop index op.
  static ErrorOr<Value> lowerParamOperator(OpBuilder &builder, Location loc,
                                           KGEN::ParamOperatorAttr opAttr,
                                           ValueRange operands);

  /// Lowers an integer attribute to a mip constant op.
  static Operation *lowerIntAttr(OpBuilder &builder, Location loc,
                                 IntegerAttr intAttr);

  /// Lower a parameter expression.
  static ErrorOr<Value> lowerExpressionImpl(ParameterLoweringState &state,
                                            OpBuilder &builder, Location loc,
                                            TypedAttr attr,
                                            ValueRange operands);

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

  // Arbitrary client state.
  std::unique_ptr<ClientState> clientState;

  /// A map of the name of the compiled kernel to their index in the
  /// compiled kernel context.
  std::unordered_map<std::string, size_t> compiledKernelsToIndex;

  /// The state and interface for parameter lowering.
  ParameterLoweringState params;

  /// Keeps track of all values that map to parameter placeholders.
  DenseMap<Value, SmallVector<Value>> placeholderMap;
};

} // namespace M::MGP

#endif // GENERICML_INTERFACES_LOWERINGSTATE_H
