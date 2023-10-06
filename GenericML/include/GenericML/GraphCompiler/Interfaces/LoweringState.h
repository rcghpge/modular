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
#include "GenericML/GraphCompiler/Interfaces/ConvertibleToMGPTypeInterface.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/MGPDialect/BufferBuilder.h"
#include "GraphRT/MIPDialect/MIPOps.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
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
  MOGraphLoweringState(M::MO::GraphOp graph,
                       llvm::StringRef compiledFrameworkLabel,
                       size_t minimumCpuAlignment)
      : graph(graph), common(compiledFrameworkLabel, minimumCpuAlignment),
        init(graph.getContext()), execute(graph.getContext()) {
    // Add converters for MO native types.
    typeConverter.addConversion(
        [this](MO::TensorType type) -> std::optional<Type> {
          return getTypeBuilder(MO::getDeviceOrDefault(this->graph, type))
              ->tensorType(this->graph.getContext(),
                           MGP::createCpuTensorSpecType(type));
        });
    // Add converters for foreign types via the ConvertibleToMGPTypeInterface.
    typeConverter.addConversion([](Type type) -> std::optional<Type> {
      if (auto interface = dyn_cast<ConvertibleToMGPTypeInterface>(type))
        return interface.convertType();
      return {};
    });
  }

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

    /// This operation checks the available runtime devices match the required
    /// devices.
    MGP::CheckRuntimeOp checkRuntimeOp;

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
    TypedValue<MGP::ContextType> mgpContext;

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

  /// Returns the graph's device_specs annotation.
  FailureOr<DeviceSpecCollectionAttr> getDeviceSpecs();

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

  /// Set the MGP context for use in the 'execute' block.
  void setExecuteMgpContext(TypedValue<MGP::ContextType> mgpContext) {
    execute.mgpContext = mgpContext;
  }

  /// Return the MGP context appropriate for the 'execute' block.
  TypedValue<MGP::ContextType> getExecuteMgpContext() {
    return execute.mgpContext;
  }

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

  /// Set the check runtime op.
  void setCheckRuntimeOp(MGP::CheckRuntimeOp checkRuntimeOp) {
    init.checkRuntimeOp = checkRuntimeOp;
  }

  /// Returns the check runtime op.
  MGP::CheckRuntimeOp getCheckRuntimeOp() {
    assert(init.checkRuntimeOp && "no mgp.check.runtime op has been emitted");
    return init.checkRuntimeOp;
  }

  /// Returns the shape of a lowered tensor which is on the given device.
  TypedValue<MGP::CpuTensorSpecType> getShape(Location loc, Value loweredTensor,
                                              DeviceRefAttr device);

  /// Returns the (op which produces the) dynamic dimensions for the given
  /// lowered tensor on the given device.
  MGP::CpuTensorSpecDimsDynamicOp
  getDynamicDims(Location loc, Value loweredTensor, DeviceRefAttr device);

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

  /// Register the given type builder for the given device.
  void registerTypeBuilder(DeviceRefAttr device,
                           std::unique_ptr<TypeBuilder> typeBuilder) {
    bool inserted =
        typeBuilderMap.insert({device, std::move(typeBuilder)}).second;
    assert(inserted && "a type builder has already been registered for device");
  }

  /// Returns the type builder to use for device. The builder is owned by
  /// this object.
  TypeBuilder *getTypeBuilder(DeviceRefAttr device) {
    auto itr = typeBuilderMap.find(device);
    assert(itr != typeBuilderMap.end() &&
           "no type builder has been registered for device");
    return itr->second.get();
  }

  /// Register the given buffer builder for the given device.
  void registerBufferBuilder(DeviceRefAttr device,
                             std::unique_ptr<BufferBuilder> bufferBuilder) {
    bool inserted =
        bufferBuilderMap.insert({device, std::move(bufferBuilder)}).second;
    assert(inserted &&
           "a buffer builder has already been registered for device");
  }

  /// Returns the buffer builder to use for device. The builder is owned by
  /// this object.
  BufferBuilder *getBufferBuilder(DeviceRefAttr device) {
    assert(device);
    auto itr = bufferBuilderMap.find(device);
#if MODULAR_DEBUG
    if (itr == bufferBuilderMap.end()) {
      llvm::errs() << "no buffer builder has been registered for device "
                   << device << "\n";
    }
#endif
    assert(itr != bufferBuilderMap.end() &&
           "no buffer builder has been registered for device");
    return itr->second.get();
  }

  /// Return the 'host' device for this graph.
  DeviceRefAttr getHostDevice() { return MO::getHostDevice(graph); }

  /// Returns the buffer builder to use for the 'host' device.
  BufferBuilder *getHostBufferBuilder() {
    return getBufferBuilder(MO::getHostDevice(graph));
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
  /// Type converter for lowering from MO types to MGP and other
  /// device-specific types.
  TypeConverter typeConverter;

  /// The type builder to use for each device.
  llvm::DenseMap<DeviceRefAttr, std::unique_ptr<TypeBuilder>> typeBuilderMap;

  /// The buffer builder to use for each device.
  llvm::DenseMap<DeviceRefAttr, std::unique_ptr<BufferBuilder>>
      bufferBuilderMap;

  CommonState common;
  InitState init;
  ExecuteState execute;

  // Arbitrary client state.
  std::unique_ptr<ClientState> clientState;

  /// The state and interface for parameter lowering.
  ParameterLoweringState params;

  /// Keeps track of all values that map to parameter placeholders.
  DenseMap<Value, SmallVector<Value>> placeholderMap;
};

inline SmallVector<MO::TensorType> castToMoTensorTypes(TypeRange types) {
  return llvm::map_to_vector(types,
                             [](Type t) { return t.cast<MO::TensorType>(); });
}

//===----------------------------------------------------------------------===//
// MOOpLoweringState
//===----------------------------------------------------------------------===//

/// Helper class for managing the value-semantic to buffer-semantic transition
/// when lowering a single MO op.
struct MOOpLoweringState {
  MOGraphLoweringState &loweringState;
  OpBuilder &builder;
  mlir::Operation *opBeingLowered;
  mlir::Location loc;
  BufferBuilder *bufferBuilder;

  MOOpLoweringState(MGP::MOGraphLoweringState &loweringState,
                    mlir::Operation *op, DeviceRefAttr device)
      : loweringState(loweringState),
        builder(loweringState.getExecuteOpBuilder()), opBeingLowered(op),
        loc(op->getLoc()),
        bufferBuilder(loweringState.getBufferBuilder(device)) {}

  //
  // The following methods redirect to the loweringState.bufferBuilder, but
  // use the builder and location from this builder state.
  // However, loadKernelArchive is not implemented since it does not apply to
  // a single MO op.
  //

  Value bufferConstant(ElementsAttr value) {
    return bufferBuilder->bufferConstant(builder, loc, value);
  }
  std::pair<Value, TypedValue<CpuTensorSpecType>>
  tensorExtract(Value loweredTensor) {
    return bufferBuilder->tensorExtract(builder, loc, loweredTensor);
  }
  TypedValue<CpuTensorSpecType> tensorExtractShape(Value loweredTensor) {
    return bufferBuilder->tensorExtractShape(builder, loc, loweredTensor);
  }
  Value tensorCreate(Value inChain, Value buffer,
                     TypedValue<CpuTensorSpecType> tensorSpec) {
    return bufferBuilder->tensorCreate(builder, loc, inChain, buffer,
                                       tensorSpec);
  }
  Value bufferAllocStatic(Value inChain, uint64_t size) {
    return bufferBuilder->bufferAllocStatic(
        builder, loc, inChain, size, loweringState.getMinimumCpuAlignment());
  }
  Value bufferAllocSpec(Value inChain,
                        TypedValue<MGP::CpuTensorSpecType> tensorSpec) {
    return bufferBuilder->bufferAllocSpec(
        builder, loc, inChain, tensorSpec,
        loweringState.getMinimumCpuAlignment());
  }
  Value bufferTransferFromHost(Value inChain,
                               TypedValue<MGP::CpuBufferType> srcBuffer,
                               Value dstBuffer) {
    return bufferBuilder->bufferTransferFromHost(builder, loc, inChain,
                                                 srcBuffer, dstBuffer);
  }
  Value bufferTransferToHost(Value inChain, Value srcBuffer,
                             TypedValue<MGP::CpuBufferType> dstBuffer) {
    return bufferBuilder->bufferTransferToHost(builder, loc, inChain, srcBuffer,
                                               dstBuffer);
  }
  Value debugTensorPrint(Value inChain, Value tensor, StringAttr label) {
    return bufferBuilder->debugTensorPrint(builder, loc, inChain, tensor,
                                           label);
  }
  Value
  kernelExecute(StringRef kernelName, bool isViaHost, Value inChain,
                TypedValue<MGP::ContextType> execContext,
                SmallVector<Value> buffers,
                ArrayRef<InOutSignatureAttr::InOutSemantics> bufferSemantics,
                SmallVector<Value> inDimensions, uint64_t numOutDimensions,
                SmallVectorImpl<Value> &outDimensions) {
    return bufferBuilder->kernelExecute(
        builder, loc, kernelName, isViaHost, inChain, execContext, buffers,
        bufferSemantics, inDimensions, numOutDimensions, outDimensions);
  }

  /// Creates an operation of specific op type at the current insertion point.
  template <typename OpTy, typename... Args>
  OpTy create(Args &&...args) {
    return builder.create<OpTy, Args...>(loc, std::forward<Args>(args)...);
  }

  /// Returns the bufferized form for the MO tensor value.
  BufferizedTensor getBufferizedInput(Value moTensor);

  /// Returns the bufferized form for all of the op operands.
  SmallVector<BufferizedTensor> getBufferizedInputs();

  /// Returns the tensor spec to represent the given MO tensor type, accounting
  /// for any dynamic ranks or dimensions.
  FailureOr<TypedValue<MGP::CpuTensorSpecType>>
  createTensorSpec(MO::TensorType moTensorType);

  /// Returns the tensor specs to represent all the op's outputs.
  FailureOr<SmallVector<TypedValue<MGP::CpuTensorSpecType>>>
  createOutputTensorSpecs();

  /// Returns the bufferized form for a buffer sizes to match the given
  /// tensor spec.
  BufferizedTensor createBuffer(Value inChain,
                                TypedValue<CpuTensorSpecType> tensorSpec);

  /// Returns the bufferized form for the op's outputs.
  FailureOr<SmallVector<BufferizedTensor>> createOutputBuffers(Value inChain);

  /// Creates a tensor from the given buffer and tensor spec, and associates it
  /// with the given original MO tensor.
  void createOutputTensor(Value moTensor, Value inChain, Value buffer,
                          TypedValue<MGP::CpuTensorSpecType> tensorSpec);

  /// Create tensors matching all the op outputs with given tensor specs and
  /// buffers, and associated them with the original MO tensor results.
  void createOutputTensors(
      Value inChain, SmallVector<Value> &buffers,
      SmallVector<TypedValue<MGP::CpuTensorSpecType>> &tensorSpecs);

  /// Create tensors matching all the op outputs in bufferized form, and
  /// associates them with the original MO tensor results.
  void createOutputTensors(Value inChain,
                           SmallVector<BufferizedTensor> &bufferizedOutputs);
};

} // namespace M::MGP

#endif // GENERICML_INTERFACES_LOWERINGSTATE_H
