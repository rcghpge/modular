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
#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/MGPDialect/BufferBuilder.h"
#include "GenericML/Support/Diagnostic.h"
#include "GraphRT/MIPDialect/MIPOps.h"
#include "Support/AssertStream.h"
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
    builderContext.hostDevice = MO::getHostDevice(graph);

    // Add converters for MO native types.
    typeConverter.addConversion(
        [this](MO::TensorType type) -> std::optional<Type> {
          return getTypeBuilder(MO::getDeviceOrDefault(this->graph, type))
              ->tensorType(this->graph.getContext(),
                           MGP::createCpuTensorSpecType(type));
        });
    typeConverter.addConversion([this](
                                    MO::ListType type) -> std::optional<Type> {
      auto tensorTy = dyn_cast<MO::TensorType>(type.getElementType());
      // TODO(#29947): Support other element types
      ASSERT_STREAM(tensorTy) << "only support lowering list of tensors";

      MLIRContext *ctx = type.getContext();
      MGP::TensorSpecType tensorSpecTy = MGP::createCpuTensorSpecType(tensorTy);
      DeviceRefAttr tensorDeviceRef =
          MO::getDeviceOrDefault(this->getGraph(), tensorTy);
      MGP::TypeBuilder *typeBuilder = getTypeBuilder(tensorDeviceRef);
      auto mgpTensorTy = typeBuilder->tensorType(ctx, tensorSpecTy);
      return cast<Type>(MGP::ListType::get(ctx, mgpTensorTy));
    });
    typeConverter.addConversion([](MO::ScalarType type) -> std::optional<Type> {
      auto errOrDtype = MO::toMLIRType(type.getContext(), type.getDtype());

      if (errOrDtype.isError()) {
        return std::nullopt;
      }

      return *errOrDtype;
    });

    typeConverter.addConversion([](MO::OpaqueType type) -> std::optional<Type> {
      return OpaqueType::get(type.getContext(), std::nullopt);
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

    /// The name of the target framework, e.g. "mgp", "tf", etc.
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

    /// The input chain passed as the first argument to the 'execute' block,
    /// or an equivalent SSA value in a control flow sub-block.
    TypedValue<MEF::ChainType> rootChain;

    /// The current side-effect chain, used to strictly sequence side-effecting
    /// operations such as for printing.
    TypedValue<MEF::ChainType> currentChain;

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
    ASSERT_STREAM(!hasClientState());
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
  CompileDiagnosticOr<DeviceSpecCollectionAttr> getDeviceSpecs();

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
  void setExecuteMgpContext(Value mgpContext) {
    execute.mgpContext = cast<TypedValue<MGP::ContextType>>(mgpContext);
  }

  /// Return the MGP context appropriate for the 'execute' block.
  TypedValue<MGP::ContextType> getExecuteMgpContext() {
    return execute.mgpContext;
  }

  /// Sets the 'root' chain for the 'execute' block to chain.
  void setRootExecuteChain(Value chain) {
    execute.rootChain = cast<TypedValue<MEF::ChainType>>(chain);
  }

  /// Returns the 'root' chain of the 'execute' block. This can be used for
  /// all non-side-effecting operations.
  ///
  /// May not be the same as getCurrentExecuteChain if the chain has been
  /// updated in order to enforce the ordering of side effecting operations,
  /// such as printing.
  TypedValue<MEF::ChainType> getExecuteChain() { return execute.rootChain; }

  /// Returns the 'current' chain of the 'execute' block. This can be used for
  /// all side-effecting operations.
  Value getCurrentExecuteChain() { return execute.currentChain; }

  /// Updates the 'current' chain of the 'execute' block.
  void setCurrentExecuteChain(Value chain) {
    execute.currentChain = cast<TypedValue<MEF::ChainType>>(chain);
  }

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
    ASSERT_STREAM(init.checkRuntimeOp)
        << "no mgp.check.runtime op has been emitted";
    return init.checkRuntimeOp;
  }

  /// Returns the shape of a lowered tensor which is on the given device.
  TypedValue<MGP::TensorSpecType> getShape(Location loc, Value loweredTensor,
                                           DeviceRefAttr device);

  /// Returns the (op which produces the) dynamic dimensions for the given
  /// lowered tensor on the given device.
  MGP::CpuTensorSpecDimsDynamicOp
  getDynamicDims(Location loc, Value loweredTensor, DeviceRefAttr device);

  /// The given MO inArg (a !mo.tensor) has been lowered to the given MGP
  /// outArg (an !mgp.tensor). If inArg is subject to any dynamic shapes
  /// (rank or dimensions), update the parameter map to capture the MGP values
  /// they are represented by, either indexes or !mgp.tensor_specs. Returns
  /// true if any parameters were mapped.
  ErrorOr<bool> lowerArg(Region *parentRegion, BlockArgument &inArg,
                         BlockArgument &outArg);

  /// Returns a lowered value for the parameter expression attribute.
  /// Parameter references not seen before are lowered to a placeholder,
  /// parameter operators are lowered recursively, and all other attributes
  /// are delegated to the provided hook (using the execute state's builder
  /// and the provided location).
  ErrorOr<Value> lowerExpression(Region *parentRegion, Location loc,
                                 TypedAttr attr);

  /// Lower a parameter declaration, resolving the placeholder if necessary. A
  /// declaration can only be lowered once, and an error is returned
  /// otherwise.
  ErrorOrSuccess lowerParamDecl(Region *parentRegion,
                                KGEN::ParamDeclAttr declAttr, Value value);

  /// Return the SSA values corresponding to the parametric dimensions in the
  /// given shape (in the order they appear in the shape).
  ErrorOr<SmallVector<Value>> lowerShapeAttrDynDims(Region *parentRegion,
                                                    MOSH::ShapeAttr shapeAttr,
                                                    Location loc);

  /// Extract the static dimensions from a shape attribute and return their
  /// product. Returns one if there are no static dimensions.
  /// For example, a shape of 2x3x?x1 would return 6.
  int64_t extractShapeAttrStaticDimsProduct(MOSH::ShapeAttr shapeAttr,
                                            Location loc);

  const BufferBuilderContext &getBufferBuilderContext() const {
    return builderContext;
  }

  /// Register the given type builder for the given device.
  void registerTypeBuilder(DeviceRefAttr device,
                           std::unique_ptr<TypeBuilder> typeBuilder) {
    builderContext.registerTypeBuilder(device, std::move(typeBuilder));
  }

  /// Returns the type builder to use for device. The builder is owned by
  /// this object.
  TypeBuilder *getTypeBuilder(DeviceRefAttr device) {
    return builderContext.getTypeBuilder(device);
  }

  /// Register the given buffer builder for the given device.
  void registerBufferBuilder(DeviceRefAttr device,
                             std::unique_ptr<BufferBuilder> bufferBuilder) {
    builderContext.registerBufferBuilder(device, std::move(bufferBuilder));
  }

  /// Returns the buffer builder to use for device. The builder is owned by
  /// this object.
  BufferBuilder *getBufferBuilder(DeviceRefAttr device) {
    return builderContext.getBufferBuilder(device);
  }

  /// Return the 'host' device for this graph.
  DeviceRefAttr getHostDevice() { return builderContext.hostDevice; }

  /// Returns the buffer builder to use for the 'host' device.
  BufferBuilder *getHostBufferBuilder() {
    return builderContext.getHostBufferBuilder();
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
  ErrorOr<Value> lowerExpressionImpl(ParameterLoweringState &state,
                                     OpBuilder &builder, Location loc,
                                     Region *parentRegion, TypedAttr attr,
                                     ValueRange operands);

  //===-----------------------------------------------------------------===//
  // Lowering state
  //===-----------------------------------------------------------------===//

  /// The graph being lowered.
  MO::GraphOp graph;
  /// Type converter for lowering from MO types to MGP and other
  /// device-specific types.
  TypeConverter typeConverter;

  /// All the type and buffer builders we've registered.
  BufferBuilderContext builderContext;

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
                             [](Type t) { return cast<MO::TensorType>(t); });
}

//===----------------------------------------------------------------------===//
// MOOpLoweringState
//===----------------------------------------------------------------------===//

/// Currently only !mo.tensors may be returned from MO ops. As we add
/// support for additional types, this type will need to change to be
/// a variant for all output storage + additional metadata representations
/// we need to manage the DPS convention.
using MOOutputStorage = BufferizedTensor;

/// Helper class for managing the value-semantic to buffer-semantic transition
/// when lowering a single MO op.
struct MOOpLoweringState {
  MOGraphLoweringState &loweringState;
  OpBuilder &builder;
  mlir::Operation *opBeingLowered;
  mlir::Location loc;
  TypedValue<MGP::ContextType> context;
  uint64_t minimumCpuAlignment;

  MOOpLoweringState(MGP::MOGraphLoweringState &loweringState,
                    mlir::Operation *op)
      : loweringState(loweringState),
        builder(loweringState.getExecuteOpBuilder()), opBeingLowered(op),
        loc(op->getLoc()), context(loweringState.getExecuteMgpContext()),
        minimumCpuAlignment(loweringState.getMinimumCpuAlignment()) {}

  //===-----------------------------------------------------------------===//
  // Main API
  //===-----------------------------------------------------------------===//

  /// Returns the lowered form for all of the op's operands:
  /// - !mo.tensor operands are passed as the buffer extracted from the lowered
  ///   tensors. Both the buffer and tensor will match the storage device
  ///   associated with the !mo.tensor type.
  /// - All other operands are passed in the lowered form directly.
  SmallVector<Value> getLoweredOperands();

  /// Returns the storage for all of the op's results, which will be passed in
  /// destination-passing-style.
  ///  - !mo.tensor results are passed as a freshly allocated buffer. The buffer
  ///    will match the storage device associated with the !mo.tensor result.
  ///  - A tensor spec will also be created, and paired with the buffer in the
  ///    results.
  ///  - No other types are currently supported in output form.
  FailureOr<SmallVector<MOOutputStorage>> createOutputStorage(Value inChain);

  /// Call the given kernel, which should execute on the same device as the
  /// op being lowered.
  Value
  kernelExecute(StringRef kernelName, Value inChain, ArrayRef<Value> args,
                ArrayRef<InOutSignatureAttr::InOutSemantics> inOutSemantics,
                ArrayRef<Value> inDimensions, uint64_t numOutDimensions,
                SmallVectorImpl<Value> &outDimensions, StringRef summary,
                bool isAsync);

  /// Create the outputs from the storage and additional metadata created for
  /// the call. Associate the outputs with the original MO op results.
  void createOutputs(Value inChain,
                     SmallVector<MOOutputStorage> &loweredOutputsStorage);

  //===-----------------------------------------------------------------===//
  // Input and Output helpers
  //===-----------------------------------------------------------------===//

  /// Returns the bufferized form for the MO tensor value.
  BufferizedTensor getBufferizedInput(Value moTensor);

  /// Returns the tensor spec to represent the given MO tensor type, accounting
  /// for any dynamic ranks or dimensions.
  FailureOr<TypedValue<MGP::TensorSpecType>>
  createTensorSpec(MO::TensorType moTensorType);

  /// Returns the tensor specs to represent all the op's outputs.
  FailureOr<SmallVector<TypedValue<MGP::TensorSpecType>>>
  createOutputTensorSpecs();

  /// Returns the bufferized form for a buffer sizes to match the given
  /// tensor spec and storage device of moTensorType.
  FailureOr<BufferizedTensor>
  createBuffer(Value inChain, TypedValue<TensorSpecType> tensorSpec,
               Type moTensorType);

  /// Creates a tensor from the given buffer and tensor spec, and associates it
  /// with the given original MO tensor. The tensor will be created on the
  /// storage device of moTensor.
  void createOutputTensor(Value moTensor, Value inChain, Value buffer,
                          TypedValue<MGP::TensorSpecType> tensorSpec);

  /// Create tensors matching all the op outputs with given tensor specs and
  /// buffers, and associated them with the original MO tensor results.
  void createOutputTensors(
      Value inChain, SmallVector<Value> &buffers,
      SmallVector<TypedValue<MGP::TensorSpecType>> &tensorSpecs);

  //===-----------------------------------------------------------------===//
  // MLIR create helpers
  //===-----------------------------------------------------------------===//

  /// Creates an operation of specific op type at the current insertion point.
  template <typename OpTy, typename... Args>
  OpTy create(Args &&...args) {
    return builder.create<OpTy, Args...>(loc, std::forward<Args>(args)...);
  }

  /// Returns the buffer builder appropriate for the device on which the
  /// given !mo.tensor should be stored.
  BufferBuilder *getStorageBufferBuilder(Type moTensorType);

  /// Returns the type builder appropriate for the device on which the
  /// given !mo.tensor should be stored.
  TypeBuilder *getStorageTypeBuilder(Type moTensorType);

  //
  // These all redirect to the similarly named BufferBuilder method using
  // the builder appropriate for the storage device of the given moTensorType.
  //

  std::pair<Value, TypedValue<TensorSpecType>>
  tensorExtract(Value loweredTensor, Type moTensorType) {
    BufferBuilder *bufferBuilder = getStorageBufferBuilder(moTensorType);
    // Be careful the evaluation order of the function args is unspecified
    // in the c++ standard. Define them separately to enforce the op order.
    auto buffer = bufferBuilder->tensorExtractBuffer(builder, loc, context,
                                                     loweredTensor);
    auto spec = bufferBuilder->tensorExtractTensorSpec(builder, loc, context,
                                                       loweredTensor);
    return std::make_pair(buffer, spec);
  }

  Value tensorCreate(Value inChain, Value buffer,
                     TypedValue<TensorSpecType> tensorSpec, Type moTensorType) {
    return getStorageBufferBuilder(moTensorType)
        ->tensorCreate(builder, loc, context, inChain, buffer, tensorSpec);
  }

  Value bufferConstant(TypedValue<MEF::ChainType> inChain, ElementsAttr value,
                       Type moTensorType) {
    return getStorageBufferBuilder(moTensorType)
        ->bufferConstant(builder, loc, inChain, context, value);
  }

  Value debugTensorPrint(Value inChain, Value buffer,
                         TypedValue<TensorSpecType> tensorSpec,
                         StringAttr label, Type moTensorType) {
    return getStorageBufferBuilder(moTensorType)
        ->debugTensorPrint(builder, loc, context, inChain, buffer, tensorSpec,
                           label);
  }

  Value bufferTransferFromHost(Value inChain,
                               TypedValue<MGP::DeviceBufferType> srcBuffer,
                               Value dstBuffer, Type dstMoTensorType) {
    return getStorageBufferBuilder(dstMoTensorType)
        ->bufferTransferFromHost(builder, loc, context, inChain,
                                 loweringState.getHostBufferBuilder(),
                                 srcBuffer, dstBuffer);
  }
  Value bufferTransferToHost(Value inChain, Value srcBuffer,
                             TypedValue<MGP::DeviceBufferType> dstBuffer,
                             Type srcMoTensorType) {
    return getStorageBufferBuilder(srcMoTensorType)
        ->bufferTransferToHost(builder, loc, context, inChain, srcBuffer,
                               loweringState.getHostBufferBuilder(), dstBuffer);
  }
  Value bufferTransferDevice(Value inChain, Value srcBuffer, Value dstBuffer,
                             Type srcMoTensorType, Type dstMoTensorType) {
    return getStorageBufferBuilder(srcMoTensorType)
        ->bufferTransferDevice(builder, loc, context, inChain, srcBuffer,
                               getStorageBufferBuilder(dstMoTensorType),
                               dstBuffer);
  }

  Value bufferConcat(Value inChain, Value outBuffer, ArrayRef<Value> inBuffers,
                     Type outMoTensorType) {
    return getStorageBufferBuilder(outMoTensorType)
        ->bufferConcat(builder, loc, context, inChain, outBuffer, inBuffers);
  }

  SmallVector<Value>
  splitBorrow(Value inChain, Value inBuffer, ArrayRef<Value> runtimeSizes,
              ArrayRef<std::optional<uint64_t>> outBufferSizes,
              Type moTensorType) {
    return getStorageBufferBuilder(moTensorType)
        ->splitBorrow(builder, loc, context, inChain, inBuffer, runtimeSizes,
                      outBufferSizes);
  }

  Value bufferToBool(Value inChain, Value buffer, Type moTensorType) {
    return getStorageBufferBuilder(moTensorType)
        ->bufferToBool(builder, loc, context, inChain, buffer);
  }

  Value bufferSetWithIndex(Value buffer, ArrayRef<Value> values,
                           Type moTensorType) {
    return getStorageBufferBuilder(moTensorType)
        ->bufferSetWithIndex(builder, loc, context, buffer, values);
  }
};

} // namespace M::MGP

#endif // GENERICML_INTERFACES_LOWERINGSTATE_H
