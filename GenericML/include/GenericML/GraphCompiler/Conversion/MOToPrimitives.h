//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOTOPRIMITIVES_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOTOPRIMITIVES_H

#include "GenericML/GraphCompiler/Conversion/MGPSupport.h"
#include "GenericML/GraphCompiler/Conversion/ParameterLowering.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GraphRT/MIPDialect/MIPOps.h"
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

  /// Helper to extract the dynamic dimensions for an MGP tensor.
  MGP::CpuTensorSpecDimsDynamicOp getDynamicDims(Value mgpTensor,
                                                 Location loc) {
    OpBuilder &exeBuilder = getExecuteOpBuilder();
    auto extractOp = exeBuilder.create<MGP::CpuTensorExtractOp>(loc, mgpTensor);
    TypedValue<MGP::CpuTensorSpecType> spec = extractOp.getTensorSpecResult();
    return exeBuilder.create<MGP::CpuTensorSpecDimsDynamicOp>(
        loc, spec, spec.getType().getRawDims());
  }

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

  /// Lower the given graph argument using the provided builder and hook. The
  /// hook (using the execute state's builder and the given argument) is
  /// responsible for updating the value mapping.
  ErrorOr<bool> lowerArg(BlockArgument &arg, BlockArgument &exeArg) {
    bool hasParams = false;
    Location loc = arg.getLoc();
    if (auto tensorArg = dyn_cast<MO::TensorValue>(arg)) {
      MO::TensorType argType = tensorArg.getType();
      // We check if the tensor is parameterized. If it has unknown shape/rank
      // or static shape, we skip it.
      if (!argType.hasUnknownShape() && !argType.hasStaticShape()) {
        // Dynamic ranks are currently not supported.
        if (isa<KGEN::ParamDeclRefAttr>(argType.getShapeAttr()))
          return Error("Shape parameters are not supported");

        // At this point, we must be dealing with a statically ranked,
        // parameterized argument. We lower to some MGP ops, but only update
        // mappings the first time we encounter a parameter. The unused MGP ops
        // can be cleaned up using DCE.
        hasParams = true;
        MGP::CpuTensorSpecDimsDynamicOp dynDims = getDynamicDims(exeArg, loc);

        // The semantics of CpuTensorSpecDimsDynamicOp are quirky: it returns
        // only dynamic runtime dimensions, so we filter for parametric
        // dimensions in the argument shape. Notably, we do not allow general
        // expressions in the signature (yet).
        SmallVector<KGEN::ParamDeclRefAttr> paramDims;
        for (TypedAttr dim : argType.getDims())
          if (auto pDim = dyn_cast<KGEN::ParamDeclRefAttr>(dim))
            paramDims.emplace_back(pDim);
        assert(dynDims->getNumResults() == paramDims.size());

        // Only update the parameter mapping the first time.
        for (auto [pDim, dimValue] :
             llvm::zip(paramDims, dynDims->getResults()))
          if (!params.lookupLowered(pDim))
            params.emplaceLowered(pDim, dimValue);
      }
    }
    return hasParams;
  }

  /// Returns a lowered value for the parameter expression attribute. Parameter
  /// references not seen before are lowered to a placeholder, parameter
  /// operators are lowered recursively, and all other attributes are delegated
  /// to the provided hook (using the execute state's builder and the provided
  /// location).
  ErrorOr<Value> lowerExpression(Location loc, TypedAttr attr) {
    UNWRAP_ERROR(newVal, params.lowerExpression(getExecuteOpBuilder(), loc,
                                                attr, lowerExpressionImpl))

    // If we got a placeholder, we need to remember it, in case we map an SSA
    // value to it.
    if (auto placeholder = newVal.getDefiningOp<ParamPlaceholderOp>())
      if (!placeholderMap.contains(placeholder))
        placeholderMap.try_emplace(placeholder, SmallVector<Value>{});
    return newVal;
  }

  /// Lower a parameter declaration, resolving the placeholder if necessary. A
  /// declaration can only be lowered once, and an error is returned otherwise.
  ErrorOrSuccess lowerParamDecl(KGEN::ParamDeclAttr declAttr, Value value) {
    UNWRAP_ERROR(maybePlaceholder, params.lowerParamDecl(declAttr, value));

    if (maybePlaceholder) {
      // If we just resolved a placeholder, we need to make sure no one will
      // ever map to it in the future.
      if (placeholderMap.contains(maybePlaceholder)) {
        for (Value key : placeholderMap.at(maybePlaceholder)) {
          execute.loweredValues.erase(key);
          execute.loweredValues.try_emplace(key, value);
        }
        placeholderMap.erase(maybePlaceholder);
      }

      // We also need to replace all its existing uses.
      maybePlaceholder.replaceAllUsesWith(value);
      maybePlaceholder.getDefiningOp<ParamPlaceholderOp>().erase();
    }
    return success();
  }

  /// Return the SSA values corresponding to the parametric dimensions in the
  /// given shape (in the order they appear in the shape).
  ErrorOr<SmallVector<Value>> lowerShapeAttrDynDims(MO::ShapeAttr shapeAttr,
                                                    Location loc) {
    SmallVector<Value> runtimeDims;
    for (TypedAttr dim : shapeAttr.getValues()) {
      if (!isa<IntegerAttr>(dim)) {
        UNWRAP_ERROR(v, lowerExpression(loc, dim));
        runtimeDims.emplace_back(v);
      }
    }
    return runtimeDims;
  }

private:
  //===-----------------------------------------------------------------===//
  // Parameter lowering helpers.
  //===-----------------------------------------------------------------===//

  /// Maps KGEN::POC opcodes to mip.index.* op names.
  static ErrorOr<StringRef> pocToMopIndexOp(KGEN::POC poc) {
    // TODO: support more opcodes
    switch (poc) {
    case KGEN::POC::Add:
      return MIP::AddOp::getOperationName();
    case KGEN::POC::Div:
      return MIP::DivOp::getOperationName();
    case KGEN::POC::EQ:
      return MIP::EqualOp::getOperationName();
    case KGEN::POC::Max:
      return MIP::MaxOp::getOperationName();
    case KGEN::POC::Mul:
    case KGEN::POC::MulNuw:
      return MIP::MulOp::getOperationName();
    case KGEN::POC::Or:
      return MIP::OrOp::getOperationName();
    default:
      return Error(Twine("unsupported POC '") + stringifyPOC(poc) + "'");
    }
  }

  /// Lowers an operator attribute to the appropriate mop index op.
  static ErrorOr<Value> lowerParamOperator(OpBuilder &builder, Location loc,
                                           KGEN::ParamOperatorAttr opAttr,
                                           ValueRange operands) {
    UNWRAP_ERROR(mopName, pocToMopIndexOp(opAttr.getOpcode()));
    auto opNameAttr = builder.getStringAttr(mopName);
    Type resTy = opAttr.getType();

    Value res = operands.front();
    do {
      operands = operands.drop_front();
      res = builder.create(loc, opNameAttr, {res, operands.front()}, {resTy})
                ->getResult(0);
    } while (operands.size() > 1);
    return res;
  }

  /// Lowers an integer attribute to a mip constant op.
  static Operation *lowerIntAttr(OpBuilder &builder, Location loc,
                                 IntegerAttr intAttr) {
    if (auto boolAttr = dyn_cast<BoolAttr>(intAttr)) {
      return builder.create<MIP::ConstantBoolOp>(loc, intAttr.getType(),
                                                 boolAttr);
    }
    IntegerType si64Ty = builder.getIntegerType(64, /*isSigned=*/true);
    auto newIntAttr = builder.getIntegerAttr(si64Ty, intAttr.getInt());
    return builder.create<MIP::ConstantIndexOp>(loc, intAttr.getType(),
                                                newIntAttr);
  }

  /// Lower a parameter expression.
  static ErrorOr<Value> lowerExpressionImpl(ParameterLoweringState &state,
                                            OpBuilder &builder, Location loc,
                                            TypedAttr attr,
                                            ValueRange operands) {
    Operation *newOp;
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      newOp = lowerIntAttr(builder, loc, intAttr);
    } else if (auto opAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr)) {
      return lowerParamOperator(builder, loc, opAttr, operands);
    } else if (auto opAttr = dyn_cast<KGEN::MLIROpAttr>(attr)) {
      if (opAttr.getName() != MO::AssertOp::getOperationName())
        return Error("cannot lower mlir operation in parameter expression");

      // We must be dealing with an assert.
      newOp = builder.create<MGP::AssertOp>(loc, operands,
                                            opAttr.getAttrs().getValue());
    } else if (auto shapeAttr = dyn_cast<MO::ShapeAttr>(attr)) {
      // Lower dimensions recursively.
      SmallVector<Value> dimVals;
      for (TypedAttr dim : shapeAttr.getValues()) {
        UNWRAP_ERROR(dimVal, state.lowerExpression(builder, loc, dim,
                                                   lowerExpressionImpl));
        dimVals.emplace_back(dimVal);
      }

      // There is no runtime shape type in MGP, so we map shape attributes to a
      // fully dynamic spec.
      SmallVector<std::optional<uint64_t>> dummyShape(dimVals.size());
      auto specTy = MGP::CpuTensorSpecType::get(loc.getContext(), dummyShape,
                                                DType::si64);
      auto specOp =
          builder.create<MGP::CpuTensorSpecCreateOp>(loc, specTy, dimVals);
      return specOp.getResult();
    } else {
      return Error("cannot lower parameter expression");
    }

    return newOp->getResult(0);
  }

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

  /// The state and interface for parameter lowering.
  ParameterLoweringState params;

  /// Keeps track of all values that map to parameter placeholders.
  DenseMap<Value, SmallVector<Value>> placeholderMap;
};

} // namespace M::MGP

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTOPRIMITIVES_H
