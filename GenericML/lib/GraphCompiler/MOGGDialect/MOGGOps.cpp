//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MO dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/Interfaces/LowerableToMGPInterface.h"
#include "GenericML/GraphCompiler/Interfaces/LoweringState.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTensorInterface.h"
#include "GenericML/MGPDialect/MGPUtils.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringMap.h"

using namespace M;
using namespace MOGG;

//===----------------------------------------------------------------------===//
// MOGG ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::lower(MGP::MOGraphLoweringState &loweringState) {
  assert(MO::isOnHost(*this) && "mogg.concat is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, *this);

  // Gather the input buffers.
  SmallVector<Value> loweredOperands = opLoweringState.getLoweredOperands();
  assert(llvm::all_of(loweredOperands,
                      [](const Value loweredOperand) {
                        return isa<MGP::CpuBufferType>(
                            loweredOperand.getType());
                      }) &&
         "expecting !mgp.cpu.buffer inputs");

  // Allocate the output buffer.
  FailureOr<SmallVector<MGP::MOOutputStorage>> loweredOutputsStorageOr =
      opLoweringState.createOutputStorage(loweringState.getExecuteChain());
  if (failed(loweredOutputsStorageOr))
    return failure();
  auto &loweredOutputsStorage = *loweredOutputsStorageOr;
  assert(loweredOutputsStorageOr->size() == 1);
  assert(isa<MGP::CpuBufferType>(loweredOutputsStorage[0].buffer.getType()) &&
         "expecting !mgp.cpu.buffer outputs");

  auto concatOp = opLoweringState.create<MGP::CpuBufferConcatOp>(
      /*inChain=*/loweringState.getExecuteChain(),
      /*outBuffer=*/loweredOutputsStorage[0].buffer,
      /*inBuffers=*/loweredOperands);

  // Create the output tensors, but only once the op has completed.
  opLoweringState.createOutputs(concatOp.getOutChain(), loweredOutputsStorage);
  return success();
}

//===----------------------------------------------------------------------===//
// MOGG SplitOp
//===----------------------------------------------------------------------===//

LogicalResult SplitOp::lower(MGP::MOGraphLoweringState &loweringState) {
  assert(MO::isOnHost(*this) && "mogg.split is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, *this);

  // Gather the input buffer.
  Value inputBuffer = opLoweringState.getBufferizedInput(getInput()).buffer;
  assert(isa<MGP::CpuBufferType>(inputBuffer.getType()) &&
         "expecting !mgp.cpu.buffer input");

  // TODO Enable this.
  if (inputBuffer.getDefiningOp<MGP::CpuBufferConstantOp>())
    return emitOpError("constant input buffer is unsupported");

  // Create specs for the outputs. We don't need any buffers since this is
  // a split.
  FailureOr<SmallVector<TypedValue<MGP::CpuTensorSpecType>>> outputSpecsOr =
      opLoweringState.createOutputTensorSpecs();
  if (failed(outputSpecsOr))
    return failure();
  auto &outputSpecs = *outputSpecsOr;

  // Get the output !mo.tensor types from which we can calculate split sizes.
  SmallVector<MO::TensorType> resultMoTypes =
      MGP::castToMoTensorTypes(getResultTypes());

  // Build/collect inputs to the MGP buffer split op
  SmallVector<Type> outBufTypes;
  SmallVector<Value> runtimeSizes;
  SmallVector<uint64_t> rawSizes;
  for (TypedValue<MGP::CpuTensorSpecType> outSpec : outputSpecs) {
    MGP::CpuTensorSpecType outSpecTy = outSpec.getType();
    std::optional<uint64_t> bufferSize = outSpecTy.getSizeInBytes();
    outBufTypes.emplace_back(MGP::CpuBufferType::get(getContext(), bufferSize));
    rawSizes.emplace_back(optSizeToRawSize(bufferSize));
    if (!bufferSize.has_value()) {
      runtimeSizes.emplace_back(
          opLoweringState.create<MGP::CpuTensorSpecSizeOp>(
              /*spec=*/outSpec));
    }
  }

  // Build MGP buffer split op
  auto mgpBufferSplitOp = opLoweringState.create<MGP::CpuBufferSplitBorrowOp>(
      /*outBuffers=*/outBufTypes,
      /*inChain=*/loweringState.getExecuteChain(),
      /*inBuffer=*/inputBuffer,
      /*runtimeSizes=*/runtimeSizes,
      /*outBufferRawSizes=*/rawSizes);

  // Create the output tensors. Note there's no new chain to wait on.
  SmallVector<Value> outputBuffers = mgpBufferSplitOp.getOutBuffers();
  opLoweringState.createOutputTensors(loweringState.getExecuteChain(),
                                      outputBuffers, outputSpecs);
  return success();
}

//===----------------------------------------------------------------------===//
// MOGG LoadCompiledKernelOp
//===----------------------------------------------------------------------===//

LogicalResult
LoadCompiledKernelOp::lower(MGP::MOGraphLoweringState &loweringState) {
  OpBuilder &builder = loweringState.getInitOpBuilder();
  MGP::BufferBuilder *bufferBuilder =
      loweringState.getBufferBuilder(MO::getDeviceOrDefault(*this));
  Value outChain = bufferBuilder->kernelLoadArchive(
      builder, getLoc(), loweringState.getInitMgpContext(),
      /*inChain=*/loweringState.getInitChain(),
      /*archive=*/getValue(),
      /*kernelNames=*/getSymbols(),
      /*constantNames=*/{});
  loweringState.setInitChain(outChain);
  return success();
}

//===----------------------------------------------------------------------===//
// MOGG ExecuteCompiledOp
//===----------------------------------------------------------------------===//

LogicalResult
ExecuteCompiledOp::lower(MGP::MOGraphLoweringState &loweringState) {
  MGP::MOOpLoweringState opLoweringState(loweringState, *this);

  // Gather the lowered op operands.
  SmallVector<Value> loweredInputOperands =
      opLoweringState.getLoweredOperands();

  // Allocate for the op's results.
  // At this stage we gate the allocations only on the overall 'execute' block's
  // chain. The DeferAllocsPass will change this to use the chain which signals
  // when all of the operation's input buffers are ready to be read, thus
  // ensuring we allocate buffers just-in-time for their writers.
  FailureOr<SmallVector<MGP::MOOutputStorage>> loweredOutputsStorageOr =
      opLoweringState.createOutputStorage(loweringState.getExecuteChain());
  if (failed(loweredOutputsStorageOr))
    return failure();
  auto &loweredOutputsStorage = *loweredOutputsStorageOr;

  size_t numOperands =
      loweredInputOperands.size() + loweredOutputsStorage.size();

  // Gather all the input and output operands and their in/out semantics.
  SmallVector<Value> operands;
  SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics;
  operands.reserve(numOperands);
  operandSemantics.reserve(numOperands);
  for (Value loweredInputOperand : loweredInputOperands) {
    operands.push_back(loweredInputOperand);
    operandSemantics.emplace_back(InOutSignatureAttr::kIn);
  }
  for (auto &outputStorage : loweredOutputsStorage) {
    operands.push_back(outputStorage.buffer);
    operandSemantics.emplace_back(InOutSignatureAttr::kOut);
  }

  // Gather the input dimensions from the dimension parameters in the
  // !mo.tensor types.
  SmallVector<Value> inDimensions;
  ArrayAttr usedParams = getUsedParamsAttr();
  for (auto param : usedParams.getAsRange<KGEN::ParamDeclRefAttr>()) {
    auto symbolicDimOr =
        loweringState.lowerExpression(opLoweringState.loc, param);
    if (symbolicDimOr.isError())
      return failure();
    inDimensions.push_back(symbolicDimOr.takeValue());
  }

  // Emit the kernel call.
  SmallVector<Value> outDimensions;
  auto outChain = opLoweringState.kernelExecute(
      /*kernelName=*/getKernelName(),
      /*inChain=*/loweringState.getExecuteChain(),
      /*args=*/operands,
      /*inOutSemantics=*/operandSemantics,
      /*inDimensions=*/inDimensions,
      /*numOutDimensions=*/static_cast<uint64_t>(getOutputParamDecls().size()),
      /*outDimensions=*/outDimensions,
      /*summary=*/getSummary(),
      /*isAsync*/ getIsAsync());

  // Each of the parameter outputs now maps onto a concrete dimension value.
  for (const auto &[param, value] :
       llvm::zip(getOutputParamDecls(), outDimensions)) {
    if (failed(loweringState.lowerParamDecl(param, value)))
      return failure();
  }

  // Create outputs from their output storage, but only once the op has
  // completed.
  opLoweringState.createOutputs(outChain, loweredOutputsStorage);

  return success();
}

//===----------------------------------------------------------------------===//
// Parameter printing/parsing support in form: <P1, P2, P3>
//===----------------------------------------------------------------------===//

static ParseResult parseParamDecls(OpAsmParser &p,
                                   KGEN::ParamDeclArrayAttr &paramDecls) {
  if (succeeded(p.parseOptionalLess()) && failed(p.parseOptionalGreater())) {
    if (KGEN::parseParamDecls(p, paramDecls) || p.parseGreater())
      return failure();
  } else {
    paramDecls = KGEN::ParamDeclArrayAttr::get(p.getContext(), {});
  }
  return success();
}

static void printParamDecls(OpAsmPrinter &p, Operation *,
                            KGEN::ParamDeclArrayAttr paramDecls) {
  if (!paramDecls.empty()) {
    p << "<";
    printParamDecls(p, paramDecls);
    p << '>';
  }
}

static ParseResult parseLambda(OpAsmParser &p, StringAttr &name, TypeAttr &func,
                               Region &region) {
  SmallVector<OpAsmParser::Argument> args;
  llvm::SMLoc loc;
  KGEN::SignatureType sigType;
  FunctionType functionType;
  KGEN::ParamDeclArrayAttr inputParams, resultParams;

  llvm::SMLoc sigLoc;
  if (p.parseSymbolName(name) || p.parseEqual() ||
      p.getCurrentLocation(&sigLoc) ||
      KGEN::parseFunctionSignature(p, args, inputParams, resultParams,
                                   functionType, sigType) ||
      p.getCurrentLocation(&loc) || p.parseRegion(region, args))
    return failure();
  if (!inputParams.empty() || !resultParams.empty())
    return p.emitError(sigLoc, "unexpected input or result parameters");

  func = TypeAttr::get(functionType);
  return success();
}

static void printLambda(OpAsmPrinter &p, mlir::Operation *lambda,
                        StringAttr name, TypeAttr sig, Region &region) {
  p.printSymbolName(name.strref());
  p << " = ";

  auto func = cast<FunctionType>(sig.getValue());

  // TODO: what conventions should we use here?
  SmallVector<KGEN::ValueInputConvention> conventions(
      func.getInputs().size(), KGEN::ValueInputConvention::None);
  auto funcType = FunctionType::get(lambda->getContext(), func.getInputs(),
                                    func.getResults());
  auto sigTy = KGEN::SignatureType::get(funcType, conventions);
  KGEN::printFunctionSignature(p, &region, {}, {}, func, sigTy);
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// MOGG CallOp
//===----------------------------------------------------------------------===//

bool CallOp::setKGENParamToValueIfPresent(StringRef name, Attribute value) {
  // Exit out if we don't have the parameter already (we can't add new ones as
  // the params come from mojo).
  std::optional<mlir::DictionaryAttr> dict = getKgenParams();
  if (!dict)
    return false;

  mlir::NamedAttrList attributes(*dict);
  // Exit if the parameter doesn't exist in the dictionary attr.
  if (!attributes.getNamed(name).has_value())
    return false;

  Attribute oldAttr = attributes.set(name, value);

  // Update the dict if it doesn't already have that value.
  if (oldAttr != value)
    setKgenParamsAttr(attributes.getDictionary(getContext()));

  return true;
}

//===----------------------------------------------------------------------===//
// MOGG KernelOp
//===----------------------------------------------------------------------===//

OutputPlaceholdersOp KernelOp::getOutputPlaceholder() {
  auto placeholder = this->getOps<MOGG::OutputPlaceholdersOp>();
  if (placeholder.empty())
    return nullptr;

  return *placeholder.begin();
}

void KernelOp::setInsertionPointToTop(OpBuilder &builder) {
  OutputPlaceholdersOp outputPh = getOutputPlaceholder();

  // Not everything has to have an input placeholder, e.g a shape function.
  if (outputPh)
    builder.setInsertionPointAfter(outputPh);
  else
    builder.setInsertionPointToStart(&getRegion().front());
}

/// Disables future fusion on this input.
void KernelOp::disableInputFusionOnInput(size_t index) {
  std::optional<ArrayAttr> inputLambdas = getInputFusionInterfaceAttr();

  if (inputLambdas.has_value()) {
    SmallVector<Attribute> lambdaNames;

    // Add all the old attributes to the new one but remove the attribute at the
    // disabled index.
    size_t i = 0;
    for (StringAttr attr : inputLambdas->getAsRange<StringAttr>()) {
      if (i == index) {
        lambdaNames.push_back(StringAttr::get(getContext(), ""));
      } else {
        lambdaNames.push_back(attr);
      }

      i++;
    }

    // Update the attribute.
    setInputFusionInterfaceAttr(ArrayAttr::get(getContext(), lambdaNames));
  }
}

void KernelOp::removeTemporaryFusionMarks() {
  SmallVector<Attribute> lambdaNames;
  for (StringAttr attr : getInputFusionInterface()->getAsRange<StringAttr>()) {
    // If this is the temporary string then replace it with null.
    if (attr == KernelOp::getTempLambdaName())
      lambdaNames.push_back(StringAttr::get(getContext(), ""));
    else
      lambdaNames.push_back(attr);
  }
  setInputFusionInterfaceAttr(ArrayAttr::get(getContext(), lambdaNames));
}

/// Removes an operand at the given index and disables input fusion on that
/// position if this kernel otherwise supports it.
void KernelOp::removeOperandAtIndex(size_t index) {
  (*this)->eraseOperand(index);
  getBody().front().eraseArgument(index);

  std::optional<ArrayAttr> inputInterfaces = getInputFusionInterface();

  // Update the input fusion interfaces if we have them to remove fusion on this
  // attr.
  if (inputInterfaces.has_value()) {
    SmallVector<Attribute> lambdaNames;
    size_t i = 0;
    // Add each of the current input hooks to the list, skipping the input we
    // just removed.
    for (StringAttr name : inputInterfaces->getAsRange<StringAttr>()) {
      if (i != index)
        lambdaNames.push_back(name);
      i++;
    }
    setInputFusionInterfaceAttr(ArrayAttr::get(getContext(), lambdaNames));
  }
}

/// Add an operand to the kernel and adds an argument to the kernel region.
/// Fixes up the input fusion interfaces to disallow input fusion on this
/// position. Returns the added kernel argument.
Value KernelOp::addOperand(Value operand) {
  (*this)->insertOperands(getNumOperands(), ValueRange{operand});
  Value blockArg =
      getBody().front().addArgument(operand.getType(), operand.getLoc());

  std::optional<ArrayAttr> inputInterfaces = getInputFusionInterface();

  // Update the input fusion interfaces if we have them to remove fusion on this
  // attr.
  if (inputInterfaces.has_value()) {
    SmallVector<Attribute> lambdaNames;
    for (StringAttr name : inputInterfaces->getAsRange<StringAttr>())
      lambdaNames.push_back(name);
    // Add an empty lambda name for the operand we just added.
    lambdaNames.push_back(StringAttr::get(getContext(), ""));
    setInputFusionInterfaceAttr(ArrayAttr::get(getContext(), lambdaNames));
  }

  return blockArg;
}

void KernelOp::setInputFusionInterface(size_t index, StringRef lambdaName) {
  std::optional<ArrayAttr> inputInterfaces = getInputFusionInterface();

  // Update the input fusion interfaces if we have them to remove fusion on this
  // attr.
  if (inputInterfaces.has_value()) {
    SmallVector<Attribute> lambdaNames;
    for (StringAttr name : inputInterfaces->getAsRange<StringAttr>()) {
      if (lambdaNames.size() == index)
        lambdaNames.push_back(StringAttr::get(getContext(), lambdaName));
      else
        lambdaNames.push_back(name);
    }
    setInputFusionInterfaceAttr(ArrayAttr::get(getContext(), lambdaNames));
  }
}

// Fuse the body of this kernel into another.
void KernelOp::fuseBodyIntoNew(
    OpBuilder &builder, mlir::SymbolTable &symTable, IRMapping &remapper,
    llvm::StringMap<StringRef> &oldLambdaNamesToNew) {
  auto neverSkip = [](Operation *) { return false; };

  fuseBodyIntoNewWithSkip(builder, symTable, remapper, oldLambdaNamesToNew,
                          neverSkip);
}

// Fuse the body of this kernel into another.
void KernelOp::fuseBodyIntoNewWithSkip(
    OpBuilder &builder, mlir::SymbolTable &symTable, IRMapping &remapper,
    llvm::StringMap<StringRef> &oldLambdaNamesToNew,
    std::function<bool(Operation *)> shouldSkip) {

  // Clone the incoming kernel into this kernel.
  for (Operation &op : getBody().front()) {
    if (shouldSkip(&op))
      continue;

    if (isa<MOGG::OutputOp>(op)) {
      // Skip placeholders & output, already remapped.
    } else if (auto lambda = dyn_cast<LambdaOp>(op)) {
      // Add lambdas seperately so we rename them on insertion. (The
      // symbol table does this for us).
      auto newLambda = cast<LambdaOp>(builder.clone(op, remapper));
      symTable.insert(newLambda, builder.getInsertionPoint());
      builder.setInsertionPointAfter(newLambda);

      oldLambdaNamesToNew[lambda.getSymName()] = newLambda.getSymName();
    } else if (auto call = dyn_cast<CallOp>(op)) {
      // Lambdas will have been inlined. The only place we can have
      // a lambda reference is in a call.
      auto newCall = dyn_cast<CallOp>(builder.clone(op, remapper));

      SmallVector<Attribute> newAttrs;

      // Remap any lambda references to point to the new lambda name.
      std::optional<ArrayAttr> params = newCall.getParams();
      if (params) {
        for (SymbolRefAttr sym : params->getAsRange<SymbolRefAttr>()) {
          StringRef name = sym.getLeafReference().strref();
          auto itr = oldLambdaNamesToNew.find(name);
          assert(itr != oldLambdaNamesToNew.end() &&
                 "First use of lambda inserted before the lambda itself");
          newAttrs.push_back(FlatSymbolRefAttr::get(getContext(), itr->second));
        }
        newCall.setParamsAttr(builder.getArrayAttr(newAttrs));
      }
    } else {
      builder.clone(op, remapper);
    }
  }
}

// Verify that all the `mogg.set_param` set the symbolic dimensions properly.
static LogicalResult verifyKernelSetParam(KernelOp kernel) {
  llvm::SmallSet<StringRef, 4> symbolicDimSet;

  if (kernel.getOps<MOGG::SetParameterOp>().empty())
    return success();

  // Check that each symbolic dim gets assigned at most once.
  for (auto op : kernel.getOps<MOGG::SetParameterOp>()) {
    if (!symbolicDimSet.insert(op.getParamName()).second)
      return mlir::emitError(op->getLoc(),
                             "Symbolic dimension has been assigned more than "
                             "once");
  }

  // Check that the output (at most one) has all the symbolic dimensions
  // assigned by `mogg.set_param`.
  for (auto outputPlaceholder : kernel.getOps<MOGG::OutputPlaceholdersOp>()) {
    if (!llvm::all_of(outputPlaceholder.getResults(), [&](mlir::OpResult res) {
          auto resType = cast<MO::TensorType>(res.getType());

          return llvm::all_of(
              resType.getTypedShapeAttr().getValues(),
              [&](mlir::TypedAttr dim) {
                auto symblicDim = dyn_cast<KGEN::ParamDeclRefAttr>(dim);
                if (!symblicDim)
                  return true;
                return symbolicDimSet.contains(symblicDim.getName());
              });
        }))
      return mlir::emitError(outputPlaceholder->getLoc(),
                             "Not all symbolic dimensions are assigned");
  }

  return success();
}

LogicalResult KernelOp::verify() {
  if (llvm::range_size(getOps<MOGG::OutputPlaceholdersOp>()) > 1)
    return emitError("A KernelOp must have at most one output placeholder op");

  auto inputInterfaces = getInputFusionInterface();
  auto outputInterfaces = getOutputFusionInterface();

  if (inputInterfaces.has_value() &&
      getNumOperands() != inputInterfaces->size()) {
    return emitError("Number of input fusion interfaces does not match number "
                     "of inputs")
           << " (" << inputInterfaces->size() << " vs " << getNumOperands()
           << ")";
  }

  if (outputInterfaces.has_value() &&
      getNumResults() != outputInterfaces->size()) {
    return emitError("Number of output fusion interfaces does not match number "
                     "of results")
           << " (" << outputInterfaces->size() << " vs " << getNumResults()
           << ")";
  }

  mlir::SymbolTable symTable{*this};
  if (inputInterfaces.has_value()) {
    for (auto interface : inputInterfaces->getAsValueRange<StringAttr>()) {
      if (interface.empty())
        continue;

      Operation *lambda = symTable.lookup(interface);
      if (!lambda || !isa<LambdaOp, KGEN::ParamDeclareRegionOp>(lambda)) {
        return emitError("Could not find lambda op specified in the input "
                         "fusion interface as \"")
               << interface.str() << "\"";
      }
    }
  }

  if (outputInterfaces.has_value()) {
    for (auto interface : outputInterfaces->getAsValueRange<StringAttr>()) {
      if (interface.empty())
        continue;

      Operation *lambda = symTable.lookup(interface);
      if (!lambda || !isa<LambdaOp, KGEN::ParamDeclareRegionOp>(lambda)) {
        return emitError("Could not find lambda op specified in the output "
                         "fusion interface as \"")
               << interface.str() << "\"";
      }
    }
  }

  if (getBody().empty()) {
    return emitError("Region cannot be empty");
  }

  uint64_t numArguments = llvm::range_size(getBody().getArguments());
  if (getNumOperands() != numArguments) {
    return emitError(
               "Number of operands does not match number of body arguments ")
           << " (" << getNumOperands() << " vs " << numArguments << ")";
  }

  // Verify that symbolic dims are set properly in the kernel.
  auto setParamCheck =
      verifyKernelSetParam(mlir::cast<KernelOp>(getOperation()));
  if (mlir::failed(setParamCheck))
    return setParamCheck;

  return success();
}

static inline bool allNullFusionInterface(ArrayAttr arr) {
  auto strArr = arr.getAsRange<StringAttr>();
  return std::all_of(strArr.begin(), strArr.end(), [](StringAttr strAttr) {
    return strAttr.strref().empty();
  });
}

LogicalResult KernelOp::canonicalize(KernelOp op, PatternRewriter &) {
  if (op.getInputFusionInterface().has_value() &&
      allNullFusionInterface(op.getInputFusionInterfaceAttr())) {
    op.removeInputFusionInterfaceAttr();
  }

  if (op.getOutputFusionInterface().has_value() &&
      allNullFusionInterface(op.getOutputFusionInterfaceAttr())) {
    op.removeOutputFusionInterfaceAttr();
  }

  return success();
}

static ParseResult parseKernel(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &operandTypes, SmallVectorImpl<Type> &resultTypes,
    Region &region, KGEN::ParamDeclArrayAttr &resultParams) {
  SmallVector<OpAsmParser::Argument> args;
  llvm::SMLoc loc;
  KGEN::SignatureType sigType;
  FunctionType functionType;
  KGEN::ParamDeclArrayAttr inputParams, dummyRes;

  llvm::SMLoc sigLoc;
  if (parseParamDecls(p, resultParams) ||
      p.parseOperandList(operands, AsmParser::Delimiter::Paren) ||
      p.parseColon() || p.getCurrentLocation(&sigLoc) ||
      KGEN::parseFunctionSignature(p, args, inputParams, dummyRes, functionType,
                                   sigType) ||
      p.getCurrentLocation(&loc) || p.parseRegion(region, args))
    return failure();
  if (!inputParams.empty() || !dummyRes.empty())
    return p.emitError(sigLoc, "unexpected input or result parameters");

  for (const OpAsmParser::Argument &arg : args)
    operandTypes.push_back(arg.type);

  for (Type res : functionType.getResults())
    resultTypes.push_back(res);

  return success();
}

static void printKernel(OpAsmPrinter &p, mlir::Operation *kernel,
                        ValueRange operands, TypeRange operandTypes,
                        TypeRange resultTypes, Region &region,
                        KGEN::ParamDeclArrayAttr resultParams) {
  printParamDecls(p, kernel, resultParams);
  p << '(';
  p.printOperands(operands);
  p << ") : ";

  auto functionType =
      FunctionType::get(kernel->getContext(), operandTypes, resultTypes);
  SmallVector<KGEN::ValueInputConvention> conventions(
      operandTypes.size(), KGEN::ValueInputConvention::None);

  KGEN::printFunctionSignature(p, &region, /*inputParams=*/{},
                               /*resultParams=*/{}, functionType,
                               SignatureType::get(functionType, conventions));
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

void MOGGDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GenericML/GraphCompiler/MOGGDialect/MOGG.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/MOGGDialect/MOGG.cpp.inc"
