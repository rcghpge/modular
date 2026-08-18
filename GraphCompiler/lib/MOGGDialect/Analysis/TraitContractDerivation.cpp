//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/Analysis/TraitContractDerivation.h"

#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/DenseSet.h"

using namespace M;
using namespace M::MOGG;
using namespace mlir;

class M::MOGG::KernelIOContractDeriver::ContractEvidence {
public:
  LogicalResult record(std::optional<TensorArgContract> contract,
                       OpOperand &sourceOperand) {
    Source source{sourceOperand.getOwner(), sourceOperand.getOperandNumber()};
    if (!firstSource) {
      merged = contract;
      firstSource = source;
      return mlir::success();
    }
    if (merged == contract)
      return mlir::success();

    // TODO(GEX-4071): a value reaching two operand positions with different
    // contracts needs one init op per distinct contract from a shared buffer.
    // In-stub constants already get that split (per-use lanes via
    // `deriveUse`); boundary kernel I/O lowered through `derive` does not
    // yet, so reject the conflict.
    return emitConflict(source, contract, *firstSource, merged);
  }

  bool hasEvidence() const { return firstSource.has_value(); }

  bool hasContract() const { return merged.has_value(); }

  const std::optional<TensorArgContract> &getContract() const { return merged; }

  LogicalResult emitConflictWith(const ContractEvidence &first) const {
    return emitConflict(*firstSource, merged, *first.firstSource, first.merged);
  }

private:
  struct Source {
    Operation *owner;
    unsigned operandNumber;
  };

  template <typename DiagnosticT>
  static void appendContract(DiagnosticT &diag, const Source &source,
                             std::optional<TensorArgContract> contract) {
    if (contract) {
      diag << tensorArgContractToAttr(source.owner->getContext(), *contract);
      return;
    }
    diag << "legacy `ManagedTensorSlice`";
  }

  static LogicalResult
  emitConflict(const Source &source, std::optional<TensorArgContract> contract,
               const Source &firstSource,
               std::optional<TensorArgContract> firstContract) {
    InFlightDiagnostic diag =
        source.owner->emitError()
        << "kernel I/O value derives conflicting tensor-arg contracts: call "
           "operand #"
        << source.operandNumber << " requires ";
    appendContract(diag, source, contract);
    Diagnostic &note = diag.attachNote(firstSource.owner->getLoc());
    note << "call operand #" << firstSource.operandNumber << " first required ";
    appendContract(note, firstSource, firstContract);
    return diag;
  }

  std::optional<TensorArgContract> merged;
  std::optional<Source> firstSource;
};

std::optional<TensorArgContract> M::MOGG::declaredExecuteOperandContract(
    const KernelDeclFunctionAdaptor &executeFn, unsigned operandIdx,
    bool isOutput) {
  llvm::ArrayRef<KernelDeclOperandAdaptor> args =
      isOutput ? executeFn.outputArguments : executeFn.inputArguments;
  if (operandIdx >= args.size())
    return std::nullopt;
  return args[operandIdx].getTraitContract();
}

std::optional<TensorArgContract>
M::MOGG::declaredShapeOperandContract(const KernelDeclFunctionAdaptor &shapeFn,
                                      unsigned operandIdx) {
  if (operandIdx >= shapeFn.inputArguments.size())
    return std::nullopt;
  return shapeFn.inputArguments[operandIdx].getTraitContract();
}

const KernelDeclAdaptor *
KernelIOContractDeriver::kernelAdaptorFor(MOGG::CallExecuteOp executeOp) {
  StringAttr symName = executeOp.getKernelSymNameAttr();
  auto [it, inserted] = kernelAdaptorCache.try_emplace(symName);
  if (inserted) {
    if (auto decl = symbolTable.lookup<KernelDeclOp>(symName))
      it->second = std::make_unique<KernelDeclAdaptor>(decl);
  }
  return it->second.get();
}

const KernelDeclFunctionAdaptor *
KernelIOContractDeriver::shapeAdaptorFor(MOGG::CallShapeOp shapeOp) {
  StringAttr symName = shapeOp.getKernelSymNameAttr();
  auto [it, inserted] = shapeAdaptorCache.try_emplace(symName);
  if (inserted) {
    if (auto decl = symbolTable.lookup<ShapeFunctionDeclOp>(symName))
      it->second = std::make_unique<KernelDeclFunctionAdaptor>(decl);
  }
  return it->second.get();
}

FailureOr<bool>
KernelIOContractDeriver::recordCallContract(OpOperand &use,
                                            ContractEvidence &evidence) {
  Operation *owner = use.getOwner();

  if (auto executeOp = dyn_cast<MOGG::CallExecuteOp>(owner)) {
    const KernelDeclAdaptor *adaptor = kernelAdaptorFor(executeOp);
    if (!adaptor) {
      executeOp.emitError()
          << "no kernel declaration named '" << executeOp.getKernelSymName()
          << "' in the module; tensor-arg contracts are derived from the "
             "declaration (import packages or seed the module via "
             "copyKernelDecls)";
      return failure();
    }
    if (!adaptor->executeFunction) {
      executeOp.emitError()
          << "kernel declaration '" << executeOp.getKernelSymName()
          << "' has no execute function";
      return failure();
    }

    unsigned opIdx = use.getOperandNumber();
    unsigned numInputs = executeOp.getInputs().size();
    std::optional<TensorArgContract> contract =
        opIdx < numInputs
            ? declaredExecuteOperandContract(*adaptor->executeFunction, opIdx,
                                             /*isOutput=*/false)
            : declaredExecuteOperandContract(*adaptor->executeFunction,
                                             opIdx - numInputs,
                                             /*isOutput=*/true);
    if (failed(evidence.record(contract, use)))
      return failure();
    return true;
  }

  if (auto shapeOp = dyn_cast<MOGG::CallShapeOp>(owner)) {
    const KernelDeclFunctionAdaptor *shapeFn = shapeAdaptorFor(shapeOp);
    if (!shapeFn) {
      shapeOp.emitError()
          << "no shape-function declaration named '"
          << shapeOp.getKernelSymName()
          << "' in the module; tensor-arg contracts are derived from the "
             "declaration (import packages or seed the module via "
             "copyKernelDecls)";
      return failure();
    }
    if (failed(evidence.record(
            declaredShapeOperandContract(*shapeFn, use.getOperandNumber()),
            use)))
      return failure();
    return true;
  }

  if (isa<MOGG::CallForeachOp, MOGG::CallMaterializeOp>(owner)) {
    if (failed(evidence.record(std::nullopt, use)))
      return failure();
    return true;
  }

  return false;
}

FailureOr<TensorArgContractAttr>
KernelIOContractDeriver::derive(Value kernelIOValue) {
  return deriveImpl(kernelIOValue, /*seedUse=*/nullptr);
}

FailureOr<TensorArgContractAttr>
KernelIOContractDeriver::deriveUse(OpOperand &use) {
  return deriveImpl(use.get(), &use);
}

FailureOr<TensorArgContractAttr>
KernelIOContractDeriver::deriveImpl(Value kernelIOValue, OpOperand *seedUse) {
  struct UseTraversal {
    llvm::SmallVector<Value, 4> worklist;
    llvm::DenseSet<Value> processed;
    ContractEvidence evidence;
  };

  // An explicit use reaches a kernel call through ordinary SSA forwarding. A
  // captured use occurs inside a `mogg.bind` lambda; the lambda output, rather
  // than the captured value, reaches the call. Explicit uses determine the
  // exact contract, so process them before captured uses.
  UseTraversal explicitUses;
  UseTraversal capturedUses;

  auto handleUse = [&](OpOperand &use, UseTraversal &current) -> LogicalResult {
    Operation *owner = use.getOwner();
    FailureOr<bool> recorded = recordCallContract(use, current.evidence);
    if (failed(recorded))
      return failure();
    if (*recorded)
      return success();

    // Value-forwarding ops: the contract belongs to wherever the forwarded
    // value lands.
    if (isa<MOGG::BindLambdaOp, MOGG::TensorFusedOp, MOGG::CallViewOp,
            MOGG::CallAffineOp, MOGG::AsScalarOp, MOGG::AsTupleOp,
            MOGG::TensorReshapeOp>(owner)) {
      for (Value result : owner->getResults())
        current.worklist.push_back(result);
      return success();
    }

    // Defer the lambda output until all explicit uses are processed.
    if (auto bindOp = owner->getParentOfType<MOGG::BindLambdaOp>()) {
      capturedUses.worklist.push_back(bindOp.getOutput());
      return success();
    }

    owner->emitError()
        << "cannot derive a tensor-arg contract through this use of a kernel "
           "I/O value ('"
        << owner->getName() << "' is not a recognized kernel-body op)";
    return failure();
  };

  // `deriveUse`: only `seedUse` of `kernelIOValue` votes.
  if (seedUse) {
    explicitUses.processed.insert(kernelIOValue);
    if (failed(handleUse(*seedUse, explicitUses)))
      return failure();
  } else {
    explicitUses.worklist.push_back(kernelIOValue);
  }

  while (!explicitUses.worklist.empty() || !capturedUses.worklist.empty()) {
    bool processingExplicitUse = !explicitUses.worklist.empty();
    UseTraversal &current = processingExplicitUse ? explicitUses : capturedUses;

    Value value = current.worklist.pop_back_val();
    if (!processingExplicitUse && explicitUses.processed.contains(value))
      continue;
    if (!current.processed.insert(value).second)
      continue;

    for (OpOperand &use : value.getUses())
      if (failed(handleUse(use, current)))
        return failure();
  }

  // If both explicit and captured uses reach calls, both must be trait-based
  // or both must use legacy `ManagedTensorSlice`.
  if (explicitUses.evidence.hasEvidence() &&
      capturedUses.evidence.hasEvidence() &&
      explicitUses.evidence.hasContract() !=
          capturedUses.evidence.hasContract()) {
    return capturedUses.evidence.emitConflictWith(explicitUses.evidence);
  }

  const ContractEvidence &result = explicitUses.evidence.hasEvidence()
                                       ? explicitUses.evidence
                                       : capturedUses.evidence;
  if (result.getContract())
    return tensorArgContractToAttr(kernelIOValue.getContext(),
                                   *result.getContract());
  return TensorArgContractAttr();
}
