//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// On-demand derivation of tensor-arg trait contracts from kernel declarations
// (GEX-4071).
//
// The trait contract of a kernel I/O tensor is a function of how the
// corresponding exec-body value reaches the kernel's call operation: the
// call operand position is the stable key, and the declaration
// (`mogg.kernel.decl` / `mogg.shape_func.decl`) is the source of truth for
// what that position's contract is. Deriving the contract on demand makes it
// impossible for operand-list transforms (fusion, dedup) to invalidate it, in
// contrast to a positional side-table stamped once at mo-to-mogg.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_TRAITCONTRACTDERIVATION_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_TRAITCONTRACTDERIVATION_H

#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>
#include <optional>

namespace M::MOGG {

/// The trait contract declared for one `execute` call operand: the input
/// (`isOutput == false`) or DPS-output operand at `operandIdx`, indexed
/// exactly as `CallExecuteOp::emitMojo` indexes the declaration (raw
/// `getInputs()` position for inputs -- the trailing device context, when
/// present, lands on the declaration's launch argument, which carries no
/// contract). Returns `std::nullopt` for a legacy or non-tensor argument, and
/// for an out-of-range index (exotic legacy variadic packing can legitimately
/// shorten the operand list; a trait-bound declaration never hits this).
std::optional<TensorArgContract>
declaredExecuteOperandContract(const KernelDeclFunctionAdaptor &executeFn,
                               unsigned operandIdx, bool isOutput);

/// The trait contract declared for one `mogg.call.shape` operand. Shape
/// functions carry role-less (`TensorArgRole::none`) contracts for their
/// trait-bound tensor arguments; legacy arguments yield `std::nullopt`.
std::optional<TensorArgContract>
declaredShapeOperandContract(const KernelDeclFunctionAdaptor &shapeFn,
                             unsigned operandIdx);

/// Derives trait contracts for kernel I/O values by walking their uses to the
/// kernel call and reading the declaration.
///
/// A kernel I/O value is an exec-body / kernel-stub block argument (an input)
/// or a `mogg.output_placeholders` result (a DPS output). The traversal follows
/// value-forwarding ops (`mogg.bind`, `mogg.tensor.fused`, `mogg.call.view`,
/// `mogg.as_scalar`, `mogg.as_tuple`, ...). For a value captured by a
/// `mogg.bind` lambda, the traversal continues from the lambda output to find
/// the call operand carrying the capture. When the value also reaches a call
/// directly (`SimplifyKernelOperands` can deduplicate a capture with a direct
/// operand), the direct operand determines the exact contract. The captured
/// path only verifies that both uses require the same representation:
/// trait-based or legacy.
///
/// Call kinds resolve as:
///
///   - `mogg.call.execute`  -> declaration `execute` argument contract
///   - `mogg.call.shape`    -> shape-function declaration argument contract
///   - `mogg.call.foreach` / `mogg.call.materialize` -> no contract (these
///     calls declare no trait-bound arguments)
///
/// A missing declaration or an unrecognized use is a hard diagnostic --
/// never a silent legacy fallback, which would recreate the silent-mislabel
/// failure mode of the positional side-table this derivation replaced.
///
/// Declaration adaptors are cached per callee symbol, so deriving every
/// kernel I/O value decodes each declaration once.
class KernelIOContractDeriver {
public:
  /// `module` is the module holding the kernel and shape-function
  /// declarations. The deriver builds its own symbol table over it: the
  /// pass-cached `SymbolTableAnalysis` table may predate `ImportPackages`
  /// (it is never invalidated), so it cannot be trusted to see the decls.
  explicit KernelIOContractDeriver(mlir::ModuleOp module)
      : symbolTable(module) {}

  /// Derive the unique trait contract for `kernelIOValue`, or a null
  /// attribute for a legacy (non-trait) operand. Emits a diagnostic and
  /// fails when a declaration is missing, a use is unrecognized, or two
  /// uses derive conflicting contracts.
  mlir::FailureOr<TensorArgContractAttr> derive(mlir::Value kernelIOValue);

  /// Derive the trait contract a single use-edge reaches.
  ///
  /// Same walk as `derive`, but only `use` of the used value votes -- its
  /// sibling uses do not. Lane is a property of the use-edge, not the op:
  /// CSE may merge equal constants across a legacy and a trait operand, and
  /// each edge can then be materialized on its own lane instead of the
  /// merged value being rejected as a conflict.
  mlir::FailureOr<TensorArgContractAttr> deriveUse(mlir::OpOperand &use);

private:
  class ContractEvidence;

  mlir::FailureOr<TensorArgContractAttr> deriveImpl(mlir::Value kernelIOValue,
                                                    mlir::OpOperand *seedUse);

  mlir::FailureOr<bool> recordCallContract(mlir::OpOperand &use,
                                           ContractEvidence &evidence);

  const KernelDeclAdaptor *kernelAdaptorFor(MOGG::CallExecuteOp executeOp);
  const KernelDeclFunctionAdaptor *shapeAdaptorFor(MOGG::CallShapeOp shapeOp);

  mlir::SymbolTable symbolTable;
  llvm::DenseMap<mlir::StringAttr, std::unique_ptr<KernelDeclAdaptor>>
      kernelAdaptorCache;
  llvm::DenseMap<mlir::StringAttr, std::unique_ptr<KernelDeclFunctionAdaptor>>
      shapeAdaptorCache;
};

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_TRAITCONTRACTDERIVATION_H
