//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJOREGISTRY_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJOREGISTRY_H

#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace M::MOGG {

/// Indexes `mogg.kernel.decl`, `mogg.primitive.decl`, and
/// `mogg.shape_func.decl` ops in a module for fast lookup by registration
/// name or sym_name.
///
/// This is a plain helper, not an MLIR analysis: construct one at the entry
/// of a pass that needs lookups, and let it go out of scope at the end. The
/// index is built eagerly; subsequent lookups are O(1).
class MojoRegistry {
public:
  explicit MojoRegistry(mlir::ModuleOp module);

  /// Returns all `mogg.kernel.decl` ops whose `registration` matches `name`.
  /// Multiple structs may share the same registration. Returns an empty range
  /// if none is found. The returned view aliases storage owned by the
  /// registry; it stays valid for the registry's lifetime.
  llvm::ArrayRef<KernelDeclOp> lookupKernel(llvm::StringRef name) const;

  /// Returns true if at least one `mogg.kernel.decl` op is registered under
  /// `name`.
  bool containsKernel(llvm::StringRef name) const;

  KernelDeclOp lookupKernelBySymName(llvm::StringRef symName) const;

  /// Returns the `mogg.primitive.decl` op whose `registration` matches
  /// `name`. Asserts that the primitive exists; use `containsPrimitive` for
  /// an existence check.
  PrimitiveDeclOp lookupPrimitive(llvm::StringRef name) const;

  PrimitiveDeclOp lookupPrimitiveBySymName(llvm::StringRef symName) const;

  /// Returns true if a `mogg.primitive.decl` op exists for `name`.
  bool containsPrimitive(llvm::StringRef name) const;

  /// Returns the `mogg.shape_func.decl` op whose `registration` matches
  /// `name`, or a null op if none exists.
  ShapeFunctionDeclOp lookupShapeFunction(llvm::StringRef name) const;

  /// Returns the `mogg.shape_func.decl` op whose `sym_name` matches
  /// `symName`. Asserts that it exists.
  ShapeFunctionDeclOp
  lookupShapeFunctionBySymName(llvm::StringRef symName) const;

  /// Returns true if a `mogg.shape_func.decl` op exists for `name`.
  bool containsShapeFunction(llvm::StringRef name) const;

  /// Resolves the kernel to use for `name`, applying user-over-builtin override
  /// semantics. Asserts that the kernel exists; use `containsKernel` for an
  /// existence check. Also asserts that at most one user-defined decl claims
  /// the registration (the `ImportPackages` import-time dedup keeps the
  /// invariant).
  KernelDeclOp getResolvedKernel(llvm::StringRef name) const;

private:
  llvm::StringMap<llvm::SmallVector<KernelDeclOp>> kernelByName;
  llvm::StringMap<KernelDeclOp> kernelBySymName;
  llvm::StringMap<PrimitiveDeclOp> primitiveByName;
  llvm::StringMap<PrimitiveDeclOp> primitiveBySymName;
  llvm::StringMap<ShapeFunctionDeclOp> shapeFuncByName;
  llvm::StringMap<ShapeFunctionDeclOp> shapeFuncBySymName;
};

/// Clone every `mogg.kernel.decl` and `mogg.primitive.decl` op from `src`
/// into `dst`. Used when a fresh sub-module needs the same kernel registry as
/// an existing module (e.g. constant-folding sub-modules) without re-running
/// `--mogg-import-packages` from scratch — the decl ops are fully resolved
/// and safe to copy.
///
/// Existing decls in `dst` are left untouched. Callers that need
/// deduplication should run `--mogg-import-packages` with `--append=true` on
/// `dst` afterward, or just avoid double-population.
void copyKernelDecls(mlir::ModuleOp src, mlir::ModuleOp dst);

/// Verify that an MO operation matches the signature of its registered Mojo
/// kernel. Reuses an already-built `MojoRegistry`. Prefer this overload from
/// long-running consumers (Python `Analysis`) so the registry doesn't have
/// to be rebuilt on every call.
ErrorOrSuccess verifyCustomOp(const MojoRegistry &registry,
                              mlir::Operation *op);

/// Convenience overload that builds a fresh `MojoRegistry` over
/// `kernelDecls` and delegates. `kernelDecls` is the module that holds the
/// `mogg.kernel.decl` / `mogg.primitive.decl` ops (typically populated by
/// `--mogg-import-packages` or by `copyKernelDecls`). Backs the public C
/// API `MAXG_verifyCustomOp`.
ErrorOrSuccess verifyCustomOp(mlir::ModuleOp kernelDecls, mlir::Operation *op);

/// Resolve the filesystem paths of the built-in Mojo kernel packages
/// (`kernels` and `primitives`). Used for cache-key construction
/// (the cache invalidates when these binaries change). Returns an error if
/// the runtime mojo path resolver can't find them.
ErrorOr<llvm::SmallVector<std::string>>
getBuiltInKernelPackagePaths(mlir::MLIRContext *ctx);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJOREGISTRY_H
