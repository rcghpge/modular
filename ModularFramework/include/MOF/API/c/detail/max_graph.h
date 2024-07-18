//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOF_API_C_DETAIL_MAX_GRAPH_H
#define MOF_API_C_DETAIL_MAX_GRAPH_H

#include "MOF/API/c/symbol_export.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir-c/IR.h"

/// An opaque wrapper around a pointer to an AsyncRT::Runtime.
struct AsyncRTRuntimeRef {
  void *ptr;
};

/// Contains the success or failure of an API call.
///
/// In general, any API that may fail accepts a `MAXG_Status` argument
/// that is filled in with a meaningful error message on failure.
///
/// Construct a `MAXG_Status` with `MAXG_newStatus()`.
/// Deallocate it with `MAXG_freeStatus()`.
typedef struct MAXG_Status MAXG_Status;

/// Creates a new status object.
MODULAR_API_EXPORT MAXG_Status *MAXG_newStatus();

/// Gets an error message from a status object.
/// The returned pointer is owned by the status and only valid for the lifetime
/// of the status.
MODULAR_API_EXPORT const char *MAXG_getError(const MAXG_Status *status);

/// Checks if status holds an error value.
MODULAR_API_EXPORT int MAXG_isError(const MAXG_Status *status);

/// Deallocates the memory for the status object.  No-op if `status` is NULL.
MODULAR_API_EXPORT void MAXG_freeStatus(MAXG_Status *status);

/// Loads all MLIR dialects needed for building MAX graphs.
MODULAR_API_EXPORT void MAXG_loadModularDialects(MlirDialectRegistry registry);

//===----------------------------------------------------------------------===//
// Op factories
//===----------------------------------------------------------------------===//

/// Creates a graph op at the start of the module's body region and returns it.
MODULAR_API_EXPORT MlirOperation MAXG_graphNew(MlirModule m, MlirLocation loc,
                                               llvm::StringRef name,
                                               MlirType signature);

//===----------------------------------------------------------------------===//
// Attribute factories
//===----------------------------------------------------------------------===//

/// Creates a dense_resource attribute backed by an mmap'ed read-only file.
MODULAR_API_EXPORT MlirNamedAttribute MAXG_weightsAttr(const char *filepath,
                                                       size_t offset,
                                                       MlirType type,
                                                       llvm::StringRef name,
                                                       MAXG_Status *status);

/// Creates a named ArrayElementsAttr (dense_array) backed by a builtin tensor.
/// The tensor data is always borrowed.
MODULAR_API_EXPORT MlirNamedAttribute
MAXG_attrNewTensorBorrowed(llvm::StringRef name, void *data, MlirType type);

/// Creates a named ArrayElementsAttr (dense_array) backed by a builtin tensor.
MODULAR_API_EXPORT MlirNamedAttribute
MAXG_attrNewTensor(llvm::StringRef name, void *data, MlirType t, bool owned,
                   AsyncRTRuntimeRef runtime);

/// Creates a ParamDeclAttr (param.decl) with the given name and index type.
MODULAR_API_EXPORT MlirAttribute MAXG_attrNewDimParamDecl(MlirContext ctx,
                                                          llvm::StringRef name);

/// Creates a ParamDeclArrayAttr (param.decls) from an array of ParamDeclAttrs.
MODULAR_API_EXPORT MlirAttribute
MAXG_attrNewParamDeclArray(MlirContext ctx, MlirAttribute *attrs, int32_t size);

/// Creates a MOSH ShapeAttr (#mosh<ape[]>).
MODULAR_API_EXPORT MlirAttribute MAXG_attrNewShape(MlirContext ctx,
                                                   MlirAttribute *dims,
                                                   int32_t rank);

//===----------------------------------------------------------------------===//
// Type helpers
//===----------------------------------------------------------------------===//

/// Converts the given M::DType corresponding to `value` to an MLIR type.
MODULAR_API_EXPORT MlirType MAXG_dTypeNewFromString(MlirContext ctx,
                                                    const char *value,
                                                    MAXG_Status *status);

/// Converts the given M::DType corresponding to `value` to an MLIR type.
MODULAR_API_EXPORT MlirType MAXG_dTypeNew(MlirContext ctx, uint8_t value);

/// Creates a new MO::TensorType (!mo.tensor).
MODULAR_API_EXPORT MlirType MAXG_tensorTypeNew(MlirContext ctx, MlirType dtype,
                                               bool ranked, MlirAttribute *dims,
                                               int32_t rank);

/// Accessors for !mo.tensor.
/// Gets the DType as a null-terminated string owned by the caller.
MODULAR_API_EXPORT char *MAXG_tensorTypeGetDTypeAsString(MlirType type);
MODULAR_API_EXPORT uint8_t MAXG_tensorTypeGetDType(MlirType t);
MODULAR_API_EXPORT int64_t MAXG_tensorTypeGetRank(MlirType t);
MODULAR_API_EXPORT MlirAttribute MAXG_tensorTypeShapeGetDim(MlirType t,
                                                            int64_t dim);

/// Create static or symbolic dimension attributes.
MODULAR_API_EXPORT MlirAttribute MAXG_dimNewStatic(MlirContext context,
                                                   int64_t dim);
MODULAR_API_EXPORT MlirAttribute MAXG_dimNewSymbolic(MlirContext context,
                                                     llvm::StringRef name);

/// Query functions for the type of a given dim.
MODULAR_API_EXPORT bool MAXG_dimIsSymbolic(MlirAttribute a);
MODULAR_API_EXPORT bool MAXG_dimIsSymbolicExpression(MlirAttribute a);
MODULAR_API_EXPORT bool MAXG_dimIsStatic(MlirAttribute a);

/// Accessors for dim attributes.
MODULAR_API_EXPORT int64_t MAXG_dimStaticValue(MlirAttribute a);
MODULAR_API_EXPORT MlirIdentifier MAXG_dimSymbolicName(MlirAttribute a);

/// Query functions for tensor and opaque types.
MODULAR_API_EXPORT bool MAXG_typeIsTensor(MlirType t);
MODULAR_API_EXPORT bool MAXG_typeIsOpaque(MlirType t);

/// Creates a new MO::Opaque type (!mo.opaque).
MODULAR_API_EXPORT MlirType MAXG_opaqueTypeNew(MlirContext context,
                                               llvm::StringRef name);

/// Accessors for !mo.opaque.
MODULAR_API_EXPORT MlirStringRef MAXG_opaqueTypeName(MlirType t);

#endif // MOF_API_C_DETAIL_MAX_GRAPH_H
