//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares types for the MOGG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_SUPPORT_TYPETRAITS_H
#define GENERICML_GRAPHCOMPILER_SUPPORT_TYPETRAITS_H

#include "Mojo/POPDialect/POPTypes.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Types.h"
#include <string>

//===----------------------------------------------------------------------===//
// MO supported types and conversion to KGEN types
//
// If you want to add a new supported type, add it to TypeTraits.cpp.
//===----------------------------------------------------------------------===//

namespace M::MOGG {

// This struct implements functions around converting mlir::Type to DType.
// Multiple mlir::Types can map to the same DType.
struct TypeTraits {
  // Type name in Mojo. This will usually be something longer like float32 or
  // uint8.
  const llvm::StringLiteral mojoTypeName;
  // KGENDType that moggType will lower to
  const KGEN::KGENDType kgenType;
  // The MOGG dtype that this class implements conversion functions for. No two
  // implementations of TypeTraits should share the same moggType.
  const DType moggType;
  // Convert array of this dtype to an array of simd values.
  llvm::SmallVector<mlir::TypedAttr> convertToSIMD(mlir::MLIRContext *context,
                                                   const void *data,
                                                   int64_t numElems) const;

  // Lookup the traits for a given mlir::Type. This should be an element type,
  // not a tensor type.
  static ErrorOr<TypeTraits> lookupType(Type);
  // Lookup the traits for a given DType
  static ErrorOr<TypeTraits> lookupDType(DType);
  // Checks if a Type has a corresponding DType.
  static bool isSupportedMOGGType(Type);

  TypeTraits(llvm::StringLiteral mojoTypeName, KGEN::KGENDType kgenType,
             DType moggType)
      : mojoTypeName(mojoTypeName), kgenType(kgenType), moggType(moggType) {}
};

// A list of all the registered type traits.
llvm::SmallVector<TypeTraits> allTypeTraits();

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_SUPPORT_TYPETRAITS_H
