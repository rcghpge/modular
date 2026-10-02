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

// Converts mlir::Type to DType. Multiple mlir::Types can map to the same DType.
struct TypeTraits {
  // Type name in Mojo, e.g. float32 or uint8.
  const llvm::StringLiteral mojoTypeName;
  const KGEN::KGENDType kgenType;
  // Unique across all TypeTraits.
  const DType moggType;
  llvm::SmallVector<mlir::TypedAttr> convertToSIMD(mlir::MLIRContext *context,
                                                   const void *data,
                                                   int64_t numElems) const;

  // Takes an element type, not a tensor type.
  static ErrorOr<TypeTraits> lookupType(Type);
  static ErrorOr<TypeTraits> lookupDType(DType);
  static bool isSupportedMOGGType(Type);

  TypeTraits(llvm::StringLiteral mojoTypeName, KGEN::KGENDType kgenType,
             DType moggType)
      : mojoTypeName(mojoTypeName), kgenType(kgenType), moggType(moggType) {}
};

llvm::SmallVector<TypeTraits> allTypeTraits();

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_SUPPORT_TYPETRAITS_H
