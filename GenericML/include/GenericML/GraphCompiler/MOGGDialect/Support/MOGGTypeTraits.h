//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares types for the MOGG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_SUPPORT_MOGGTYPETRAITS_H
#define GENERICML_GRAPHCOMPILER_SUPPORT_MOGGTYPETRAITS_H

#include "KGEN/POPDialect/POPTypes.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Types.h"
#include <string>

//===----------------------------------------------------------------------===//
// MO supported types and conversion to KGEN types
//
// If you want to add a new supported type, add it to MOGGTypeTraits.cpp.
//===----------------------------------------------------------------------===//

namespace M::MOGG {

// This struct implements functions around converting mlir::Type to DType.
// Multiple mlir::Types can map to the same DType.
struct MOGGTypeTraits {
  // Type name in Mojo. This will usually be something longer like float32 or
  // uint8.
  const llvm::StringLiteral mojoTypeName;
  // KGENDType that moggType will lower to
  const KGEN::KGENDType kgenType;
  // The MOGG dtype that this class implements conversion functions for. No two
  // implementations of MOGGTypeTraits should share the same moggType.
  const DType moggType;
  // Convert array of this dtype to an array of simd values.
  llvm::SmallVector<mlir::TypedAttr> convertToSIMD(mlir::MLIRContext *context,
                                                   const void *data,
                                                   int64_t numElems) const;

  // Lookup the traits for a given mlir::Type. This should be an element type,
  // not a tensor type.
  static ErrorOr<MOGGTypeTraits> lookupType(Type);
  // Lookup the traits for a given DType
  static ErrorOr<MOGGTypeTraits> lookupDType(DType);
  // Checks if a Type has a corresponding DType.
  static bool isSupportedMOGGType(Type);
  // Simd type for a given tensor type.
  static ErrorOr<Type> simdTypeForTensor(Type type);

  MOGGTypeTraits(llvm::StringLiteral mojoTypeName, KGEN::KGENDType kgenType,
                 DType moggType)
      : mojoTypeName(mojoTypeName), kgenType(kgenType), moggType(moggType) {}
};

// A list of all the registered type traits.
llvm::SmallVector<MOGGTypeTraits> allTypeTraits();

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_SUPPORT_MOGGTYPETRAITS_H
