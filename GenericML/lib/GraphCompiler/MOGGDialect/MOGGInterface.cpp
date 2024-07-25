//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/MOGGInterfaces.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MOGGPreElab/MOGGDecorators.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/MOGGInterfaces.cpp.inc"

namespace M::MOGG {

FailureOr<DictionaryAttr>
buildMojoObjectInfosOutputMethodsDict(Operation *mojoKernelOp,
                                      llvm::ArrayRef<StringRef> methods) {
  auto mojoKernel = cast<KGEN::GeneratorOp>(mojoKernelOp);
  KGEN::ParameterExprArrayAttr conformancesAttr =
      dyn_cast_or_null<KGEN::ParameterExprArrayAttr>(
          mojoKernel->getAttr(KGEN::MOGGPreElab::MOGG_RESULT_CONFORMANCES));
  if (!conformancesAttr) {
    return mojoKernel.emitError()
           << "no MOGG type conformance informations found (maybe kernel is "
              "missing a register decorator like `@mogg_register`)";
  }

  auto findMethod =
      [&](StringRef methodName) -> FailureOr<KGEN::VTableEntryAttr> {
    for (TypedAttr attr : conformancesAttr) {
      auto typeAttr = cast<KGEN::TypeConstantAttr>(attr);
      for (KGEN::VTableEntryAttr entry : typeAttr.getVTable().getEntries()) {
        if (entry.getName() == methodName)
          return entry;
      }
    }
    return mojoKernel.emitError()
           << "kernel output type doesn't implement the method `" << methodName
           << "`. Maybe it's missing a trait implementation.";
  };

  // Build the dictionary of methods
  NamedAttrList methodsDict;
  for (StringRef methName : methods) {
    FailureOr<KGEN::VTableEntryAttr> entryOr = findMethod(methName);
    if (failed(entryOr))
      return failure();
    methodsDict.set(methName, entryOr->getMethod());
  }
  return DictionaryAttr::get(mojoKernel.getContext(), methodsDict);
}

} // namespace M::MOGG
