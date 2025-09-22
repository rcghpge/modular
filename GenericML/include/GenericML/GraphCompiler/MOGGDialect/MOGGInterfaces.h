//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGINTERFACES_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGINTERFACES_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/MOGGInterfaces.h.inc"

namespace M::MOGG {

/// Build the typeMethodsDict attr for the MojoObjectInfosInterface from a
/// kernel function `mojoKernel`. Returns the dictionary of needed methods of
/// the output type. It must have a decorator like @mogg_kernel to have the I/O
/// vtable attrs defined. This function doesn't get the whole table, but only
/// the methods we are interested (to avoid bloating the IR). `methods` list all
/// methods that will be added to the VTable. Returns failure if there isn't any
/// type information for this input / output, or if one of the requested methods
/// isn't found.
FailureOr<DictionaryAttr>
buildMojoObjectInfosOutputMethodsDict(Operation *mojoKernelOp,
                                      llvm::ArrayRef<StringRef> methods);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_H
