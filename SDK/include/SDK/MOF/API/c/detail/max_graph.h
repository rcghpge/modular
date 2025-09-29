//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOF_API_C_DETAIL_MAX_GRAPH_H
#define MOF_API_C_DETAIL_MAX_GRAPH_H

#include "SDK/MOF/API/c/symbol_export.h"
#include "mlir-c/IR.h"

/// Loads all MLIR dialects needed for building MAX graphs.
MODULAR_API_EXPORT void MAXG_loadModularDialects(MlirDialectRegistry registry);

//===----------------------------------------------------------------------===//
// Attribute factories
//===----------------------------------------------------------------------===//

/// Creates an opaque location for a python frame.
MODULAR_API_EXPORT MlirLocation MAXG_opaqueFrameLocNew(
    MlirContext context, void *frame_ptr, char *(*to_str_fn)(void *),
    void (*incref_fn)(void *), void (*decref_fn)(void *));

#endif // MOF_API_C_DETAIL_MAX_GRAPH_H
