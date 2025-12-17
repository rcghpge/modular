//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#ifndef GENERICML_MEFMODEL_MEFMODEL_H
#define GENERICML_MEFMODEL_MEFMODEL_H

#include "GenericML/Support/ModelCollection.h"
#include "MLRT/GraphRT/MEFExecutor/MEFFile.h"
#include "Support/ForwardDecls.h"
#include "Support/RCRef.h"
#include "Support/SymbolExport.h"
#include <memory>

namespace M::MO {

/// Load all the models from the contents of the specified MEFFile, which
/// holds them compiled to MEF format. Assumes the
/// MemoryBuffer will outlive the ModelCollection.
DRIVER_VISIBILITY_EXPORT // for MT
    ErrorOr<std::unique_ptr<M::ModelCollection>>
    loadMGPMEFFile(RCRef<MEFFile> mefFile);

} // namespace M::MO

#endif // GENERICML_MEFMODEL_MEFMODEL_H
