//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MODULARMOUDEL_TFMODELFILE_H
#define MODULARMOUDEL_TFMODELFILE_H

#include "Support/ForwardDecls.h"
#include <memory>

namespace M {
class MEFFile;
class ModelCollection;
} // namespace M

namespace M::MOF {

/// Load all the models from the contents of the specified MEFFile, which
/// holds them compiled to MEF format. Assumes the
/// MemoryBuffer will outlive the ModelCollection.
ErrorOr<std::unique_ptr<ModelCollection>>
loadMGPMEFFile(RCRef<M::MEFFile> mefFile);

} // namespace M::MOF

#endif // MODULARMOUDEL_TFMODELFILE_H
