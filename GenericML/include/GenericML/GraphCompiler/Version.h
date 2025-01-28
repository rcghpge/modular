//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_VERSION_H
#define GENERICML_GRAPHCOMPILER_VERSION_H

#include <string>

namespace M::MO {

/// Get a unique string that embeds the binary version of the graph compiler
/// implementation.
std::string getGraphCompilerVersionString();

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_VERSION_H
