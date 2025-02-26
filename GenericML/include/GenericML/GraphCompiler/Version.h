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

/// Get a unique string that embeds the binary version of the MOGG compiler
/// implementation. The string is a hex representation of the hash of the
/// version string.
std::string getMOGGCompilerVersionString();

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_VERSION_H
