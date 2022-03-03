//===- GraphRT/Primitives.h -------------------------------------*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the interfaces to register the primitives in the GraphRT
// library.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_PRIMITIVES_H
#define GRAPHRT_PRIMITIVES_H

namespace M {
class PrimitiveRegistry;

/// This registers the 'test' dialect primitives, intended for writing executor
/// testcases.
void registerGraphRTTestPrimitives(PrimitiveRegistry &registry);

/// This registers primitives used for core control flow operations.
void registerGraphRTControlFlowPrimitives(PrimitiveRegistry &registry);

/// Finalizes the registry and allows it to be queried.
void finalizePrimitiveRegistry(PrimitiveRegistry &registry);

/// TODO: Control flow, integer, etc primitives.

/// This registers all the primitive libraries in the GraphRT subsystem.
inline void registerAllGraphRTPrimitives(PrimitiveRegistry &registry) {
  registerGraphRTTestPrimitives(registry);
  registerGraphRTControlFlowPrimitives(registry);
  finalizePrimitiveRegistry(registry);
}

} // namespace M

#endif // GRAPHRT_PRIMITIVES_H