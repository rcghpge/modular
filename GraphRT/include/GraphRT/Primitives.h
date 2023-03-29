//===----------------------------------------------------------------------===//
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

/// This registers the 'graphrt' dialect primitives for working with integers.
void registerGraphRTIntegerPrimitives(PrimitiveRegistry &registry);

/// This registers the 'mip' dialect primitives for working with index values.
void registerMipPrimitives(PrimitiveRegistry &registry);

/// This registers the 'graphrt' dialect primitives for control flow.
void registerGraphRTControlFlowPrimitives(PrimitiveRegistry &registry);

/// This registers the 'grt' dialect primitives for benchmarking.
void registerGraphRTBenchmarkingPrimitives(PrimitiveRegistry &registry);

/// This registers all the primitive libraries in the GraphRT subsystem.
inline void registerAllGraphRTPrimitives(PrimitiveRegistry &registry) {
  registerGraphRTTestPrimitives(registry);
  registerMipPrimitives(registry);
  registerGraphRTIntegerPrimitives(registry);
  registerGraphRTControlFlowPrimitives(registry);
  registerGraphRTBenchmarkingPrimitives(registry);
}

} // namespace M

#endif // GRAPHRT_PRIMITIVES_H
