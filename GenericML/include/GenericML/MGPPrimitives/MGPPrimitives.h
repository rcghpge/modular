//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPPRIMITIVES_MGPPRIMITIVES_H
#define GENERICML_MGPPRIMITIVES_MGPPRIMITIVES_H

namespace M {
class PrimitiveRegistry;

/// Registers the device-generic subset of the 'mgp' dialect primitives.
void registerMGPDevicePrimitives(PrimitiveRegistry &registry);

/// Registers all the 'mgp' dialect primitives.
void registerMGPPrimitives(PrimitiveRegistry &registry);

} // namespace M

#endif // GENERICML_MGPPRIMITIVES_MGPPRIMITIVES_H
