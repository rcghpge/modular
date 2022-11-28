//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MOPPRIMITIVES_MOPPRIMITIVES_H
#define GENERICML_MOPPRIMITIVES_MOPPRIMITIVES_H

namespace M {
class PrimitiveRegistry;

/// This registers the 'mop' dialect primitives.
void registerMOPPrimitives(PrimitiveRegistry &registry);

} // namespace M

#endif // GENERICML_MOPPRIMITIVES_MOPPRIMITIVES_H
