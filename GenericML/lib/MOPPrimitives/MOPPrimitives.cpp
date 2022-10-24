//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MOP/MOPPrimitives.h"
#include "GenericML/MOP/Context.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"

using namespace M;

//===----------------------------------------------------------------------===//
// mop.context.create
//===----------------------------------------------------------------------===//

// Returns a mop context initialized with `numPrimContexts` primitive
// contexts.
static void mop_context_create(Result<MOP::Context> result,
                               BEFAttr<uint32_t> numPrimContexts) {
  result->emplace<MOP::Context>(*numPrimContexts);
}

//===----------------------------------------------------------------------===//
// Primitive registration
//===----------------------------------------------------------------------===//

void mop_model() {}

void M::registerMOPPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "mop.model", mop_model);
  ADD_PRIMITIVE(registry, "mop.context.create", mop_context_create);
}
