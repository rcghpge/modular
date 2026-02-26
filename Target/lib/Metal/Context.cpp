//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"

using namespace M::Target::Metal;

Context *Context::create(Device *device) { return new Context(device); }

void Context::destroy(Context *context) { delete context; }
