//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "max/mcl/MOF/Driver/DriverImpl.h"

#include "max/mcl/MOF/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

using namespace M;
using namespace M::MOF;

OwningOpRef<ModuleOp> MOF::parseMODialect(M::BufferRef buffer,
                                          MLIRContext *ctx) {
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  ctx->appendDialectRegistry(registry);
  ctx->allowUnregisteredDialects(true);
  auto filePathOr = buffer->getFilePath();
  if (filePathOr)
    return mlir::parseSourceString<ModuleOp>(buffer->getBuffer(), ctx,
                                             filePathOr->string());
  else
    return mlir::parseSourceString<ModuleOp>(buffer->getBuffer(), ctx);
}
