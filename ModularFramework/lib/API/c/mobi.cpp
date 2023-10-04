//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "MOF/API/c/mobi.h"

#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "MOF/API/c/detail/mobi.h"
#include "MOF/MOBI/Graph.h"
#include "MOF/MOBI/Loc.h"
#include "MOF/MOBI/Module.h"
#include "MOF/MOBI/Symbol.h"
#include "MOF/MOBI/Type.h"
#include "mlir/IR/MLIRContext.h"

using namespace M::MOF::Builder;
using namespace llvm;
using namespace mlir;

using namespace M::MOF::Builder;

/// Graph

MODULAR_API_EXPORT M_Graph *MOBI_newGraph(M_Module *m, M_Loc *l,
                                          llvm::StringRef name,
                                          const M_Arity *inTypes,
                                          const M_Arity *outTypes) {
  return new M_Graph{
      m->module.newGraph(l->loc, name, inTypes->arity, outTypes->arity)};
}

MODULAR_API_EXPORT M_Symbol *MOBI_getGraphInput(M_Graph *g, uint32_t pos) {
  return new M_Symbol{g->graph.getArg(pos)};
}

MODULAR_API_EXPORT M_Tuple *MOBI_newOp(M_Graph *g, M_Loc *l,
                                       llvm::StringRef name,
                                       const M_Tuple *inputs,
                                       const M_Arity *outTypes) {
  return new M_Tuple{
      g->graph.newOp(l->loc, name, inputs->tuple, outTypes->arity)};
}

/// Location

MODULAR_API_EXPORT M_Loc *MOBI_unknownLoc(M_Module *m) {
  return new M_Loc{m->module.unknownLoc()};
}

/// Module

static MLIRContext *newContext() {
  // TODO: Attach the context to modules instead? Let user control it?
  static MLIRContext ctx;
  ctx.loadDialect<M::MO::MODialect>();
  return &ctx;
}

MODULAR_API_EXPORT M_Module *MOBI_newModule() {
  static MLIRContext *ctx = newContext();
  Location testLoc = UnknownLoc::get(ctx);

  auto *m = new M_Module{Module(testLoc)};
  return m;
}

MODULAR_API_EXPORT void MOBI_debugPrintModule(const M_Module *const m) {
  m->module.getModuleOp()->print(outs());
}

/// Symbol

MODULAR_API_EXPORT M_Tuple *MOBI_newTuple() { return new M_Tuple{Tuple()}; }

MODULAR_API_EXPORT void MOBI_extendTuple(M_Tuple *t, M_Symbol *s) {
  t->tuple.add(s->symbol);
}

/// Type

MODULAR_API_EXPORT M_Type *MOBI_si32Scalar(M_Module *m) {
  return new M_Type{m->module.si32Scalar()};
}

MODULAR_API_EXPORT M_Arity *MOBI_newArity() { return new M_Arity{Arity()}; }

MODULAR_API_EXPORT void MOBI_increaseArity(M_Arity *a, M_Type *t) {
  a->arity.add(t->type);
}
