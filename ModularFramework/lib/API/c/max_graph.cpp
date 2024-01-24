//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MOF/API/c/detail/max_graph.h"
#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "MOF/API/c/symbol_export.h"
#include "MOF/GraphBuilder/Attr.h"
#include "MOF/GraphBuilder/Graph.h"
#include "MOF/GraphBuilder/Loc.h"
#include "MOF/GraphBuilder/Module.h"
#include "MOF/GraphBuilder/Symbol.h"
#include "MOF/GraphBuilder/Type.h"
#include "Support/Buffer.h"
#include "Support/ML/TensorSpec.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/bit.h"

using namespace M;
using namespace M::MOF::GraphBuilder;
using namespace llvm;
using namespace mlir;

static char *unownedString(const std::string &buffer, int64_t *len) {
  *len = buffer.size() + 1; // Include terminator.
  char *cstr = reinterpret_cast<char *>(std::malloc(*len));
  std::strcpy(cstr, buffer.c_str());
  return cstr;
}

static MLIRContext *newContext() {
  // TODO: Attach the context to modules instead? Let user control it?
  static MLIRContext ctx;
  mlir::DialectRegistry registry;
  M::MO::loadDialects(registry);
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();
  return &ctx;
}

// Note: Keep sections below in sync with max_graph.cpp.

//===----------------------------------------------------------------------===//
// Attribute factories
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT M_Attr *MAXG_attrNewTensor(M_Module *m, StringRef name,
                                              void *data, M_Type *t,
                                              bool owned) {
  auto moType = cast<M::MO::TensorType>(t->type.getVal());
  size_t size =
      moType.getDtype().getSizeInBytes() * moType.getNumElements().value();
  return new M_Attr{m->module.tensorAttr(
      name, ArrayRef<uint8_t>(reinterpret_cast<uint8_t *>(data), size), t->type,
      owned)};
}

MODULAR_API_EXPORT M_Attr *MAXG_attrNewTensorFromFile(M_Module *m,
                                                      StringRef name,
                                                      StringRef fileName,
                                                      M_Type *t) {
  auto moType = cast<M::MO::TensorType>(t->type.getVal());
  assert(moType.getNumElements().has_value() &&
         "tensor attrs should always be statically sized");
  int64_t size =
      moType.getDtype().getSizeInBytes() * moType.getNumElements().value();

  auto bufferOr = M::Buffer::getFile(std::filesystem::path{fileName.str()});
  if (bufferOr.isError())
    errs() << bufferOr.takeError() << "\n";
  assert(!bufferOr.isError());
  M::BufferRef buffer = bufferOr.takeValue();

  auto buff_size = static_cast<int64_t>(buffer->getBufferSize());
  if (buff_size != size)
    errs() << "Size mismatch, want " << size << " got " << buff_size << "\n";
  assert(buff_size == size);

  auto data = buffer->getBufferStart();
  return new M_Attr{m->module.tensorAttr(
      name,
      ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(data),
                        static_cast<size_t>(size)),
      t->type, /*owned=*/false)};
}

MODULAR_API_EXPORT M_Attr *MAXG_attrNewString(M_Module *m, StringRef name,
                                              StringRef value) {
  return new M_Attr{m->module.stringAttr(name, value)};
}

//===----------------------------------------------------------------------===//
// AttrMap support
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT M_AttrMap *MAXG_attrMapNew() { return new M_AttrMap(); }

MODULAR_API_EXPORT uint32_t MAXG_attrMapSize(M_AttrMap *m) {
  return m->map.size();
}

MODULAR_API_EXPORT void MAXG_attrMapAddAttr(M_AttrMap *m, M_Attr *a) {
  m->map.add(a->attr);
}

//===----------------------------------------------------------------------===//
// Graph support
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT M_Graph *MAXG_graphNew(M_Module *m, M_Loc *l,
                                          llvm::StringRef name,
                                          const M_Arity *inTypes,
                                          const M_Arity *outTypes) {
  return new M_Graph{
      m->module.newGraph(l->loc, name, inTypes->arity, outTypes->arity)};
}

MODULAR_API_EXPORT M_Module *MAXG_graphGetModule(M_Graph *g) {
  return new M_Module{g->graph.getModule()};
}

MODULAR_API_EXPORT M_Symbol *MAXG_graphGetArg(M_Graph *g, uint32_t pos) {
  return new M_Symbol{g->graph.getArg(pos)};
}

MODULAR_API_EXPORT M_Tuple *MAXG_graphNewOp(M_Graph *g, M_Loc *l,
                                            llvm::StringRef name,
                                            const M_Tuple *inputs,
                                            const M_Arity *outTypes,
                                            const M_AttrMap *attrs) {
  return new M_Tuple{
      g->graph.newOp(l->loc, name, inputs->tuple, outTypes->arity, attrs->map)};
}

//===----------------------------------------------------------------------===//
// Location helpers
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT M_Loc *MAXG_locNewUnknown(M_Module *m) {
  return new M_Loc{m->module.unknownLoc()};
}

//===----------------------------------------------------------------------===//
// Module support
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT M_Module *MAXG_moduleNew() {
  static MLIRContext *ctx = newContext();
  Location testLoc = UnknownLoc::get(ctx);

  auto *m = new M_Module{Module(testLoc)};
  return m;
}

MODULAR_API_EXPORT bool MAXG_moduleVerify(M_Module *m) {
  return mlir::verify(m->module.getModuleOp()).succeeded();
}

MODULAR_API_EXPORT char *MAXG_moduleToString(const M_Module *const m,
                                             int64_t *len) {
  std::string buffer;
  raw_string_ostream os(buffer);
  m->module.getModuleOp()->print(os);

  return unownedString(buffer, len);
}

MODULAR_API_EXPORT bool MAXG_moduleToBytecode(const M_Module *const m,
                                              StringRef fileName) {
  std::error_code ec;
  raw_fd_stream fs(fileName, ec);
  if (ec)
    return false;
  return writeBytecodeToFile(m->module.getModuleOp(), fs).succeeded();
}

//===----------------------------------------------------------------------===//
// Symbol support
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT M_Graph *MAXG_symbolGetGraph(M_Symbol *s) {
  return new M_Graph{s->symbol.getGraph()};
}

MODULAR_API_EXPORT char *MAXG_symbolToString(M_Symbol *s, int64_t *len) {
  return unownedString(s->symbol.toString(), len);
}

MODULAR_API_EXPORT M_Tuple *MAXG_tupleNew() { return new M_Tuple{Tuple()}; }

MODULAR_API_EXPORT uint32_t MAXG_tupleSize(M_Tuple *t) {
  return t->tuple.size();
}

MODULAR_API_EXPORT void MAXG_tupleAddSymbol(M_Tuple *t, M_Symbol *s) {
  t->tuple.add(s->symbol);
}

MODULAR_API_EXPORT M_Symbol *MAXG_getSymbol(M_Tuple *t, uint32_t pos) {
  return new M_Symbol{t->tuple.getItem(pos)};
}

//===----------------------------------------------------------------------===//
// Type helpers
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT M_Type *MAXG_dTypeNew(M_Module *m, uint8_t value) {
  return new M_Type{m->module.newDType(M::DType(value))};
}

MODULAR_API_EXPORT int64_t MAXG_dimTypeNewDynamic() {
  return ShapedType::kDynamic;
}

MODULAR_API_EXPORT M_Type *MAXG_tensorTypeNew(M_Module *m, M_Type *dtype,
                                              bool ranked, int64_t *dims,
                                              int32_t rank) {
  return new M_Type{m->module.newTensorType(
      dtype->type, ranked, llvm::ArrayRef<int64_t>(dims, rank))};
}

MODULAR_API_EXPORT uint8_t MAXG_tensorTypeGetDType(M_Symbol *s) {
  return s->symbol.getTensorDType().getValue();
}

MODULAR_API_EXPORT int64_t *MAXG_tensorTypeGetShape(M_Symbol *s,
                                                    int32_t *rank) {
  SmallVector<int64_t> dims = s->symbol.getTensorShape();
  *rank = dims.size();
  int64_t *retval = new int64_t[*rank];
  for (size_t i = 0; i < dims.size(); i++)
    retval[i] = dims[i];
  return retval;
}

MODULAR_API_EXPORT bool MAXG_tensorTypeIsRanked(M_Symbol *s) {
  return s->symbol.getTensorDType().getValue();
}

MODULAR_API_EXPORT char *MAXG_typeToString(M_Type *t, int64_t *len) {
  return unownedString(t->type.toString(), len);
}

MODULAR_API_EXPORT M_Arity *MAXG_arityNew() { return new M_Arity{Arity()}; }

MODULAR_API_EXPORT uint32_t MAXG_aritySize(M_Arity *a) {
  return a->arity.size();
}

MODULAR_API_EXPORT void MAXG_arityAddType(M_Arity *a, M_Type *t) {
  a->arity.add(t->type);
}
