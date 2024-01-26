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
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
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

MlirNamedAttribute wrap(mlir::NamedAttribute attr) {
  return MlirNamedAttribute{
      wrap(attr.getName()),
      wrap(attr.getValue()),
  };
}

mlir::NamedAttribute unwrap(MlirNamedAttribute attr) {
  return {unwrap(attr.name), unwrap(attr.attribute)};
}

MODULAR_API_EXPORT void MAXG_loadModularDialects(MlirDialectRegistry registry) {
  M::MO::loadDialects(*unwrap(registry));
}

// Note: Keep sections below in sync with max_graph.cpp.

//===----------------------------------------------------------------------===//
// Attribute factories
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT MlirNamedAttribute MAXG_attrNewTensor(MlirModule m,
                                                         StringRef name,
                                                         void *data, MlirType t,
                                                         bool owned) {
  auto moType = cast<M::MO::TensorType>(unwrap(t));
  size_t size =
      moType.getDtype().getSizeInBytes() * moType.getNumElements().value();
  return wrap(Module(unwrap(m)).tensorAttr(
      name, ArrayRef<uint8_t>(reinterpret_cast<uint8_t *>(data), size),
      unwrap(t), owned));
}

MODULAR_API_EXPORT MlirNamedAttribute MAXG_attrNewTensorFromFile(
    MlirModule m, StringRef name, StringRef fileName, MlirType t) {
  auto moType = cast<M::MO::TensorType>(unwrap(t));
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
  return wrap(Module(unwrap(m)).tensorAttr(
      name,
      ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(data),
                        static_cast<size_t>(size)),
      unwrap(t), /*owned=*/false));
}

MODULAR_API_EXPORT MlirNamedAttribute MAXG_attrNewString(MlirModule m,
                                                         StringRef name,
                                                         StringRef value) {
  return wrap(Module(unwrap(m)).stringAttr(name, value));
}

//===----------------------------------------------------------------------===//
// AttrMap support
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT M_AttrMap *MAXG_attrMapNew() { return new M_AttrMap(); }

MODULAR_API_EXPORT uint32_t MAXG_attrMapSize(M_AttrMap *m) {
  return m->map.size();
}

MODULAR_API_EXPORT void MAXG_attrMapAddAttr(M_AttrMap *m,
                                            MlirNamedAttribute a) {
  m->map.add(unwrap(a));
}

//===----------------------------------------------------------------------===//
// Graph support
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT M_Graph *MAXG_graphNew(MlirModule m, MlirLocation l,
                                          llvm::StringRef name,
                                          const M_Arity *inTypes,
                                          const M_Arity *outTypes) {
  return new M_Graph{Module(unwrap(m)).newGraph(unwrap(l), name, inTypes->arity,
                                                outTypes->arity)};
}

MODULAR_API_EXPORT MlirModule MAXG_graphGetModule(M_Graph *g) {
  return wrap(g->graph.getModule().getModuleOp());
}

MODULAR_API_EXPORT M_Symbol *MAXG_graphGetArg(M_Graph *g, uint32_t pos) {
  return new M_Symbol{g->graph.getArg(pos)};
}

MODULAR_API_EXPORT M_Tuple *MAXG_graphNewOp(M_Graph *g, MlirLocation l,
                                            llvm::StringRef name,
                                            const M_Tuple *inputs,
                                            const M_Arity *outTypes,
                                            const M_AttrMap *attrs) {
  return new M_Tuple{g->graph.newOp(unwrap(l), name, inputs->tuple,
                                    outTypes->arity, attrs->map)};
}

//===----------------------------------------------------------------------===//
// Module support
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT bool MAXG_moduleToBytecode(const MlirModule m,
                                              StringRef fileName) {
  std::error_code ec;
  raw_fd_stream fs(fileName, ec);
  if (ec)
    return false;
  return writeBytecodeToFile(unwrap(m), fs).succeeded();
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

MODULAR_API_EXPORT MlirType MAXG_dTypeNew(MlirModule m, uint8_t value) {
  return wrap(Module(unwrap(m)).newDType(M::DType(value)));
}

MODULAR_API_EXPORT int64_t MAXG_dimTypeNewDynamic() {
  return ShapedType::kDynamic;
}

MODULAR_API_EXPORT MlirType MAXG_tensorTypeNew(MlirModule m, MlirType dtype,
                                               bool ranked, int64_t *dims,
                                               int32_t rank) {
  return wrap(Module(unwrap(m)).newTensorType(
      unwrap(dtype), ranked, llvm::ArrayRef<int64_t>(dims, rank)));
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

MODULAR_API_EXPORT M_Arity *MAXG_arityNew() { return new M_Arity{Arity()}; }

MODULAR_API_EXPORT uint32_t MAXG_aritySize(M_Arity *a) {
  return a->arity.size();
}

MODULAR_API_EXPORT void MAXG_arityAddType(M_Arity *a, MlirType t) {
  a->arity.add(unwrap(t));
}
