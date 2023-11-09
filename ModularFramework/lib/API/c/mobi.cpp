//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MOF/API/c/mobi.h"

#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "MOF/API/c/detail/mobi.h"
#include "MOF/InitAllDialects.h"
#include "MOF/MOBI/Attr.h"
#include "MOF/MOBI/Graph.h"
#include "MOF/MOBI/Loc.h"
#include "MOF/MOBI/Module.h"
#include "MOF/MOBI/Symbol.h"
#include "MOF/MOBI/Type.h"
#include "Support/Buffer.h"
#include "Support/ML/TensorShape.h"
#include "mlir/IR/MLIRContext.h"

using namespace M::MOF::Builder;
using namespace llvm;
using namespace mlir;

/// Attr

MODULAR_API_EXPORT M_Attr *MOBI_tensorAttr(M_Module *m, StringRef name,
                                           void *data, M_Type *t) {
  auto moType = cast<M::MO::TensorType>(t->type.getVal());
  auto dims = moType.getIntDims();
  int64_t size = moType.getDtype().getSizeInBytes();
  for (auto d : dims) {
    assert(d >= 0 && "the caller should enforce this");
    size *= d;
  }
  return new M_Attr{
      m->module.tensorAttr(name,
                           ArrayRef<uint8_t>(reinterpret_cast<uint8_t *>(data),
                                             static_cast<size_t>(size)),
                           t->type)};
}

MODULAR_API_EXPORT M_Attr *MOBI_tensorAttrFromFile(M_Module *m, StringRef name,
                                                   StringRef fileName,
                                                   M_Type *t) {
  auto moType = cast<M::MO::TensorType>(t->type.getVal());
  auto dims = moType.getIntDims();
  int64_t size = moType.getDtype().getSizeInBytes();
  for (auto d : dims) {
    assert(d >= 0 && "the caller should enforce this");
    size *= d;
  }

  auto buff = M::Buffer::getFile(std::filesystem::path{fileName.str()});
  if (buff.isError()) {
    errs() << buff.takeError() << "\n";
  }
  assert(!buff.isError());
  auto buff_size = static_cast<int64_t>((*buff)->getBufferSize());
  if (buff_size != size) {
    errs() << "Size mismatch, want " << size << " got " << buff_size << "\n";
  }
  assert(buff_size == size);
  auto data = (*buff)->getBufferStart();

  return new M_Attr{m->module.tensorAttr(
      name,
      ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(data),
                        static_cast<size_t>(size)),
      t->type)};
}

MODULAR_API_EXPORT M_AttrMap *MOBI_newAttrMap() { return new M_AttrMap(); }

MODULAR_API_EXPORT void MOBI_addAttr(M_AttrMap *m, M_Attr *a) {
  m->map.add(a->attr);
}

/// Graph

MODULAR_API_EXPORT M_Graph *MOBI_newGraph(M_Module *m, M_Loc *l,
                                          llvm::StringRef name,
                                          const M_Arity *inTypes,
                                          const M_Arity *outTypes) {
  return new M_Graph{
      m->module.newGraph(l->loc, name, inTypes->arity, outTypes->arity)};
}

MODULAR_API_EXPORT M_Symbol *MOBI_getInput(M_Graph *g, uint32_t pos) {
  return new M_Symbol{g->graph.getArg(pos)};
}

MODULAR_API_EXPORT M_Tuple *
MOBI_newOp(M_Graph *g, M_Loc *l, llvm::StringRef name, const M_Tuple *inputs,
           const M_Arity *outTypes, const M_AttrMap *attrs) {
  return new M_Tuple{
      g->graph.newOp(l->loc, name, inputs->tuple, outTypes->arity, attrs->map)};
}

/// Location

MODULAR_API_EXPORT M_Loc *MOBI_unknownLoc(M_Module *m) {
  return new M_Loc{m->module.unknownLoc()};
}

/// Module

static MLIRContext *newContext() {
  // TODO: Attach the context to modules instead? Let user control it?
  static MLIRContext ctx;
  mlir::DialectRegistry registry;
  M::MOF::registerAllDialects(registry);
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();
  return &ctx;
}

MODULAR_API_EXPORT M_Module *MOBI_newModule() {
  static MLIRContext *ctx = newContext();
  Location testLoc = UnknownLoc::get(ctx);

  auto *m = new M_Module{Module(testLoc)};
  return m;
}

MODULAR_API_EXPORT char *MOBI_moduleToString(const M_Module *const m) {
  SmallVector<char> buffer;
  raw_svector_ostream os(buffer);
  m->module.getModuleOp()->print(os);

  // TODO: Do it without so many copies.
  void *copy = std::malloc(buffer.size());
  memcpy(copy, buffer.data(), buffer.size());
  return reinterpret_cast<char *>(copy);
}

MODULAR_API_EXPORT void MOBI_debugPrintModule(const M_Module *const m) {
  m->module.getModuleOp()->print(outs());
}

/// Symbol

MODULAR_API_EXPORT M_Tuple *MOBI_newTuple() { return new M_Tuple{Tuple()}; }

MODULAR_API_EXPORT void MOBI_addSymbol(M_Tuple *t, M_Symbol *s) {
  t->tuple.add(s->symbol);
}

MODULAR_API_EXPORT M_Symbol *MOBI_getSymbol(M_Tuple *t, uint32_t pos) {
  return new M_Symbol{t->tuple.getItem(pos)};
}

/// Type

MODULAR_API_EXPORT M_Type *MOBI_typeFromSpec(M_Module *m, uint8_t dtype,
                                             const void *shape) {
  const M::TensorShape &tensorShape = *(const M::TensorShape *)shape;
  SmallVector<int64_t> dims(tensorShape.begin(), tensorShape.end());
  auto outTy = M::MO::TensorType::get(
      dims, *M::MO::toMLIRType(m->module.getContext(), M::DType(dtype)),
      /*metadata=*/{});
  return new M_Type{outTy};
}

MODULAR_API_EXPORT void MOBI_shapeFromType(M_Type *t, void *shape) {
  M::TensorShape &tensorShape = *(M::TensorShape *)shape;
  auto tensorTy = cast<M::MO::TensorType>(t->type.getVal());
  assert(tensorTy.hasRank() && "no unranked TensorShape support yet");

  tensorShape = M::TensorShape(tensorTy.getIntDims());
}

MODULAR_API_EXPORT uint8_t MOBI_dtypeFromType(M_Type *t) {
  // Directly return type `t`'s element type as `uint8_t` since Mojo's DType is
  // ABI compatible.
  auto tensorTy = cast<M::MO::TensorType>(t->type.getVal());
  return tensorTy.getDtype().getValue();
}

MODULAR_API_EXPORT int64_t MOBI_dynamicDim(M_Module *m) {
  return m->module.dynamicDim();
}

MODULAR_API_EXPORT M_Type *MOBI_f32(M_Module *m, int64_t *dims, int64_t rank) {
  return new M_Type{m->module.f32(ArrayRef<int64_t>(dims, rank))};
}

MODULAR_API_EXPORT M_Type *MOBI_si(M_Module *m, uint64_t width, int64_t *dims,
                                   int64_t rank) {
  return new M_Type{m->module.si(width, ArrayRef<int64_t>(dims, rank))};
}

MODULAR_API_EXPORT M_Type *MOBI_ui(M_Module *m, uint64_t width, int64_t *dims,
                                   int64_t rank) {
  return new M_Type{m->module.ui(width, ArrayRef<int64_t>(dims, rank))};
}

MODULAR_API_EXPORT M_Type *MOBI_bool(M_Module *m, int64_t *dims, int64_t rank) {
  return new M_Type{m->module.bool_(ArrayRef<int64_t>(dims, rank))};
}

MODULAR_API_EXPORT M_Arity *MOBI_newArity() { return new M_Arity{Arity()}; }

MODULAR_API_EXPORT void MOBI_addType(M_Arity *a, M_Type *t) {
  a->arity.add(t->type);
}

MODULAR_API_EXPORT uint64_t MOBI_typeIsTensor(M_Type *t) {
  return isa<M::MO::TensorType>(t->type.getVal());
}

MODULAR_API_EXPORT bool MOBI_typeIsFloatTensor(M_Type *t) {
  return isa<M::MO::TensorType>(t->type.getVal()) &&
         cast<M::MO::TensorType>(t->type.getVal()).getDtype().isFloat();
}

MODULAR_API_EXPORT bool MOBI_typeIsSITensor(M_Type *t) {
  return isa<M::MO::TensorType>(t->type.getVal()) &&
         cast<M::MO::TensorType>(t->type.getVal()).getDtype().isSInt();
}

MODULAR_API_EXPORT bool MOBI_typeIsUITensor(M_Type *t) {
  return isa<M::MO::TensorType>(t->type.getVal()) &&
         cast<M::MO::TensorType>(t->type.getVal()).getDtype().isUInt();
}

MODULAR_API_EXPORT bool MOBI_typeIsBoolTensor(M_Type *t) {
  return isa<M::MO::TensorType>(t->type.getVal()) &&
         cast<M::MO::TensorType>(t->type.getVal()).getDtype().isBool();
}

MODULAR_API_EXPORT uint64_t MOBI_getTensorTypeDTypeWidth(M_Type *t) {
  return cast<M::MO::TensorType>(t->type.getVal()).getDtype().getWidthInBits();
}

MODULAR_API_EXPORT uint64_t MOBI_getTensorTypeRank(M_Type *t) {
  return cast<M::MO::TensorType>(t->type.getVal()).getRank();
}

// Doesn't currently support parametric dimensions.
// Dynamic dims return ShapedDim::kDynamic.
MODULAR_API_EXPORT int64_t MOBI_getTensorTypeDim(M_Type *t, uint64_t dim) {
  return cast<M::MO::TensorType>(t->type.getVal()).getIntDim(dim);
}

/// Retrieves the type of `symbol`.
/// The caller is responsible for freeing the returning `M_Type *` using
/// `MOBI_freeType()`.
///
/// Returns a pointer to a new `M_Type` containing `symbol`'s type on success.
MODULAR_API_EXPORT M_Type *MOBI_typeOfSymbol(M_Symbol *symbol) {
  return new M_Type{MOBI::Type(symbol->symbol.getVal().getType())};
}

/// Deallocates memory associated with type `t`.
MODULAR_API_EXPORT void MOBI_freeType(M_Type *t) { delete t; }
