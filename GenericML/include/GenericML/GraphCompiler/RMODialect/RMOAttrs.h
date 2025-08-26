//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares attributes for the RMO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOATTRS_H
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"

namespace M::RMO {

struct PythonFrame {
  void *ptr;
  char *(*to_str)(void *);
  void (*incref)(void *);
  void (*decref)(void *);
  char *cached_output = nullptr;
};

class PythonFrameAttrStorage : public mlir::AttributeStorage {
public:
  using KeyTy = PythonFrame;

  PythonFrameAttrStorage(PythonFrame frame) : frame(frame) {
    // Frame object must live as long as the new storage object.
    frame.incref(frame.ptr);
  }

  ~PythonFrameAttrStorage() {
    frame.decref(frame.ptr);
    if (frame.cached_output)
      delete frame.cached_output;
  }

  char *stack_trace() {
    if (frame.cached_output)
      return frame.cached_output;

    frame.cached_output = frame.to_str(frame.ptr);
    return frame.cached_output;
  }

  bool operator==(const KeyTy &key) const {
    return key.ptr == frame.ptr && key.to_str == frame.to_str &&
           key.incref == frame.incref && key.decref == frame.decref;
  }

  static KeyTy getKey(PythonFrame frame) { return frame; }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.ptr, key.to_str);
  }

  static PythonFrameAttrStorage *
  construct(mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<PythonFrameAttrStorage>())
        PythonFrameAttrStorage(key);
  }

  PythonFrame frame;
};

/// Location objects represent source locations information in MLIR.
/// LocationAttr acts as the anchor for all Location based attributes.
class PythonFrameAttr
    : public mlir::Attribute::AttrBase<PythonFrameAttr, mlir::Attribute,
                                       PythonFrameAttrStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "rmo.opaque_py_frame";

  using ImplType = PythonFrameAttrStorage;

  static PythonFrameAttr get(mlir::MLIRContext *context, PythonFrame frame) {
    return Base::get(context, frame);
  }

  static mlir::OpaqueLoc getOpaque(mlir::MLIRContext *context,
                                   PythonFrame frame) {
    PythonFrameAttr frameAttr = get(context, frame);
    return mlir::OpaqueLoc::get(frameAttr.getImpl(), context);
  }

  static PythonFrameAttr fromOpaque(mlir::OpaqueLoc opaque) {
    auto ptr = mlir::OpaqueLoc::getUnderlyingLocationOrNull<
        RMO::PythonFrameAttr::ImplType *>(opaque);
    return PythonFrameAttr(ptr);
  }

  static PythonFrameAttr fromLocation(mlir::Location loc) {
    if (auto opaque = dyn_cast<mlir::OpaqueLoc>(loc))
      return fromOpaque(opaque);
    return {};
  }

  void *frame() { return getImpl()->frame.ptr; }

  char *stack_trace() { return getImpl()->stack_trace(); }
};

} // namespace M::RMO

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GenericML/GraphCompiler/RMODialect/RMOAttrs.h.inc"

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOATTRS_H
