//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_LIST_H
#define GENERICML_SUPPORT_LIST_H

#include "LLCL/Runtime/AnyAsyncValueRef.h"

namespace M::GML {

/// A vector of arbitrary AnyAsyncValueRef elements.
struct List {
  SmallVector<AnyAsyncValueRef> elements;

  List copy() {
    List listCopy;
    for (const AnyAsyncValueRef &val : elements)
      listCopy.elements.push_back(val.copy());
    return listCopy;
  }
};

} // namespace M::GML

#endif // GENERICML_SUPPORT_LIST_H
