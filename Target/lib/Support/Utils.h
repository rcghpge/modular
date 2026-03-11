//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef TARGET_LIB_UTILS_UTILS_H
#define TARGET_LIB_UTILS_UTILS_H

#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

template <typename Stringish>
void copyStatusMessage(char *dstMsgBuf, int64_t dstMsgBufSize,
                       Stringish const &srcMsg) {
  assert(dstMsgBufSize > 0);
  auto srcBeginIter = std::begin(srcMsg);
  auto srcEndIter = (int64_t(srcMsg.size()) >= (dstMsgBufSize - 1))
                        ? (srcBeginIter + (dstMsgBufSize - 1))
                        : std::end(srcMsg);
  auto dstEndIter = std::copy(srcBeginIter, srcEndIter, dstMsgBuf);
  *dstEndIter = '\0';
}

void copyStatusMessage(char *dstMsgBuf, int64_t dstMsgBufSize,
                       const char *srcMsg);
void copyStatusMessage(char *dstMsgBuf, int64_t dstMsgBufSize,
                       llvm::Twine const &srcMsg);

#endif // TARGET_LIB_UTILS_UTILS_H
