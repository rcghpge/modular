//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "Utils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cstring>

void copyStatusMessage(char *dstMessageBuffer, int64_t dstMessageBufferSize,
                       const char *srcErrorString) {
  int64_t messageLength = (int64_t)strlen(srcErrorString);
  if (messageLength >= dstMessageBufferSize)
    messageLength = dstMessageBufferSize - 1;
  [[maybe_unused]] auto _ =
      strncpy(dstMessageBuffer, srcErrorString, messageLength);
  dstMessageBuffer[messageLength] = '\0';
}

void copyStatusMessage(char *dstMsgBuf, int64_t dstMsgBufSize,
                       llvm::Twine const &srcMsg) {
  llvm::SmallVector<char, 128> tmp;
  llvm::StringRef str = srcMsg.toStringRef(tmp);
  int64_t strSize = int64_t(str.size());
  int64_t msgSize =
      strSize < (dstMsgBufSize - 1) ? strSize : (dstMsgBufSize - 1);
  [[maybe_unused]] auto _ = strncpy(dstMsgBuf, str.data(), msgSize);
  dstMsgBuf[msgSize] = '\0';
}
