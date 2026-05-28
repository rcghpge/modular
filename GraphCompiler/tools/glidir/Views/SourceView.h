//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// This view was vibe-coded by Claude.
//
// Displays source files referenced by FileLineColLoc locations in the IR.
// Uses a TextViewer so the source panel looks identical to the IR panel.

#ifndef GLIDIR_VIEWS_SOURCEVIEW_H
#define GLIDIR_VIEWS_SOURCEVIEW_H

#include "TextViewer/TextViewer.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace M {
namespace GlidIR {

// Recursively collects all FileLineColLoc leaves from any Location, including
// FusedLoc nodes produced when the GC fuses ops.
void collectFileLineColLocs(
    mlir::Location loc, llvm::SmallVectorImpl<mlir::FileLineColLoc> &result);

struct SourceView {
  // One entry per unique source file referenced by any op's FileLineColLoc.
  struct SourceFile {
    // Absolute path to the source file, used as the tab label.
    std::string path;
    // Raw file contents loaded into the TextViewer via SetText.
    std::string content;
    // Maps each 0-indexed source line to all ops whose location points there.
    llvm::DenseMap<int, llvm::SmallVector<mlir::Operation *>> lineToOp;
  };

  // All source files referenced by the module, one per unique path.
  std::vector<SourceFile> files;
  // Index into files for the tab currently selected in the UI.
  int currentFileIndex = 0;
  // Index of the file whose text is currently loaded into textViewer.
  // Empty until the first render; compared against currentFileIndex each frame
  // to detect tab switches that require reloading the viewer content.
  std::optional<int> loadedFileIndex;
  // Text viewer widget for the currently displayed source file.
  TextViewer textViewer;
  // The op highlighted in the previous render call; used to detect when the
  // highlighted op changes so the viewer can scroll to its new location.
  mlir::Operation *prevHighlightedOp = nullptr;

  // Search bar state — kept as member fields because ImGui input widgets
  // read from persistent char buffers.
  char searchBuffer[512] = {};
  char prevSearchBuffer[512] = {};
  // When true, the search input widget is re-focused on the next frame so
  // repeated Enter presses continue navigating matches.
  bool searchRefocus = false;
  // Source lines (0-indexed) that match the current search query.
  llvm::SmallVector<int> searchMatchedLines;
  // Index into searchMatchedLines for the currently focused match, if any.
  std::optional<int> searchCurrentIndex;
  // Pre-split lines of the currently displayed file, kept for regex matching.
  std::vector<std::string> cachedTextLines;
  // When non-negative, the viewer scrolls to this line after loading a new
  // file (so the highlighted op stays visible on tab switch).
  int pendingSearchLine = -1;

  SourceView() = default;
  // Walk all ops in root, collect FileLineColLoc locations, and load the
  // referenced source files from disk.
  explicit SourceView(mlir::ModuleOp root);

  // Apply the same syntax-highlighting palette used by the IR text panel.
  // Must be called after the palette is finalised (end of Renderer ctor).
  void initViewer(const TextViewer::Palette &palette);

  // Draw the regex search bar and prev/next navigation buttons.
  void renderSourceFilter();

  // Render the file-tab bar, the source text, and any line highlights.
  // onSourceLineClicked is fired with all ops mapped to the clicked line.
  void render(mlir::Operation *highlightedOp,
              std::function<void(llvm::ArrayRef<mlir::Operation *>)>
                  onSourceLineClicked);
};

} // namespace GlidIR
} // namespace M

#endif // GLIDIR_VIEWS_SOURCEVIEW_H
