//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GLIDIR_VIEWS_TEXTVIEW_H
#define GLIDIR_VIEWS_TEXTVIEW_H

#include "Common.h"
#include "TextViewer/TextViewer.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <array>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace M {
namespace GlidIR {

// Callbacks injected into views that need to drive the IR text panel. The text
// panel owns the canonical op-to-line mapping, so other views ask it to
// navigate rather than performing the lookup themselves.
struct TextViewCallbacks {
  // Scroll and highlight a specific 0-indexed line in the IR text panel.
  std::function<void(int)> highlightOpInTextViewer;
  // Navigate the IR text panel to the line for the given op and also expand
  // the op in the tree view.
  std::function<void(mlir::Operation *)> navigateToOp;
  // Scroll the IR text panel to the line for the given op without affecting
  // the tree view.
  std::function<void(mlir::Operation *)> scrollToOpInTextViewer;
};

// Displays the IR dump as a read-only, syntax-highlighted text viewer.
// Maintains a bidirectional mapping between source lines and MLIR operations
// so that navigating either panel keeps the other in sync.
class TextView {
public:
  // Callbacks fired when the text cursor moves or the user toggles a bookmark,
  // so the tree view can expand and highlight the corresponding op.
  struct TreeCallbacks {
    // Called with the op whose line the cursor is on; nullptr if no op there.
    std::function<void(mlir::Operation *)> highlightOp;
    // Called when the cursor moves off a line that had an op.
    std::function<void()> clearHighlight;
  };

  TextView() = default;

  // Register callbacks that the text view fires to keep the tree view in sync.
  void setTreeCallbacks(TreeCallbacks cbs) { treeCallbacks = cbs; }

  // Load the IR dump text, build op<->line mappings, and configure the editor.
  // dumpRoot is a re-parsed copy of root whose ops carry FileLineColLoc
  // locations pointing into dumpText; the two modules are walked in parallel
  // to build lineNumberToOp / opLineNumber.  Must be called once before any
  // render method.
  void init(mlir::ModuleOp root, mlir::ModuleOp dumpRoot,
            const std::string &dumpText, const TextViewer::Palette &palette);

  // The underlying text viewer widget, exposed so the renderer can query
  // cursor state and pass it to other views.
  TextViewer &getViewer() { return textViewer; }

  // Map from op pointer to its 0-indexed line number in the dump text.
  const llvm::DenseMap<mlir::Operation *, int> &getOpLineNumber() const {
    return opLineNumber;
  }

  // Map from 0-indexed line number to the op printed on that line.
  // Lines that carry no op (e.g. region / block header lines) map to nullptr.
  const llvm::SmallVector<mlir::Operation *> &getLineNumberToOp() const {
    return lineNumberToOp;
  }

  // Index of the active bookmark color group used for new double-click
  // bookmarks.
  int getActiveBookmarkColor() const { return activeBookmarkColor; }
  // Set the active bookmark color group index.
  void setActiveBookmarkColor(int idx) { activeBookmarkColor = idx; }
  // True if at least one bookmark group is non-empty.
  bool hasAnyBookmarks() const;
  // Return 1-indexed line numbers for group idx, in ascending order.
  llvm::SmallVector<int> sortedBookmarksForGroup(int idx) const;
  // Return all ops bookmarked in group idx, in line order.
  llvm::SmallVector<mlir::Operation *> opsForGroup(int idx) const;
  // Add all ops on lines [minLine, maxLine] (0-indexed) to color group
  // colorIdx.
  void bookmarkSelection(int minLine, int maxLine, int colorIdx);

  // Move the editor cursor to lineNo (0-indexed) and notify the tree view.
  void highlightLine(int lineNo);

  // Scroll the editor to the line for the given op, if it is in the mapping.
  void scrollToOp(mlir::Operation *op);

  // Draw the regex search bar and prev/next navigation buttons above the
  // editor.
  void renderTextFilter();

  // Check whether the cursor moved this frame and fire tree callbacks.
  // Also handles double-click to toggle a bookmark on the current line.
  void processTextViewerCursorEvents();

  // Draw the list of bookmarked ops with jump-to and per-item delete controls.
  void renderBookmarks();

private:
  // The ImGui text viewer widget that renders the IR dump.
  TextViewer textViewer;
  // All lines of the dump text, kept for regex search matching.
  std::vector<std::string> textAsLines;
  // Parallel to textAsLines: the op whose printed form starts on each line.
  llvm::SmallVector<mlir::Operation *> lineNumberToOp;
  // Reverse of lineNumberToOp: op pointer -> 0-indexed line number.
  llvm::DenseMap<mlir::Operation *, int> opLineNumber;
  // Lines matching the current search regex, in document order.
  llvm::SmallVector<int> matchedLines;
  // Index into matchedLines for the currently focused match, if any.
  std::optional<int> currentMatchedIndex;
  // Callbacks to the tree view, fired on cursor movement and bookmark changes.
  TreeCallbacks treeCallbacks;
  // Per-color bookmark sets; each set holds 1-indexed line numbers.
  std::array<std::set<int>, kNumBookmarkColors> coloredBookmarks;
  // Index of the color group that new double-click bookmarks are assigned to.
  int activeBookmarkColor = 0;

  // Rebuild textViewer.GetBreakpoints() as the union of all coloredBookmarks
  // groups.
  void syncBreakpoints();
};

} // namespace GlidIR
} // namespace M

#endif // GLIDIR_VIEWS_TEXTVIEW_H
