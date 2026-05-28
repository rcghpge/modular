//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// This view was vibe-coded by Claude and is not yet well understood.
//
// Interactive use-def graph view. Layout algorithm is inspired by iongraph:
//   https://spidermonkey.dev/blog/2025/10/28/iongraph-web.html
//
// Specifically uses a Sugiyama-style layered approach:
//   1. BFS layering (entry = layer 0, each layer flows downward)
//   2. Backedge detection via DFS (drawn as right-side loops)
//   3. Within-layer centering with a single parent-alignment pass
//   4. Railroad-style elbow edges (vertical + horizontal segments)
//   5. Infinite canvas with pan (drag) and zoom (Ctrl+scroll)

#ifndef GLIDIR_VIEWS_GRAPHVIEW_H
#define GLIDIR_VIEWS_GRAPHVIEW_H

#include "Common.h"
#include "TextView.h"

#include "imgui.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <vector>

namespace M {
namespace GlidIR {

struct GraphView {
  // A single node in the graph, corresponding to one bookmarked op.
  struct Node {
    mlir::Operation *op = nullptr;
    // Layer assigned by BFS; layer 0 is the topmost row.
    int layer = 0;
    // Position and size in world (pre-zoom) coordinates.
    float x = 0, y = 0, w = 0, h = 0;
    // "dialect.opname" string shown in the coloured header stripe.
    std::string name;
    // Full printed op line, truncated to 72 chars with an ellipsis.
    std::string text;
    // text further truncated to fit inside the node width; recomputed each
    // frame in computeLayout() after the node width is determined.
    std::string displayText;
  };

  // A directed use-def edge from src node to dst node.
  struct Edge {
    int src = 0, dst = 0;
    // Slot index and total count of edges leaving src / entering dst, used
    // to spread attachment points evenly across the node's bottom/top edge.
    int srcSlot = 0, srcCount = 1;
    int dstSlot = 0, dstCount = 1;
    // Printed SSA value + type, shown in a tooltip when hovering the edge.
    std::string valueText;
  };

  // Layout constants (world units).
  static constexpr float kPadX = 10, kPadY = 4;
  static constexpr float kLayerGap = 60, kNodeGap = 24;
  static constexpr float kMaxNodeW = 320.0f;

  // All nodes in the current graph, one per bookmarked op.
  llvm::SmallVector<Node, 0> nodes;
  // All use-def edges between bookmarked ops.
  llvm::SmallVector<Edge, 0> edges;
  // Fast op-pointer → node-index lookup, rebuilt whenever bookmarks change.
  llvm::DenseMap<mlir::Operation *, int> opIndex;
  // The bookmark set used to build the current graph; compared each frame to
  // detect changes that require a full graph rebuild.
  llvm::DenseSet<mlir::Operation *> lastBookmarkSet;
  // Op under the mouse cursor in the previous frame; used to avoid firing
  // navigateToOp on every frame when the cursor stays on the same node.
  mlir::Operation *lastHoveredOp = nullptr;

  // Callbacks used to drive the IR text panel when the user hovers or clicks.
  TextViewCallbacks callbacks;
  // Current pan offset in screen pixels (world origin → screen origin).
  ImVec2 pan = {300, 80};
  // Current zoom factor; 1.0 = 100 %.
  float zoom = 1.0f;
  // Index of the bookmark color group currently shown in the graph canvas.
  int activeGroupIndex = 0;
  // Set to true when the user presses the save button; consumed by the
  // renderer after the frame has been drawn to the back buffer.
  bool pendingSave = false;
  // Font size and font pointer captured during the most recent render() call,
  // while the ImGui frame is still active.  buildExportContext() runs after
  // ImGui::Render() when GetFontSize()/GetFont() are no longer reliable.
  float lastLH = 13.0f;
  ImFont *lastFont = nullptr;
  // Screen-space rectangle of the graph canvas from the most recent render,
  // in ImGui logical coordinates.  Used to crop the framebuffer capture.
  ImVec2 lastCanvasOrigin{}, lastCanvasEnd{};

  // Encapsulates the draw list and transform needed to render graph content
  // into any coordinate space — live canvas or offscreen export buffer.
  struct DrawContext {
    ImDrawList *dl;
    // Top-left of the drawing area in draw-list coordinate space.
    ImVec2 origin;
    // Bottom-right corner; used for clip rect setup and visibility culling.
    ImVec2 canvasEnd;
    // Translation from world (0,0) to draw-list space: origin + pan offset.
    float panX, panY;
    float zoom;
    // Line height (ImGui::GetFontSize()) captured at layout time.
    float lh;
    // Font captured during render() while the ImGui frame is active.
    // ImGui::GetFont() returns null after Render(), so we store it here.
    ImFont *font = nullptr;

    // Map a world-space point to draw-list coordinates.
    ImVec2 w(float wx, float wy) const {
      return {origin.x + panX + wx * zoom, origin.y + panY + wy * zoom};
    }
  };

  GraphView() = default;
  GraphView(mlir::ModuleOp, TextViewCallbacks &cb);

  // Build nodes and use-def edges from the given op list.
  void buildGraph(llvm::ArrayRef<mlir::Operation *> ops);
  // Run the layered layout algorithm: assign layers, compute node sizes, and
  // place nodes so that parent ops appear above their consumers.
  void computeLayout();
  // Draw edges and nodes into dl using the given transform.
  // hoveredOp, selectedOp, and hoveredEdge may all be null (treated as none).
  void drawContent(const DrawContext &ctx, mlir::Operation *hoveredOp,
                   mlir::Operation *selectedOp, const Edge *hoveredEdge) const;
  // Build a DrawContext that fits every laid-out node into an offscreen image
  // with a small margin.  Sets outW/outH to the required pixel dimensions.
  // scale=2 doubles both dimensions (4× pixels) for a higher-resolution export.
  DrawContext buildExportContext(ImDrawList *dl, int &outW, int &outH,
                                 float scale = 1.5f) const;
  // Draw the tab bar and graph canvas.  Rebuilds the graph when the active
  // group's bookmark set changes.  groups[i] holds the ops for color group i.
  // selectedOp, if non-null, is drawn with a highlighted border.
  void render(llvm::ArrayRef<llvm::SmallVector<mlir::Operation *>> groups,
              mlir::Operation *selectedOp);
};

} // namespace GlidIR
} // namespace M

#endif // GLIDIR_VIEWS_GRAPHVIEW_H
