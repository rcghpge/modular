//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GLIDIR_VIEWS_TREEVIEW_H
#define GLIDIR_VIEWS_TREEVIEW_H

#include "TextView.h"

#include "imgui.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace M {
namespace GlidIR {

struct TreeView {
  // Callbacks into the text and graph panels, injected at construction.
  TextViewCallbacks callbacks;
  // Root of the IR being visualized; tree traversal starts here.
  mlir::ModuleOp root;
  // Op to visually highlight in the tree (set by the text panel or graph
  // panel when the user moves the cursor or hovers over a node).
  mlir::Operation *opToHighlight = nullptr;
  // When true, the tree will auto-scroll so the highlighted op is visible.
  // Reset to false once the scroll has been applied, so that the user can
  // scroll freely without the view snapping back on every frame.
  bool adjustYPosToHighlightedOp = true;
  // Ops / regions / blocks that must be expanded so the highlighted op is
  // visible.  Populated whenever opToHighlight changes, then consumed in a
  // single render frame (controlled by pendingExpansion).
  llvm::DenseSet<void *> mlirEntitiesToExpandForHighlightedOp;
  // True for exactly one frame after the highlight target changes.  During
  // that frame applyExpandState forces the path-to-root open; after that,
  // user-driven collapse is respected.
  bool pendingExpansion = false;
  // Op requested to be brought into focus (e.g. via a double-click in the
  // graph panel).  Distinct from opToHighlight so focus and highlight can be
  // driven independently.
  mlir::Operation *opToFocus = nullptr;
  // Screen-space centre of every entity rendered this frame.  Used by
  // renderUsers() to draw use-def elbow arrows between ops.
  llvm::DenseMap<void *, ImVec2> renderedMLIREntities;
  // Remembers which nodes the user has expanded or collapsed across frames so
  // that re-renders don't reset manually adjusted tree state.
  llvm::DenseMap<void *, bool> mlirEntitiesPreviouslyExpanded;
  // The op / region / block from whose context menu "collapse children" was
  // triggered.  Children of this entity are forced closed until the user
  // expands them again.
  void *mlirEntityWhereWeCollapsed = nullptr;
  // Each "Open in new window" action creates one of these.
  struct DetachedWindow {
    std::string title;
    std::function<void(bool &)> renderFn;
    llvm::DenseMap<void *, bool> expandedState;
    DetachedWindow(std::string title, std::function<void(bool &)> renderFn)
        : title(std::move(title)), renderFn(std::move(renderFn)) {}
  };
  std::vector<DetachedWindow> detachedWindows;
  // Monotonically increasing counter used to give each detached window a
  // unique ImGui ID (via the ###id suffix).
  int nextDetachedId = 0;

  // RAII guard returned by CollapsibleTreeNode / ColouredCollapsibleNode.
  // Calls ImGui::TreePop() on destruction when the node was open.
  struct CollapsibleTreeNodeGuard {
    CollapsibleTreeNodeGuard(bool recursiveCollapseBeforeThis, bool wasExpanded)
        : recursiveCollapseBeforeThis(recursiveCollapseBeforeThis),
          wasExpanded(wasExpanded) {}
    ~CollapsibleTreeNodeGuard() {
      if (wasExpanded || recursiveCollapseBeforeThis)
        ImGui::TreePop();
    }

    operator bool() const { return wasExpanded; }

    bool recursiveCollapseBeforeThis;
    bool wasExpanded;
  };

  TreeView() = default;
  TreeView(mlir::ModuleOp root, TextViewCallbacks &callbacks);

  // Clear opToHighlight and reset the auto-scroll flag.
  void clearHighlightedOp();
  // Set opToHighlight and compute the set of ancestors that must be expanded
  // to make it visible, scheduling a one-frame expansion pass.
  void highlightOpOnTreeView(mlir::Operation *op);
  // Reset per-frame data (rendered entity positions) at the start of each
  // render.
  void clearPerFrameData();
  // Top-level render entry point: walks the module and draws the tree.
  void render();

  // Draw a right-angle (elbow) arrow from src to dst on the given draw list.
  static void drawElbowArrow(ImDrawList *dl, ImVec2 src, ImVec2 dst,
                             ImU32 color);
  // Overlay use-def arrows on all ops that were rendered this frame.
  void renderUsers();
  // Restore or persist the expand/collapse state for entity across frames.
  void applyExpandState(void *entity);

  // Render a single block and its child ops.
  void render(mlir::Block &block, uint64_t id, bool recursiveCollapse);
  // Render a region and its blocks.
  void render(mlir::Region &region, bool recursiveCollapse);
  // Render a type as a leaf node (or expand its sub-elements when open).
  void render(mlir::Type type, bool recursiveCollapse);

  // Show the printed form of mlirEntity as an ImGui tooltip on hover.
  template <typename T>
  static void setToolTipOnMLIREntity(T mlirEntity) {
    if (ImGui::BeginItemTooltip()) {
      std::string str;
      llvm::raw_string_ostream stream(str);
      mlirEntity.print(stream);
      ImGui::Text("%s", str.c_str());
      ImGui::EndTooltip();
    }
  }

  // Render an attribute as a leaf or expandable node depending on its kind.
  void render(mlir::Attribute attr, bool recursiveCollapse);

  // Render a collapsible tree node drawn in a custom colour.  Pops the style
  // colour before opening the context-menu popup so the menu text is not
  // tinted.  When openFn is set, the context menu gains an "Open in new window"
  // item that invokes openFn to populate detachedRenderFn/detachedWindowTitle.
  CollapsibleTreeNodeGuard
  ColouredCollapsibleNode(ImU32 colour, bool &recursiveCollapse,
                          const char *name, ImGuiTreeNodeFlags flags = 0,
                          void *mlirEntity = nullptr,
                          std::function<void()> openFn = nullptr);

  // Render a DictionaryAttr as an expandable "attr-dict" node.
  void render(mlir::DictionaryAttr dict, bool recursiveCollapse);

  // Render an SSA value range (operands or results) as a coloured expandable
  // group, with each value showing its type on expansion.
  template <typename Range>
  void renderValueRange(Range values, bool recursiveCollapse,
                        const char *groupIcon, const char *groupLabel,
                        const char *itemIcon) {
    if (values.empty())
      return;

    mlir::OpPrintingFlags flags;
    flags.useLocalScope(false);

    if (recursiveCollapse)
      ImGui::SetNextItemOpen(false, ImGuiCond_Always);

    static constexpr ImU32 kColour = IM_COL32(0, 255, 0, 255);
    auto groupLabelStr = llvm::formatv("{}{}", groupIcon, groupLabel).str();
    auto treeNode = ColouredCollapsibleNode(kColour, recursiveCollapse,
                                            groupLabelStr.c_str());
    if (treeNode || recursiveCollapse) {
      for (auto &&value : values) {
        std::string valueStr;
        llvm::raw_string_ostream stream(valueStr);
        value.printAsOperand(stream, flags);

        if (recursiveCollapse)
          ImGui::SetNextItemOpen(false, ImGuiCond_Always);

        auto itemLabel = llvm::formatv("{}{}", itemIcon, valueStr).str();
        mlir::Type valType = value.getType();
        std::string openTitle =
            llvm::formatv("{}{}###detached", itemIcon, valueStr).str();
        auto openFn = [this, valType, t = std::move(openTitle)]() mutable {
          int wid = nextDetachedId++;
          detachedWindows.push_back(DetachedWindow{
              llvm::formatv("{}###detached_{}", t, wid).str(),
              [this, valType](bool &rc) { render(valType, rc); }});
        };
        auto node = ColouredCollapsibleNode(kColour, recursiveCollapse,
                                            itemLabel.c_str(), 0, nullptr,
                                            std::move(openFn));
        setToolTipOnMLIREntity(value.getType());
        if (node || recursiveCollapse)
          render(value.getType(), recursiveCollapse);
      }
    }
  }

  // Render operands as a coloured expandable group.
  void render(mlir::OperandRange operands, bool recursiveCollapse);
  // Render results as a coloured expandable group.
  void render(mlir::ResultRange results, bool recursiveCollapse);
  // Record the screen-space position of entity so use-def arrows can be drawn
  // to it later in the same frame.
  void registerEntityAsRendered(void *entity);
  // Render a single op node with its regions, operands, results, and attrs.
  void render(mlir::Operation &op, uint64_t id, bool recursiveCollapse);

  // Create a collapsible tree node that supports recursive collapse.
  // recursiveCollapse is passed by reference: when the user right-clicks and
  // selects "collapse children" it is set to true, and every subsequent
  // node on the same render path forces itself closed, propagating the
  // collapse down the entire subtree in one frame.
  // When openFn is set, the context menu gains an "Open in new window" item.
  CollapsibleTreeNodeGuard
  CollapsibleTreeNode(bool &recursiveCollapse, const char *name,
                      ImGuiTreeNodeFlags flags = 0, void *mlirEntity = nullptr,
                      std::function<void()> openFn = nullptr);

  // Render all open detached floating windows (no-op when none are open).
  void renderDetachedWindows();
};

} // namespace GlidIR
} // namespace M

#endif // GLIDIR_VIEWS_TREEVIEW_H
