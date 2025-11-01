//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// MLIR graph and tree visualization tool.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GenericML/GraphCompiler/RMODialect/RMODialect.h"
#include "GenericML/MGPDialect/MGPDialect.h"
#include "GraphRT/GRTDialect/GRTDialect.h"
#include "GraphRT/MEFDialect/MEFDialect.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"

#define GL_SILENCE_DEPRECATION
#include "TextEditor.h"
#include "hello_imgui/imgui_theme.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h> // Will drag system OpenGL headers
#undef GL_SILENCE_DEPRECATION

#include <cstdio>
#include <fstream>

using namespace mlir;
using namespace M;

[[maybe_unused]] static constexpr auto kIconTrash = "\xef\x87\xb8";
static constexpr ImVec4 kClearColor = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

//===----------------------------------------------------------------------===//
// GLFW + OpenGL3 specific implementation
//===----------------------------------------------------------------------===//

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

static void pollEventsAndCreateNewFrameForBackends() {
  glfwPollEvents();
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
}

static void renderUsingBackends(auto window) {
  int displayWidth, displayHeight;
  glfwGetFramebufferSize(window, &displayWidth, &displayHeight);
  glViewport(0, 0, displayWidth, displayHeight);
  glClearColor(kClearColor.x * kClearColor.w, kClearColor.y * kClearColor.w,
               kClearColor.z * kClearColor.w, kClearColor.w);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  glfwSwapBuffers(window);
}

static bool shouldCloseWindow(auto window) {
  return glfwWindowShouldClose(window);
}

static void deinitImGuiWithBackends(GLFWwindow *window) {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
}

// This was blatantly copy-pasted from the official examples
static GLFWwindow *initImGuiWithBackends() {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit())
    exit(1);

  const char *glslVersion = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Required on Mac

  GLFWwindow *window =
      glfwCreateWindow(1280, 720, "mlir-viz", nullptr, nullptr);
  if (window == nullptr)
    exit(1);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glslVersion);

  return window;
}

//===----------------------------------------------------------------------===//
// Rendering
//===----------------------------------------------------------------------===//

/// The following will return the line number associated with the op. Note that
/// the line number is 1-indexed. Therefore, we subtract 1 to make it possible
/// to index using the result of this function.
static std::optional<int> locNumber(Operation *op) {
  if (auto lineLoc = dyn_cast<FileLineColLoc>(op->getLoc()))
    return std::max(0, int(lineLoc.getLine()) - 1);
  return std::nullopt;
}

// Will display (?) with a tooltip when hovered.
static void HelpMarker(const char *desc) {
  ImGui::TextDisabled("(?)");
  if (ImGui::BeginItemTooltip()) {
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(desc);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}

struct MLIRTextCallbacks {
  std::function<void(int)> highlightOpInTextEditor;
};

struct MLIRTreeRenderer {
  MLIRTextCallbacks callbacks;
  // Module op that acts as the root of the IR.
  ModuleOp root;
  // Op that should be highlighted in the tree view.
  Operation *opToHighlight;
  // Whether or not we have adjusted the Y position to the highlighted op. This
  // is to avoid constantly bringing back the highlighted op to focus when the
  // user scrolls away from it but we still want it to be highlighted.
  bool adjustYPosToHighlightedOp;
  // Set of operations, regions and blocks that should be expanded to reach
  // the op to highlight.
  DenseSet<void *> mlirEntitiesToExpandForHighlightedOp;
  // Op on which we requested the users from the tree view.
  Operation *opToFocus;
  // Set of operations, regions and blocks that were rendered. This is useful
  // to know where to draw use-def lines.
  DenseMap<void *, ImVec2> renderedMLIREntities;
  // Set of operations, regions and blocks that were expanded in the past.
  DenseMap<void *, bool> mlirEntitiesPreviouslyExpanded;
  // Operation, region or block where we asked to collapse children.
  void *mlirEntityWhereWeCollapsed;

  MLIRTreeRenderer() = default;
  MLIRTreeRenderer(Operation *root, MLIRTextCallbacks &callbacks)
      : callbacks(callbacks), root(root), opToHighlight(nullptr),
        adjustYPosToHighlightedOp(true), opToFocus(nullptr) {}

  void clearHighlightedOp() { opToHighlight = nullptr; }

  // Will evaluate what regions, blocks and ops must be expanded in the tree
  // view to highlight the op where the cursor is pointing to.
  void highlightOpOnTreeView(Operation *op) {
    if (!op)
      return;

    // Check if the current line is associated with an operation.
    opToHighlight = op;
    opToFocus = opToHighlight;
    if (!opToHighlight)
      return;

    adjustYPosToHighlightedOp = true;
    // Walk the parent operations and add blocks and regions along the way.
    do {
      auto block = op->getBlock();
      auto region = op->getParentRegion();
      mlirEntitiesToExpandForHighlightedOp.insert(block);
      mlirEntitiesToExpandForHighlightedOp.insert(region);
      op = op->getParentOp();
    } while (op && mlirEntitiesToExpandForHighlightedOp.insert(op).second);
  }

  void clearPerFrameData() {
    // Clearing those at the end of the frame because they are computed on a
    // frame-by-frame basis.
    mlirEntitiesToExpandForHighlightedOp.clear();
    renderedMLIREntities.clear();
  }

  void render() {
    render(*root, 0, false);
    renderUsers();
  }

  void renderUsers() {
    if (!opToFocus)
      return;

    const std::function<ImVec2(Operation *)> walkFromOp = [&](Operation *op) {
      auto it = renderedMLIREntities.find(op);
      if (it != renderedMLIREntities.end())
        return it->second;

      it = renderedMLIREntities.find(op->getBlock());
      if (it != renderedMLIREntities.end())
        return it->second;

      it = renderedMLIREntities.find(op->getParentRegion());
      if (it != renderedMLIREntities.end())
        return it->second;

      return walkFromOp(op->getParentOp());
    };

    const std::function<ImVec2(Block *)> walkFromBlock = [&](Block *block) {
      auto it = renderedMLIREntities.find(block);
      if (it != renderedMLIREntities.end())
        return it->second;

      it = renderedMLIREntities.find(block->getParent());
      if (it != renderedMLIREntities.end())
        return it->second;

      return walkFromOp(block->getParentOp());
    };

    for (auto &&user : opToFocus->getUsers()) {
      auto dst = walkFromOp(user);
      auto src = walkFromOp(opToFocus);

      auto p = src;
      auto q = dst;
      p.x = std::min(p.x, q.x) - 10;
      q.x = p.x;

      ImDrawList *drawList = ImGui::GetWindowDrawList();
      drawList->AddLine(src, p, IM_COL32(255, 0, 0, 255), 1);
      drawList->AddLine(p, q, IM_COL32(255, 0, 0, 255), 1);
      drawList->AddLine(q, dst, IM_COL32(255, 0, 0, 255), 1);
    }

    for (auto &&operand : opToFocus->getOperands()) {
      ImVec2 dst;
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        dst = walkFromBlock(arg.getOwner());
      } else {
        dst = walkFromOp(operand.getDefiningOp());
      }
      auto src = walkFromOp(opToFocus);

      auto p = src;
      auto q = dst;
      p.x = std::min(p.x, q.x) - 10;
      q.x = p.x;

      ImDrawList *drawList = ImGui::GetWindowDrawList();
      drawList->AddLine(src, p, IM_COL32(255, 192, 203, 255), 1);
      drawList->AddLine(p, q, IM_COL32(255, 192, 203, 255), 1);
      drawList->AddLine(q, dst, IM_COL32(255, 192, 203, 255), 1);
    }
  }

  void render(Block &block, uint64_t id, bool recursiveCollapse) {
    registerEntityAsRendered(&block);

    if (mlirEntitiesToExpandForHighlightedOp.contains(&block))
      ImGui::SetNextItemOpen(true, ImGuiCond_Always);
    else if (&block == mlirEntityWhereWeCollapsed) {
      mlirEntityWhereWeCollapsed = nullptr;
      ImGui::SetNextItemOpen(false, ImGuiCond_Always);
    }

    auto treeNode = CollapsibleTreeNode(
        recursiveCollapse, llvm::formatv("bb{}", id).str().c_str(), 0, &block);

    bool &wasExpandedInThePast = mlirEntitiesPreviouslyExpanded[&block];
    if (recursiveCollapse) {
      if (wasExpandedInThePast) {
        wasExpandedInThePast = false;
        for (auto &&[id_, op] : llvm::enumerate(block.getOperations()))
          render(op, id_, true);
      }
    } else if (treeNode) {
      wasExpandedInThePast = true;
      for (auto &&[id_, op] : llvm::enumerate(block.getOperations()))
        render(op, id_, false);
    }
  }

  void render(Region &region, bool recursiveCollapse) {
    if (mlirEntitiesToExpandForHighlightedOp.contains(&region))
      ImGui::SetNextItemOpen(true, ImGuiCond_Always);
    else if (&region == mlirEntityWhereWeCollapsed) {
      mlirEntityWhereWeCollapsed = nullptr;
      ImGui::SetNextItemOpen(false, ImGuiCond_Always);
    }

    registerEntityAsRendered(&region);
    auto treeNode = CollapsibleTreeNode(
        recursiveCollapse,
        llvm::formatv("r{}", region.getRegionNumber()).str().c_str(), 0,
        &region);

    bool &wasExpandedInThePast = mlirEntitiesPreviouslyExpanded[&region];
    if (recursiveCollapse) {
      if (wasExpandedInThePast) {
        wasExpandedInThePast = false;
        for (auto &&[i, block] : llvm::enumerate(region.getBlocks()))
          render(block, i, recursiveCollapse);
      }
    } else if (treeNode) {
      wasExpandedInThePast = true;
      for (auto &&[i, block] : llvm::enumerate(region.getBlocks()))
        render(block, i, recursiveCollapse);
    }
  }

  void render(Type type, bool recursiveCollapse) {
    if (recursiveCollapse)
      ImGui::SetNextItemOpen(false, ImGuiCond_Always);

    auto treeNode = CollapsibleTreeNode(
        recursiveCollapse, type.getAbstractType().getName().data());
    setToolTipOnMLIREntity(type);
    if (treeNode || recursiveCollapse) {
      int id = 0;
      type.walkImmediateSubElements(
          [&](Attribute attr_) {
            ImGui::PushID(id++);
            render(attr_, recursiveCollapse);
            ImGui::PopID();
          },
          [&](Type type) {
            ImGui::PushID(id++);
            render(type, recursiveCollapse);
            ImGui::PopID();
          });
    }
  }

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

  void render(Attribute attr, bool recursiveCollapse) {
    /// TODO: Do the other cases
    if (auto strAttr = dyn_cast<StringAttr>(attr)) {
      ImGui::Text("%s", strAttr.data());
    } else if (auto unitAttr = dyn_cast<UnitAttr>(attr)) {
      ImGui::Text("unit");
    } else if (auto typeAttr = dyn_cast<TypeAttr>(attr)) {
      render(typeAttr.getValue(), recursiveCollapse);
    } else {
      if (recursiveCollapse)
        ImGui::SetNextItemOpen(false, ImGuiCond_Always);

      auto treeNode = CollapsibleTreeNode(
          recursiveCollapse, attr.getAbstractAttribute().getName().data());
      setToolTipOnMLIREntity(attr);
      if (treeNode || recursiveCollapse) {
        int id = 0;
        attr.walkImmediateSubElements(
            [&](Attribute attr_) {
              ImGui::PushID(id++);
              render(attr_, recursiveCollapse);
              ImGui::PopID();
            },
            [&](Type type) {
              ImGui::PushID(id++);
              render(type, recursiveCollapse);
              ImGui::PopID();
            });
      }
    }
  }

  /// Helper for creating a TreeNode that can recursively collapse its children.
  /// It's RAII in that the destructor will TreePop automatically.
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

  CollapsibleTreeNodeGuard ColouredCollapsibleNode(ImU32 colour,
                                                   bool &recursiveCollapse,
                                                   const char *name,
                                                   ImGuiTreeNodeFlags flags = 0,
                                                   void *mlirEntity = nullptr) {
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    auto treeNode =
        CollapsibleTreeNode(recursiveCollapse, name, flags, mlirEntity);
    ImGui::PopStyleColor();

    return treeNode;
  }

  void render(DictionaryAttr dict, bool recursiveCollapse) {
    if (dict.empty())
      return;

    if (recursiveCollapse)
      ImGui::SetNextItemOpen(false, ImGuiCond_Always);

    auto treeNode = ColouredCollapsibleNode(IM_COL32(0, 255, 0, 255),
                                            recursiveCollapse, "attr-dict");
    if (treeNode || recursiveCollapse) {
      for (auto &&namedAttr : dict) {
        if (recursiveCollapse)
          ImGui::SetNextItemOpen(false, ImGuiCond_Always);

        auto treeNode =
            CollapsibleTreeNode(recursiveCollapse, namedAttr.getName().data());
        setToolTipOnMLIREntity(namedAttr.getValue());
        if (treeNode || recursiveCollapse)
          render(namedAttr.getValue(), recursiveCollapse);
      }
    }
  }

  void render(mlir::OperandRange operands, bool recursiveCollapse) {
    if (operands.empty())
      return;

    mlir::OpPrintingFlags flags;
    flags.useLocalScope(false);
    flags.printNameLocAsPrefix(true);
    flags.printUniqueSSAIDs(true);

    if (recursiveCollapse)
      ImGui::SetNextItemOpen(false, ImGuiCond_Always);

    static constexpr ImU32 kOperandColour = IM_COL32(0, 255, 0, 255);
    auto treeNode =
        ColouredCollapsibleNode(kOperandColour, recursiveCollapse, "operands");
    if (treeNode || recursiveCollapse) {
      for (auto &&operand : operands) {
        std::string operandStr;
        llvm::raw_string_ostream stream(operandStr);
        operand.printAsOperand(stream, flags);

        if (recursiveCollapse)
          ImGui::SetNextItemOpen(false, ImGuiCond_Always);

        auto treeNode = ColouredCollapsibleNode(
            kOperandColour, recursiveCollapse, operandStr.c_str());
        setToolTipOnMLIREntity(operand.getType());

        if (treeNode || recursiveCollapse)
          render(operand.getType(), recursiveCollapse);
      }
    }
  }

  void render(mlir::ResultRange results, bool recursiveCollapse) {
    if (results.empty())
      return;

    mlir::OpPrintingFlags flags;
    flags.useLocalScope(false);
    flags.printNameLocAsPrefix(true);
    flags.printUniqueSSAIDs(true);

    if (recursiveCollapse)
      ImGui::SetNextItemOpen(false, ImGuiCond_Always);

    static constexpr ImU32 kResultColour = IM_COL32(0, 255, 0, 255);
    auto treeNode =
        ColouredCollapsibleNode(kResultColour, recursiveCollapse, "results");
    if (treeNode || recursiveCollapse) {
      for (auto &&result : results) {
        std::string resultStr;
        llvm::raw_string_ostream stream(resultStr);
        result.printAsOperand(stream, flags);

        if (recursiveCollapse)
          ImGui::SetNextItemOpen(false, ImGuiCond_Always);

        auto treeNode = ColouredCollapsibleNode(
            kResultColour, recursiveCollapse, resultStr.c_str());
        setToolTipOnMLIREntity(result.getType());

        if (treeNode || recursiveCollapse)
          render(result.getType(), recursiveCollapse);
      }
    }
  }

  void registerEntityAsRendered(void *entity) {
    ImVec2 nodeMin = ImGui::GetItemRectMin();
    ImVec2 nodeSize = ImGui::GetItemRectSize();
    ImVec2 pos = nodeMin;
    pos.y += nodeSize.y / 2;
    renderedMLIREntities[entity] = pos;
  }

  void render(Operation &op, uint64_t id, bool recursiveCollapse) {
    // Given that operations can have the same name, we push an ID to ImGui's
    // stack to make hashes unique.
    ImGui::PushID(id);

    // If this is an op that must be expanded to get to the highilighted op,
    // we force open.
    if (mlirEntitiesToExpandForHighlightedOp.contains(&op))
      ImGui::SetNextItemOpen(true, ImGuiCond_Always);
    else if (&op == mlirEntityWhereWeCollapsed) {
      mlirEntityWhereWeCollapsed = nullptr;
      ImGui::SetNextItemOpen(false, ImGuiCond_Always);
    }

    // If this ops is the highlighted op, we adjust the Y position if it is
    // not visibile and if we have not already done so and we actually
    // highlight it.
    bool isOpToHighlight = &op == opToHighlight;
    ImGuiTreeNodeFlags flags = 0;
    if (isOpToHighlight) {
      if (adjustYPosToHighlightedOp && !ImGui::IsItemVisible()) {
        adjustYPosToHighlightedOp = false;
        ImGui::SetScrollHereY();
      }
      flags |= ImGuiTreeNodeFlags_Selected;
      ImGui::PushStyleColor(ImGuiCol_Header, IM_COL32(255, 127, 0, 255));
    }

    {
      auto treeNode = CollapsibleTreeNode(
          recursiveCollapse, op.getName().getStringRef().data(), flags, &op);

      // Register the current node as being rendered such that we know we can
      // draw a line to it.
      registerEntityAsRendered(&op);

      if (isOpToHighlight)
        ImGui::PopStyleColor();

      // If we are hovering the tree node, display useful information as a tool
      // tip.
      if (ImGui::IsItemHovered()) {
        std::string locString;
        llvm::raw_string_ostream stream(locString);
        auto loc = op.getLoc();
        loc.print(stream);
        ImGui::SetTooltip(
            "%s\nnumOperands = %d\nnumResults = %d\nnumSuccessors = %d",
            locString.c_str(), op.getNumOperands(), op.getNumResults(),
            op.getNumSuccessors());

        if (auto lineLoc = locNumber(&op))
          callbacks.highlightOpInTextEditor(*lineLoc);

        opToFocus = &op;
      }

      bool &wasExpandedInThePast = mlirEntitiesPreviouslyExpanded[&op];

      // If we are currently collapsing everything, first force collapse all
      // attributes. We do not track their state because there is nothing we
      // could hash onto (attributes in MLIR are globally unique, even if
      // attached to different operations). Then, if there is at least one child
      // that is expanded, recurse. If we aren't collapsing, keep track of the
      // number of collapsed children by returning a boolean that tells you if
      // there was something expanded. Note that we do not compute it if the
      // current node is closed which means we will keep the value from the
      // previous frame. This makes sense because you need to expand a node to
      // edit its children.
      if (recursiveCollapse) {
        // This is an optimization: Do not recurse when collapsing recursively
        // if there are no open children operations. On very large graphs with
        // deeply nested attributes, this makes the difference between instant
        // and takes 30 seconds to collapse.
        if (wasExpandedInThePast) {
          wasExpandedInThePast = false;
          render(op.getAttrDictionary(), true);
          render(op.getOperands(), true);
          render(op.getResults(), true);
          for (auto &&region : op.getRegions())
            render(region, true);
        }
      } else if (treeNode) {
        render(op.getAttrDictionary(), false);
        render(op.getOperands(), false);
        render(op.getResults(), false);

        wasExpandedInThePast = true;
        for (auto &&region : op.getRegions())
          render(region, false);
      }
    }
    ImGui::PopID();
  }

  CollapsibleTreeNodeGuard CollapsibleTreeNode(bool &recursiveCollapse,
                                               const char *name,
                                               ImGuiTreeNodeFlags flags = 0,
                                               void *mlirEntity = nullptr) {
    if (recursiveCollapse)
      ImGui::SetNextItemOpen(false, ImGuiCond_Always);

    bool wasExpanded = ImGui::TreeNodeEx(name, flags);
    bool recursiveCollapseBeforeThis = recursiveCollapse;
    if (ImGui::BeginPopupContextItem()) {
      if (ImGui::MenuItem("collapse children")) {
        if (mlirEntity)
          mlirEntityWhereWeCollapsed = mlirEntity;
        recursiveCollapse = true;
      }
      ImGui::EndPopup();
    }

    if (recursiveCollapseBeforeThis)
      ImGui::TreePush(name);

    return CollapsibleTreeNodeGuard(recursiveCollapseBeforeThis, wasExpanded);
  }
};

struct MLIRSourceRenderer {
  MLIRSourceRenderer() = default;
  MLIRSourceRenderer(MLIRTextCallbacks &callbacks) {}
  void render() {
    /// TODO
  }
};

struct MLIRGraphRenderer {
  MLIRGraphRenderer() = default;
  MLIRGraphRenderer(MLIRTextCallbacks &callbacks) {}
  void render() {
    /// TODO
  }
};

static const char *getFontPath() {
  const char *path = getenv("MLIR_VIZ_FONT");
  if (!path) {
    llvm::errs() << "Could not find MLIR_VIZ_FONT or empty path\n";
    exit(1);
  }

  return path;
}

class MLIRVizRenderer {
private:
  // Index of the font size.
  int fontSizeIndex;
  // Text editor that displays the IR.
  TextEditor textEditor;
  // MLIR text but as a vector of lines. This is not stored on the TextEditor
  // class and requesting it rebuilds the vector every time. This is why we have
  // it as a member variable here.
  std::vector<std::string> textAsLines;
  // Mapping between line numbers and operation.
  SmallVector<Operation *> lineNumberToOp;
  // Line numbers that match the filter.
  SmallVector<int> matchedLines;
  // Index of the line we are currently matching.
  std::optional<int> currentMatchedIndex;

  MLIRTreeRenderer treeRenderer;
  MLIRSourceRenderer sourceRenderer;
  MLIRGraphRenderer graphRenderer;

public:
  MLIRVizRenderer(ModuleOp root, const std::string &mlirText)
      : fontSizeIndex(1) {

    // Create all the different views and provide them with callbacks for
    // changing the state of the MLIR text viewer.
    MLIRTextCallbacks callbacks;
    callbacks.highlightOpInTextEditor = std::bind(
        &MLIRVizRenderer::highlightOpInTextEditor, this, std::placeholders::_1);

    // The different views are created using placement new which means between
    // the start of this constructor and this point, the renderers are in an
    // invalid state. It is necessary because I want them to use the same
    // MLIRTextCallbacks. There may be a cleaner way.
    new (&treeRenderer) MLIRTreeRenderer(root, callbacks);
    new (&sourceRenderer) MLIRSourceRenderer(callbacks);
    new (&graphRenderer) MLIRGraphRenderer(callbacks);

    /// Set the theme and set the indentation to be smaller to reduce
    /// horizontal screen usage in deeply nested trees.
    ImGuiStyle style =
        ImGuiTheme::ThemeToStyle(ImGuiTheme::ImGuiTheme_MaterialFlat);
    ImGui::GetStyle() = style;
    ImGui::GetStyle().IndentSpacing = 16;
    /// Load a bunch of fonts of different sizes.
    ImGuiIO &io = ImGui::GetIO();
    io.Fonts->Clear();
    io.Fonts->AddFontDefault();
    auto fontPath = getFontPath();
    for (uint64_t i = 0; i < 6; ++i)
      io.Fonts->AddFontFromFileTTF(fontPath, 13.0f + i);
    io.Fonts->Build();
    io.FontDefault = io.Fonts->Fonts[1];

    /// Setup the text editor where the IR will be shown.
    textEditor.SetPalette(TextEditor::GetLightPalette());
    textEditor.SetLanguageDefinition(
        TextEditor::LanguageDefinition::CPlusPlus());
    textEditor.SetReadOnly(true);
    textEditor.SetText(mlirText);
    textAsLines = textEditor.GetTextLines();

    // Each line will have an operation entry or nullptr if no loc is associated
    // with it.
    lineNumberToOp.resize(textEditor.GetTotalLines(), nullptr);
    root->walk([&](Operation *op) {
      if (auto lineNumber = locNumber(op))
        lineNumberToOp[*lineNumber] = op;
    });
  }

  ~MLIRVizRenderer() = default;

  /// Typical CTRL/Command + (+/-) keybinds for changing the font size.
  void processFontSizeChange() {
    ImGuiIO &io = ImGui::GetIO();
    // Check if Command + Plus was pressed
    bool commandPressed =
        io.KeyMods & ImGuiMod_Super || io.KeyMods & ImGuiMod_Ctrl;
    ; // ⌘ on macOS, Windows key on Windows
    bool plusPressed = ImGui::IsKeyPressed(ImGuiKey_Equal) ||
                       ImGui::IsKeyPressed(ImGuiKey_KeypadAdd);
    bool minusPressed = ImGui::IsKeyPressed(ImGuiKey_Minus) ||
                        ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract);

    if (!commandPressed || !(plusPressed ^ minusPressed))
      return;

    if (plusPressed)
      fontSizeIndex = std::min(io.Fonts->Fonts.size() - 1, fontSizeIndex + 1);
    else
      fontSizeIndex = std::max(0, fontSizeIndex - 1);

    io.FontDefault = io.Fonts->Fonts[fontSizeIndex];
  }

  void renderTextFilter() {
    // Create an imgui input text widget for the regex string.
    bool filterChanged = false;
    ImGui::SetNextItemWidth(256);
    static char buffer[256] = {0};
    filterChanged = ImGui::InputText("Search IR", buffer, sizeof(buffer));
    ImGui::SameLine();
    HelpMarker("Search using a regex");

    // If the filter changed from the previous frame, reset the matches and find
    // the new ones.
    if (filterChanged) {
      currentMatchedIndex.reset();
      matchedLines.clear();
      llvm::Regex pattern(buffer);
      for (auto &&[idx, line] : llvm::enumerate(textAsLines)) {
        if (pattern.match(line.c_str()))
          matchedLines.push_back(idx);
      }
    }

    // If nothing was matched, just exit.
    if (matchedLines.empty())
      return;

    // Otherwise, display buttons to advance through the matched lines.
    ImGui::SameLine();
    bool prev = ImGui::SmallButton("<");
    ImGui::SameLine();
    bool next = ImGui::SmallButton(">");
    bool somethingWasClicked = prev || next;

    // If no button was clicked and there is currently no match, just show the
    // number of matched lines and exit.
    ImGui::SameLine();
    if (!somethingWasClicked && !currentMatchedIndex.has_value()) {
      ImGui::Text("-/%lu", matchedLines.size());
      return;
    }

    // If a button was pressed, first reset the highlighted matched line.
    if (next)
      currentMatchedIndex = 0;
    else if (prev)
      currentMatchedIndex = matchedLines.size() - 1;

    // Then, seek for the next one using the button that was pressed and the
    // current cursor position.
    int cursorPosition = textEditor.GetCursorPosition().mLine;
    for (auto &&[idx, match] : llvm::enumerate(matchedLines)) {
      if (next) {
        if (match < cursorPosition)
          continue;
        currentMatchedIndex = idx;
        break;
      } else if (prev) {
        if (match > cursorPosition)
          continue;
        currentMatchedIndex = idx;
        break;
      }
    }

    // Display which matched line we are focusing at this time.
    ImGui::Text("%d/%lu", *currentMatchedIndex + 1, matchedLines.size());

    // If the cursor is already on the matched line, increment or decrement the
    // index.
    int &currentMatchedIndexValue = *currentMatchedIndex;
    bool cursorAlreadyOnMatch =
        cursorPosition == matchedLines[currentMatchedIndexValue];
    if (next) {
      if (cursorAlreadyOnMatch)
        ++currentMatchedIndexValue;
    } else if (prev) {
      if (cursorAlreadyOnMatch)
        --currentMatchedIndexValue;
    }

    // Adjust index for circular iterations through the matched lines.
    if (currentMatchedIndexValue == int(matchedLines.size()))
      currentMatchedIndexValue = 0;
    else if (currentMatchedIndexValue < 0)
      currentMatchedIndexValue = matchedLines.size() - 1;

    // Finally, highlight the line.
    if (somethingWasClicked) {
      highlightOpInTextEditor(matchedLines[*currentMatchedIndex]);
      treeRenderer.highlightOpOnTreeView(
          lineNumberToOp[matchedLines[*currentMatchedIndex]]);
    }
  }

  void render() {
    // Split the current window in two:
    // Left: Text editor that displays read-only MLIR
    // Right: The different views.
    ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImVec2 windowSize = viewport->Size;
    ImVec2 halfSize = ImVec2(windowSize.x * 0.5f, windowSize.y);

    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(halfSize);

    processFontSizeChange();

    if (ImGui::Begin("IR", nullptr,
                     ImGuiWindowFlags_NoCollapse |
                         ImGuiWindowFlags_NoTitleBar)) {
      ImGui::BeginChild(
          "Bookmarks",
          ImVec2(-FLT_MIN, ImGui::GetTextLineHeightWithSpacing() * 4),
          ImGuiChildFlags_ResizeY | ImGuiChildFlags_Borders);
      renderBookmarks();
      ImGui::EndChild();

      renderTextFilter();

      textEditor.Render("IR");
      processTextEditorCursorEvents();
    }
    ImGui::End();

    ImGui::SetNextWindowPos(
        ImVec2(viewport->Pos.x + halfSize.x, viewport->Pos.y));
    ImGui::SetNextWindowSize(halfSize);
    if (ImGui::Begin("Views", nullptr,
                     ImGuiWindowFlags_NoCollapse |
                         ImGuiWindowFlags_NoTitleBar)) {
      if (ImGui::BeginTabBar("tabs")) {
        if (ImGui::BeginTabItem("Tree")) {
          treeRenderer.render();
          ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Source")) {
          sourceRenderer.render();
          ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Graph")) {
          graphRenderer.render();
          ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
      }
    }
    ImGui::End();
    treeRenderer.clearPerFrameData();
  }

private:
  void renderBookmarks() {
    auto &breakpoints = textEditor.GetBreakpoints();
    if (breakpoints.empty()) {
      HelpMarker("Double click operations in the IR to bookmark them here");
      return;
    }

    // Sort the bookmarks in order of line number.
    SmallVector<int> sortedBookmarks(breakpoints.begin(), breakpoints.end());
    llvm::sort(sortedBookmarks);

    // Evaluate how many white spaces will be needed depending on the line
    // number size as a string.
    const std::string spaces(std::ceil(std::log10(textAsLines.size())), ' ');
    for (int lineNo : sortedBookmarks) {
      // Line numbers are 1-indexed. Therefore, the line number that we should
      // be using when indexing into the text editor should be -1.
      int actualLineNo = lineNo - 1;
      if (lineNumberToOp[actualLineNo]) {
        auto &line = textAsLines[actualLineNo];
        std::string lineNoAsStr = std::to_string(lineNo);
        std::string lineNoPreceededWithSpaces =
            llvm::formatv("{}{}", &spaces.c_str()[lineNoAsStr.size()],
                          lineNoAsStr)
                .str();
        if (ImGui::SmallButton(lineNoPreceededWithSpaces.c_str())) {
          highlightOpInTextEditor(actualLineNo);
          treeRenderer.highlightOpOnTreeView(
              lineNumberToOp[textEditor.GetCursorPosition().mLine]);
        }

        ImGui::SameLine(0, 2);
        ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(192, 0, 0, 255));
        ImGui::PushID(lineNo);
        /// TODO: Figure out why it won't display the trash can on a Mac.
        // ImGui::Checkbox(kIconTrash, &deleted);
        bool deleted = ImGui::SmallButton("X");
        if (deleted)
          breakpoints.erase(lineNo);
        ImGui::PopID();
        ImGui::PopStyleColor();

        if (!deleted) {
          ImGui::SameLine();
          ImGui::Text("%s", line.c_str());
        }
      }
    }
  }

  // The input line number should be 0-indexed.
  void highlightOpInTextEditor(int lineNo) {
    TextEditor::Coordinates coords;
    coords.mLine = lineNo;
    coords.mColumn = 0;
    textEditor.SetCursorPosition(coords);
    treeRenderer.clearHighlightedOp();
  }

  void processTextEditorCursorEvents() {
    // If the cursor changed positions since the last frame, identify all
    // operations, regions and blocks that should be expanded in the tree
    // view.
    if (textEditor.IsCursorPositionChanged()) {
      treeRenderer.highlightOpOnTreeView(
          lineNumberToOp[textEditor.GetCursorPosition().mLine]);
    }

    // If there was double click, add a visual cue with a breakpoint.
    if (textEditor.DoubleClick()) {
      auto &breakpoints = textEditor.GetBreakpoints();
      int actualCursorLine = textEditor.GetCursorPosition().mLine + 1;
      if (breakpoints.contains(actualCursorLine)) {
        breakpoints.erase(actualCursorLine);
      } else if (lineNumberToOp[textEditor.GetCursorPosition().mLine]) {
        // Do not insert unless it is associated with an op.
        breakpoints.insert(actualCursorLine);
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// Main application and rendering loop
//===----------------------------------------------------------------------===//

static void runMainLoop(ModuleOp moduleOp, const std::string &mlirText) {
  auto osWindowHandle = initImGuiWithBackends();
  MLIRVizRenderer renderer(moduleOp, mlirText);

  while (!shouldCloseWindow(osWindowHandle)) {
    pollEventsAndCreateNewFrameForBackends();
    ImGui::NewFrame();
    renderer.render();
    ImGui::Render();
    renderUsingBackends(osWindowHandle);
  }
  deinitImGuiWithBackends(osWindowHandle);
}

static void validateProgramArguments(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << llvm::formatv("Usage: {} <mlir-file>\n", argv[0]);
    exit(1);
  }
}

static std::string loadInputMLIRFile(const std::string &fileName) {
  std::ifstream file(fileName);
  if (!file) {
    llvm::errs() << llvm::formatv("Could not open {}\n", fileName);
    exit(1);
  }
  std::ostringstream ss;
  ss << file.rdbuf();

  return ss.str();
}

static DialectRegistry loadNecessaryDialects() {
  DialectRegistry registry;
  registry.insert<
      // Add the MLIR Dialects.
      arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
      index::IndexDialect, LLVM::LLVMDialect, quant::QuantDialect,
      mlir::ub::UBDialect,
      // Add the Modular Dialects.
      MEF::MEFDialect, DebugInfo::DebugInfoDialect, GRT::GRTDialect, MDialect,
      MGP::MGPDialect, RMO::RMODialect, MOSH::MOSHDialect, MO::MODialect,
      MOGG::MOGGDialect>();
  // Add the KGEN Dialects.
  registerAllKGENDialects(registry);

  return registry;
}

static OwningOpRef<ModuleOp> parseMLIRText(MLIRContext &context,
                                           const std::string &mlirText) {
  context.allowUnregisteredDialects();
  auto module = parseSourceString<ModuleOp>(mlirText, &context);
  if (!module) {
    llvm::errs() << "Could not parse the MLIR\n";
    exit(1);
  }

  return module;
}

int main(int argc, char **argv) {
  validateProgramArguments(argc, argv);
  auto mlirText = loadInputMLIRFile(argv[1]);
  MLIRContext context(loadNecessaryDialects());
  OwningOpRef<ModuleOp> module = parseMLIRText(context, mlirText);
  runMainLoop(*module, mlirText);

  return 0;
}
