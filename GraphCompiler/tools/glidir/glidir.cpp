//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// MLIR graph and tree visualization tool.
//
//===----------------------------------------------------------------------===//

#include "font.h"

#include "GraphCompiler/MGPDialect/Dialect/MGPDialect.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GraphCompiler/RMODialect/RMODialect.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "MLRT/GraphRT/GRTDialect/GRTDialect.h"
#include "MLRT/GraphRT/MEFDialect/MEFDialect.h"
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

#include "Views/GraphView.h"
#include "Views/SourceView.h"
#include "Views/TextView.h"
#include "Views/TreeView.h"

#define GL_SILENCE_DEPRECATION
#include "hello_imgui/imgui_theme.h"
#include "imgui.h"
#include "imgui/imgui_internal.h"
#include "imgui/misc/freetype/imgui_freetype.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h> // Will drag system OpenGL headers
#undef GL_SILENCE_DEPRECATION

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <sstream>
#include <thread>
#include <vector>

using namespace mlir;
using namespace M;

namespace M {
namespace GlidIR {

static constexpr ImVec4 kClearColor = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

//===----------------------------------------------------------------------===//
// Minimal lossless PNG encoder (uncompressed deflate, no external deps)
//===----------------------------------------------------------------------===//

static uint32_t pngCRC32(const uint8_t *data, size_t len) {
  static uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (int i = 0; i < 256; ++i) {
      uint32_t c = (uint32_t)i;
      for (int j = 0; j < 8; ++j)
        c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      table[i] = c;
    }
    init = true;
  }
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; ++i)
    crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFFu;
}

static uint32_t pngAdler32(const uint8_t *data, size_t len) {
  uint32_t s1 = 1, s2 = 0;
  for (size_t i = 0; i < len; ++i) {
    s1 = (s1 + data[i]) % 65521u;
    s2 = (s2 + s1) % 65521u;
  }
  return (s2 << 16) | s1;
}

static void pngPush32(std::vector<uint8_t> &v, uint32_t x) {
  v.push_back((x >> 24) & 0xFF);
  v.push_back((x >> 16) & 0xFF);
  v.push_back((x >> 8) & 0xFF);
  v.push_back(x & 0xFF);
}

static void pngWriteChunk(std::vector<uint8_t> &out, const char *type,
                          const std::vector<uint8_t> &data) {
  pngPush32(out, (uint32_t)data.size());
  size_t crcStart = out.size();
  out.push_back(type[0]);
  out.push_back(type[1]);
  out.push_back(type[2]);
  out.push_back(type[3]);
  out.insert(out.end(), data.begin(), data.end());
  pngPush32(out, pngCRC32(out.data() + crcStart, 4 + data.size()));
}

// Save RGBA pixels (w×h, top-to-bottom) as a lossless PNG file.
// Returns true on success.
static bool savePNG(const std::string &filename, int w, int h,
                    const uint8_t *rgba) {
  // Build filtered scanlines: filter byte 0 (None) + RGB per pixel.
  std::vector<uint8_t> raw;
  raw.reserve((size_t)(1 + w * 3) * h);
  for (int row = 0; row < h; ++row) {
    raw.push_back(0); // filter type: None
    const uint8_t *px = rgba + (size_t)row * w * 4;
    for (int col = 0; col < w; ++col, px += 4) {
      raw.push_back(px[0]);
      raw.push_back(px[1]);
      raw.push_back(px[2]);
    }
  }

  // Wrap in zlib using non-compressed deflate blocks (max 65535 bytes each).
  std::vector<uint8_t> zlib;
  zlib.push_back(0x78); // CMF: deflate, 32 KB window
  zlib.push_back(0x01); // FLG: (0x7801 % 31 == 0, no dict)
  const uint8_t *src = raw.data();
  size_t rem = raw.size();
  while (rem > 0) {
    uint16_t blen = (uint16_t)std::min(rem, (size_t)65535);
    bool last = (blen == rem);
    zlib.push_back(last ? 1u : 0u); // BFINAL | BTYPE=00
    zlib.push_back((uint8_t)(blen & 0xFF));
    zlib.push_back((uint8_t)(blen >> 8));
    uint16_t nlen = (uint16_t)~blen;
    zlib.push_back((uint8_t)(nlen & 0xFF));
    zlib.push_back((uint8_t)(nlen >> 8));
    zlib.insert(zlib.end(), src, src + blen);
    src += blen;
    rem -= blen;
  }
  pngPush32(zlib, pngAdler32(raw.data(), raw.size()));

  // Assemble PNG.
  std::vector<uint8_t> png;
  static const uint8_t kSig[] = {0x89, 0x50, 0x4E, 0x47,
                                 0x0D, 0x0A, 0x1A, 0x0A};
  png.insert(png.end(), kSig, kSig + 8);

  std::vector<uint8_t> ihdr(13);
  ihdr[0] = (w >> 24) & 0xFF;
  ihdr[1] = (w >> 16) & 0xFF;
  ihdr[2] = (w >> 8) & 0xFF;
  ihdr[3] = w & 0xFF;
  ihdr[4] = (h >> 24) & 0xFF;
  ihdr[5] = (h >> 16) & 0xFF;
  ihdr[6] = (h >> 8) & 0xFF;
  ihdr[7] = h & 0xFF;
  ihdr[8] = 8;
  ihdr[9] = 2;
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;
  pngWriteChunk(png, "IHDR", ihdr);
  pngWriteChunk(png, "IDAT", zlib);
  pngWriteChunk(png, "IEND", {});

  std::ofstream f(filename, std::ios::binary);
  if (!f)
    return false;
  f.write((const char *)png.data(), (std::streamsize)png.size());
  return f.good();
}

//===----------------------------------------------------------------------===//
// Offscreen framebuffer for full-graph PNG export
//===----------------------------------------------------------------------===//

// GL 3.0 framebuffer constants — not present in macOS's deprecated 2.1 headers.
static constexpr GLenum kGL_FRAMEBUFFER = 0x8D40;
static constexpr GLenum kGL_COLOR_ATTACHMENT0 = 0x8CE0;

// GL 3.0 framebuffer functions loaded once via GLFW's proc-address table so
// that we never need to include the 3.x system headers directly.
struct FBOFuncs {
  void (*GenFramebuffers)(GLsizei, GLuint *) = nullptr;
  void (*BindFramebuffer)(GLenum, GLuint) = nullptr;
  void (*FramebufferTexture2D)(GLenum, GLenum, GLenum, GLuint, GLint) = nullptr;
  void (*DeleteFramebuffers)(GLsizei, const GLuint *) = nullptr;

  static FBOFuncs &get() {
    static FBOFuncs fns;
    if (!fns.GenFramebuffers) {
      auto load = glfwGetProcAddress;
      fns.GenFramebuffers = reinterpret_cast<void (*)(GLsizei, GLuint *)>(
          load("glGenFramebuffers"));
      fns.BindFramebuffer =
          reinterpret_cast<void (*)(GLenum, GLuint)>(load("glBindFramebuffer"));
      fns.FramebufferTexture2D =
          reinterpret_cast<void (*)(GLenum, GLenum, GLenum, GLuint, GLint)>(
              load("glFramebufferTexture2D"));
      fns.DeleteFramebuffers =
          reinterpret_cast<void (*)(GLsizei, const GLuint *)>(
              load("glDeleteFramebuffers"));
    }
    return fns;
  }
};

// RAII wrapper around an OpenGL FBO + RGBA texture of a fixed size.
struct OffscreenBuffer {
  GLuint fbo = 0, tex = 0;
  int w, h;

  OffscreenBuffer(int w, int h) : w(w), h(h) {
    auto &fns = FBOFuncs::get();
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    fns.GenFramebuffers(1, &fbo);
    fns.BindFramebuffer(kGL_FRAMEBUFFER, fbo);
    fns.FramebufferTexture2D(kGL_FRAMEBUFFER, kGL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, tex, 0);
    fns.BindFramebuffer(kGL_FRAMEBUFFER, 0);
  }

  ~OffscreenBuffer() {
    auto &fns = FBOFuncs::get();
    if (fbo)
      fns.DeleteFramebuffers(1, &fbo);
    if (tex)
      glDeleteTextures(1, &tex);
  }

  void bind() const {
    FBOFuncs::get().BindFramebuffer(kGL_FRAMEBUFFER, fbo);
    glViewport(0, 0, w, h);
    // Background matches the graph canvas color: IM_COL32(22, 25, 32, 255).
    glClearColor(22 / 255.f, 25 / 255.f, 32 / 255.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
  }

  void unbind() const { FBOFuncs::get().BindFramebuffer(kGL_FRAMEBUFFER, 0); }

  // Copy pixels from the FBO with Y-flip (OpenGL = bottom-up, PNG = top-down).
  std::vector<uint8_t> readPixels() const {
    std::vector<uint8_t> raw((size_t)w * h * 4);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, raw.data());
    std::vector<uint8_t> out((size_t)w * h * 4);
    for (int row = 0; row < h; ++row)
      std::memcpy(out.data() + (size_t)row * w * 4,
                  raw.data() + (size_t)(h - 1 - row) * w * 4, (size_t)w * 4);
    return out;
  }
};

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

// Draw the ImGui frame into the back buffer without swapping.
static void drawFrameToBackbuffer(auto window) {
  int displayWidth, displayHeight;
  glfwGetFramebufferSize(window, &displayWidth, &displayHeight);
  glViewport(0, 0, displayWidth, displayHeight);
  glClearColor(kClearColor.x * kClearColor.w, kClearColor.y * kClearColor.w,
               kClearColor.z * kClearColor.w, kClearColor.w);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
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

  GLFWwindow *window = glfwCreateWindow(1280, 720, "GlidIR", nullptr, nullptr);
  if (window == nullptr)
    exit(1);
  glfwMaximizeWindow(window);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glslVersion);

  return window;
}

//===----------------------------------------------------------------------===//
// Rendering
//===----------------------------------------------------------------------===//

/// Returns the path to a Unicode-capable font for BMP icon glyphs.
static std::string findUnicodeFallbackFont() {
  static const char *candidates[] = {
      "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
      "/System/Library/Fonts/Menlo.ttc",
      "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
      "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
  };
  for (auto *path : candidates)
    if (std::ifstream(path).good())
      return path;
  return {};
}

class Renderer {
private:
  GLFWwindow *window;
  float fontSizeScale = 1.0f;
  // Visibility flags for each floating panel.
  bool showBookmarks = false;
  bool showTree = true;
  bool showSource = true;
  bool showGraph = true;

  TextView textView;
  TreeView treeRenderer;
  GraphView graphRenderer;
  SourceView sourceRenderer;

public:
  Renderer(GLFWwindow *window, ModuleOp root, ModuleOp dumpRoot,
           const std::string &dumpText)
      : window(window) {

    // Create all the different views and provide them with callbacks for
    // changing the state of the MLIR text viewer.
    TextViewCallbacks callbacks;
    callbacks.highlightOpInTextViewer = [this](int line) {
      textView.highlightLine(line);
    };
    callbacks.navigateToOp = [this](Operation *op) {
      textView.scrollToOp(op);
      treeRenderer.highlightOpOnTreeView(op);
    };
    callbacks.scrollToOpInTextViewer = [this](Operation *op) {
      textView.scrollToOp(op);
    };

    treeRenderer = TreeView(root, callbacks);
    graphRenderer = GraphView(root, callbacks);
    sourceRenderer = SourceView(root);

    // Give the text view callbacks to drive tree highlighting.
    TextView::TreeCallbacks treeCbs;
    treeCbs.highlightOp = [this](Operation *op) {
      treeRenderer.highlightOpOnTreeView(op);
    };
    treeCbs.clearHighlight = [this]() { treeRenderer.clearHighlightedOp(); };
    textView.setTreeCallbacks(treeCbs);

    /// Set the theme and set the indentation to be smaller to reduce
    /// horizontal screen usage in deeply nested trees.
    ImGuiStyle style =
        ImGuiTheme::ThemeToStyle(ImGuiTheme::ImGuiTheme_MaterialFlat);
    ImGui::GetStyle() = style;
    ImGui::GetStyle().IndentSpacing = 16;
    ImGui::GetStyle().Colors[ImGuiCol_Text] = ImVec4(1.f, 1.f, 1.f, 1.f);
    // In ImGui 1.92, fonts are size-independent vectors. FontSizeBase controls
    // the rendered size; FontScaleDpi is set automatically by
    // ConfigDpiScaleFonts.
    ImGui::GetStyle().FontSizeBase = 13.0f;
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigDpiScaleFonts = true;
    // Load each font once. No size argument needed — ImGui 1.92 rasterizes
    // on demand at whatever size FontSizeBase * scale factors requires.
    io.Fonts->Clear();
    auto fontData = getFontData();
    auto emojiFontData = getEmojiFontData();
    // BMP icon ranges: arrows, math operators, geometric shapes.
    // ⚙ (U+2699) is intentionally excluded here; NotoEmoji supplies it.
    static const ImWchar kBmpIconRanges[] = {
        0x2190, 0x21FF, // Arrows (← ↩ → ↪ …)
        0x2200, 0x22FF, // Mathematical Operators (≡ ⊕ ⊤ …)
        0x2500, 0x25FF, // Geometric Shapes (▣ ◈ …)
        0,
    };
    // Emoji ranges covering all used glyphs.
    static const ImWchar kEmojiRanges[] = {
        0x2600,
        0x26FF, // Misc Symbols (⚙ U+2699)
        0x1F4A0,
        0x1F4FF, // Enclosed symbols (💾 U+1F4BE, 📂 U+1F4C2, 📦 U+1F4E6)
        0x1F530,
        0x1F540, // Symbols (🔴 U+1F534, 🔵 U+1F535)
        0x1F5A0,
        0x1F5FF, // Misc pictographs (🗑 U+1F5D1)
        0x1F7E0,
        0x1F7F0, // Geometric Shapes Extended (🟠🟡🟢🟣 U+1F7E0-3)
        0x1F900,
        0x1F9FF, // Supplemental symbols (🧶 U+1F9F6)
        0,
    };
    // ImGui expects ownership of the font data; transfer via release().
    auto addMemFont = [&](llvm::StringRef data, float size = 0.0f,
                          ImFontConfig *cfg = nullptr,
                          const ImWchar *glyphRanges = nullptr) {
      auto buf = std::make_unique<char[]>(data.size());
      std::memcpy(buf.get(), data.data(), data.size());
      io.Fonts->AddFontFromMemoryTTF((void *)buf.get(), (int)data.size(), size,
                                     cfg, glyphRanges);
      buf.release();
    };
    addMemFont(fontData); // Hack Regular — main font (Fonts[0])
    // Merge BMP icons (arrows, math operators, geometric shapes) from
    // a system Unicode font into Fonts[0].
    auto unicodeFontPath = findUnicodeFallbackFont();
    if (!unicodeFontPath.empty()) {
      ImFontConfig mergeConfig;
      mergeConfig.MergeMode = true;
      io.Fonts->AddFontFromFileTTF(unicodeFontPath.c_str(), 0.0f, &mergeConfig,
                                   kBmpIconRanges);
    }
    // Merge color emoji glyphs (⚙ 📂 📦) from Twemoji Mozilla (COLRv0).
    {
      ImFontConfig mergeConfig;
      mergeConfig.MergeMode = true;
      mergeConfig.FontLoaderFlags = ImGuiFreeTypeLoaderFlags_LoadColor;
      addMemFont(emojiFontData, 0.0f, &mergeConfig, kEmojiRanges);
    }

    // Compute the palette: match the UI theme so the viewer blends in with
    // the rest of the UI.
    auto &uiStyle = ImGui::GetStyle();
    ImU32 uiBg =
        ImGui::ColorConvertFloat4ToU32(uiStyle.Colors[ImGuiCol_WindowBg]);
    ImU32 uiText =
        ImGui::ColorConvertFloat4ToU32(uiStyle.Colors[ImGuiCol_Text]);
    TextViewer::Palette palette = TextViewer::GetDarkPalette();
    palette[(int)TextViewer::PaletteIndex::Background] = uiBg;
    palette[(int)TextViewer::PaletteIndex::Default] = uiText;
    palette[(int)TextViewer::PaletteIndex::Identifier] = uiText;
    palette[(int)TextViewer::PaletteIndex::Comment] = 0xFF55996A;
    palette[(int)TextViewer::PaletteIndex::MultiLineComment] = 0xFF55996A;
    palette[(int)TextViewer::PaletteIndex::CurrentLineFill] = 0xFF007FFF;
    palette[(int)TextViewer::PaletteIndex::CurrentLineFillInactive] =
        0xFF007FFF;

    // Initialize the text view and source viewer with the computed palette.
    textView.init(root, dumpRoot, dumpText, palette);
    sourceRenderer.initViewer(palette);
  }

  ~Renderer() = default;

  // If a save was requested this frame, render every bookmarked node into an
  // offscreen FBO sized to the full graph and write it as a PNG.  Must be
  // called after drawFrameToBackbuffer() but before glfwSwapBuffers().
  void capturePendingSave(GLFWwindow *win) {
    if (!graphRenderer.pendingSave)
      return;
    graphRenderer.pendingSave = false;

    if (graphRenderer.nodes.empty())
      return;

    // Pre-warm all glyphs at the export font size before building the draw
    // list. drawContent calls AddText at (lastLH * scale), which triggers
    // on-demand glyph loading in ImGui 1.92.x. If loading causes an atlas
    // resize, ImGui updates registered draw lists but skips our standalone one,
    // leaving its _CmdHeader.TexRef stale. By pre-loading every glyph here
    // first, the atlas is fully built before _ResetForNewFrame() runs, so the
    // draw list inherits the correct InitialTexRef from the shared data. Any
    // updated textures are then uploaded explicitly before rendering.
    const float exportScale = 1.5f;
    const float exportFontSize = graphRenderer.lastLH * exportScale;
    if (graphRenderer.lastFont) {
      for (const auto &n : graphRenderer.nodes) {
        graphRenderer.lastFont->CalcTextSizeA(exportFontSize, FLT_MAX, -1.0f,
                                              n.name.c_str());
        graphRenderer.lastFont->CalcTextSizeA(exportFontSize, FLT_MAX, -1.0f,
                                              n.displayText.c_str());
      }
    }
    for (ImTextureData *tex : ImGui::GetPlatformIO().Textures)
      if (tex->Status != ImTextureStatus_OK)
        ImGui_ImplOpenGL3_UpdateTexture(tex);

    // Build a standalone ImDrawList that will be submitted to the FBO.
    // _ResetForNewFrame() inherits InitialTexRef from the shared data, which
    // was updated above when glyphs were pre-warmed.
    ImDrawList dl(ImGui::GetDrawListSharedData());
    dl._ResetForNewFrame();

    int outW = 1, outH = 1;
    auto ctx = graphRenderer.buildExportContext(&dl, outW, outH);

    // Set up clip rect and font texture, then draw background + graph content.
    dl.PushClipRect({0, 0}, {(float)outW, (float)outH});
    dl.PushTextureID(ImGui::GetIO().Fonts->TexID);
    dl.AddRectFilled({0, 0}, {(float)outW, (float)outH},
                     IM_COL32(22, 25, 32, 255));
    // Export has no hover or selection state.
    graphRenderer.drawContent(ctx, nullptr, nullptr, nullptr);
    dl.PopTextureID();
    dl.PopClipRect();

    // Package the draw list and render to the offscreen FBO. Pass the platform
    // texture list so RenderDrawData can upload any textures that were updated
    // during drawing (belt-and-suspenders after the pre-warm above).
    OffscreenBuffer buf(outW, outH);
    buf.bind();
    ImDrawData dd;
    dd.Valid = true;
    dd.DisplayPos = ImVec2(0, 0);
    dd.DisplaySize = ImVec2((float)outW, (float)outH);
    dd.FramebufferScale = ImVec2(1, 1);
    dd.Textures = &ImGui::GetPlatformIO().Textures;
    dd.AddDrawList(&dl);
    ImGui_ImplOpenGL3_RenderDrawData(&dd);
    auto pixels = buf.readPixels();
    buf.unbind();

    // Restore the main window framebuffer's viewport.
    int fbW, fbH;
    glfwGetFramebufferSize(win, &fbW, &fbH);
    glViewport(0, 0, fbW, fbH);

    // Save with a timestamped filename in the current directory.
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char tbuf[32];
    std::strftime(tbuf, sizeof(tbuf), "%Y%m%d_%H%M%S", std::localtime(&t));
    std::string fname = std::string("glidir_graph_") + tbuf + ".png";

    if (savePNG(fname, outW, outH, pixels.data()))
      fprintf(stderr, "Saved: %s (%d×%d)\n", fname.c_str(), outW, outH);
    else
      fprintf(stderr, "Failed to save: %s\n", fname.c_str());
  }

  /// Typical CTRL/Command + (+/-) keybinds for changing the font size.
  void processFontSizeChange() {
    // Use GLFW directly to bypass ImGui's key ownership system, which can
    // suppress IsKeyPressed when a widget (e.g. InputText) locks keys.
    bool cmdOrCtrl = glfwGetKey(window, GLFW_KEY_LEFT_SUPER) == GLFW_PRESS ||
                     glfwGetKey(window, GLFW_KEY_RIGHT_SUPER) == GLFW_PRESS ||
                     glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                     glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
    if (!cmdOrCtrl)
      return;

    bool plusDown = glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_KP_ADD) == GLFW_PRESS;
    bool minusDown = glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS ||
                     glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS;

    // Detect rising edge to avoid repeating on hold.
    static bool prevPlus = false, prevMinus = false;
    bool increase = plusDown && !prevPlus;
    bool decrease = minusDown && !prevMinus;
    prevPlus = plusDown;
    prevMinus = minusDown;

    if (!increase && !decrease)
      return;

    if (increase)
      fontSizeScale = std::min(fontSizeScale + 0.1f, 2.5f);
    else
      fontSizeScale = std::max(fontSizeScale - 0.1f, 0.5f);
  }

  void render() {
    processFontSizeChange();
    // Apply font scale every frame so changes take effect on the next NewFrame.
    ImGui::GetStyle().FontScaleMain = fontSizeScale;

    // Split the current window in two:
    // Left: Text viewer that displays read-only MLIR
    // Right: The different views.
    ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImVec2 windowSize = viewport->Size;
    // Top bar: FPS counter via BeginMainMenuBar (handles sizing/padding
    // natively).
    float topBarHeight = 0;
    if (ImGui::BeginMainMenuBar()) {
      topBarHeight = ImGui::GetWindowHeight();
      if (ImGui::BeginMenu("Views")) {
        ImGui::MenuItem("Tree", nullptr, &showTree);
        ImGui::MenuItem("Source", nullptr, &showSource);
        ImGui::MenuItem("Graph", nullptr, &showGraph);
        ImGui::EndMenu();
      }
      if (ImGui::MenuItem("Bookmarks", nullptr, showBookmarks))
        showBookmarks = !showBookmarks;
      char fpsStr[64];
      snprintf(fpsStr, sizeof(fpsStr), "%.1fx  %.1f FPS", fontSizeScale,
               ImGui::GetIO().Framerate);
      ImGui::SetCursorPosX(ImGui::GetContentRegionMax().x -
                           ImGui::CalcTextSize(fpsStr).x);
      ImGui::TextDisabled("%s", fpsStr);
      ImGui::EndMainMenuBar();
    }

    float panelHeight = windowSize.y - topBarHeight;
    ImVec2 contentOrigin =
        ImVec2(viewport->Pos.x, viewport->Pos.y + topBarHeight);

    // Full-viewport DockSpace backing all panels.
    ImGuiID dsId = ImGui::GetID("##MainDS");
    ImGui::SetNextWindowPos(contentOrigin, ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(windowSize.x, panelHeight),
                             ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("##DSHost", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoBringToFrontOnFocus |
                     ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoDocking |
                     ImGuiWindowFlags_NoBackground);
    ImGui::PopStyleVar();
    ImGui::DockSpace(dsId, ImVec2(0, 0));
    // Set up the default layout once: IR on the left, Tree/Source/Graph as
    // tabs on the right. The user can re-arrange freely after first launch.
    static bool dockLayoutInitialized = false;
    if (!dockLayoutInitialized) {
      dockLayoutInitialized = true;
      ImGui::DockBuilderRemoveNode(dsId);
      ImGui::DockBuilderAddNode(dsId, ImGuiDockNodeFlags_DockSpace);
      ImGui::DockBuilderSetNodeSize(dsId, ImVec2(windowSize.x, panelHeight));
      ImGuiID left, right;
      ImGui::DockBuilderSplitNode(dsId, ImGuiDir_Left, 0.5f, &left, &right);
      ImGui::DockBuilderDockWindow("IR", left);
      ImGui::DockBuilderDockWindow("Tree", right);
      ImGui::DockBuilderDockWindow("Source", right);
      ImGui::DockBuilderDockWindow("Graph", right);
      ImGui::DockBuilderFinish(dsId);
    }
    ImGui::End();

    if (ImGui::Begin("IR")) {
      HelpMarker("Read-only MLIR text viewer.\n"
                 "Click a line to navigate the tree.\n"
                 "Double-click an op to bookmark it.\n"
                 "Use the search bar to find ops by regex.\n"
                 "Right-click to bookmark a selection.");
      ImGui::SameLine();
      textView.renderTextFilter();
      textView.getViewer().Render("IR");
      textView.processTextViewerCursorEvents();

      if (textView.getViewer().RightClick())
        ImGui::OpenPopup("##ir_ctx");
      if (ImGui::BeginPopup("##ir_ctx")) {
        auto &lineNumberToOp = textView.getLineNumberToOp();
        int minLine = textView.getViewer().GetSelectionStart().mLine;
        int maxLine = textView.getViewer().GetSelectionEnd().mLine;
        int opCount = 0;
        for (int line = minLine; line <= maxLine; ++line)
          if (line < (int)lineNumberToOp.size() && lineNumberToOp[line])
            ++opCount;
        if (opCount > 0) {
          char label[64];
          snprintf(label, sizeof(label), "Bookmark %d op%s", opCount,
                   opCount == 1 ? "" : "s");
          if (ImGui::MenuItem(label))
            textView.bookmarkSelection(minLine, maxLine,
                                       textView.getActiveBookmarkColor());
        } else {
          ImGui::TextDisabled("No ops in selection");
        }
        ImGui::EndPopup();
      }
    }
    ImGui::End();

    if (showBookmarks) {
      ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
      ImGui::SetNextWindowPos(
          ImVec2(contentOrigin.x + 20, contentOrigin.y + 30),
          ImGuiCond_FirstUseEver);
      if (ImGui::Begin("Bookmarks", &showBookmarks))
        textView.renderBookmarks();
      ImGui::End();
    }

    if (showTree) {
      if (ImGui::Begin("Tree", &showTree,
                       ImGuiWindowFlags_NoScrollbar |
                           ImGuiWindowFlags_NoScrollWithMouse)) {
        HelpMarker(
            "Hierarchical tree of the MLIR module.\n"
            "Click an op to jump to it in the IR text.\n"
            "Right-click for context menu (collapse, open in new window).\n"
            "Use-def arrows are drawn between visible ops.");
        ImGui::BeginChild("TreeScroll", {0, 0}, ImGuiChildFlags_None,
                          ImGuiWindowFlags_HorizontalScrollbar);
        treeRenderer.render();
        ImGui::EndChild();
      }
      ImGui::End();
    }

    if (showSource) {
      if (ImGui::Begin("Source", &showSource)) {
        HelpMarker(
            "Source files referenced by FileLineColLoc locations in the IR.\n"
            "Tabs show each unique source file.\n"
            "The highlighted line corresponds to the op selected in the IR.");
        sourceRenderer.render(
            treeRenderer.opToHighlight, [this](ArrayRef<Operation *> ops) {
              auto &lineNumberToOp = textView.getLineNumberToOp();
              auto &opLineNumber = textView.getOpLineNumber();
              int minLine = std::numeric_limits<int>::max(), maxLine = -1;
              for (Operation *op : ops) {
                auto it = opLineNumber.find(op);
                if (it == opLineNumber.end())
                  continue;
                minLine = std::min(minLine, it->second);
                maxLine = std::max(maxLine, it->second);
              }
              if (minLine > maxLine)
                return;
              (void)lineNumberToOp;
              textView.getViewer().SetCursorPosition({minLine, 0});
              textView.getViewer().SetSelection({minLine, 0},
                                                {maxLine, INT_MAX});
            });
      }
      ImGui::End();
    }

    if (showGraph) {
      if (ImGui::Begin("Graph", &showGraph)) {
        HelpMarker(
            "Use-def graph for bookmarked ops.\n"
            "Bookmark ops by double-clicking lines in the IR viewer.\n"
            "Drag to pan, Ctrl+scroll to zoom, right-click to reset view.\n"
            "Hover a node to highlight it in the IR. Hover an edge to see the "
            "value.");
        ImGui::SameLine();
        if (ImGui::SmallButton("\xf0\x9f\x92\xbe")) // 💾 U+1F4BE
          graphRenderer.pendingSave = true;
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip(
              "Save the graph canvas as a PNG file in the current directory");
        auto &lineNumberToOp = textView.getLineNumberToOp();
        SmallVector<SmallVector<Operation *>> groups(kNumBookmarkColors);
        for (int i = 0; i < kNumBookmarkColors; ++i)
          groups[i] = textView.opsForGroup(i);
        Operation *selectedOp =
            lineNumberToOp[textView.getViewer().GetCursorPosition().mLine];
        graphRenderer.render(groups, selectedOp);
      }
      ImGui::End();
    }

    treeRenderer.clearPerFrameData();
  }
};

//===----------------------------------------------------------------------===//
// Main application and rendering loop
//===----------------------------------------------------------------------===//

static void runMainLoop(ModuleOp moduleOp, ModuleOp dumpModule,
                        const std::string &dumpText) {
  auto osWindowHandle = initImGuiWithBackends();
  Renderer renderer(osWindowHandle, moduleOp, dumpModule, dumpText);

  using Clock = std::chrono::steady_clock;
  static constexpr auto kTargetFrameTime =
      std::chrono::duration<double>(1.0 / 60.0);
  while (!shouldCloseWindow(osWindowHandle)) {
    auto frameStart = Clock::now();
    pollEventsAndCreateNewFrameForBackends();
    ImGui::NewFrame();
    renderer.render();
    ImGui::Render();
    drawFrameToBackbuffer(osWindowHandle);
    // Capture graph canvas before swapping so pixels are still in the back
    // buffer.
    renderer.capturePendingSave(osWindowHandle);
    glfwSwapBuffers(osWindowHandle);
    auto elapsed = Clock::now() - frameStart;
    if (elapsed < kTargetFrameTime)
      std::this_thread::sleep_for(kTargetFrameTime - elapsed);
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

} // namespace GlidIR
} // namespace M

int main(int argc, char **argv) {
  using namespace M::GlidIR;
  validateProgramArguments(argc, argv);
  auto mlirText = loadInputMLIRFile(argv[1]);
  MLIRContext context(loadNecessaryDialects());
  OwningOpRef<ModuleOp> module = parseMLIRText(context, mlirText);

  // Dump the parsed IR without location attributes, then re-parse. The
  // re-parsed ops have no explicit loc(), so the parser assigns each op a
  // FileLineColLoc pointing to its position in the dump text. This gives exact
  // line numbers for the IR text viewer without any regex scanning.
  //
  // The full print is used for re-parsing (elided forms may not be parseable).
  // The elided print is used for display: large inline elements and dialect
  // resource blobs (which come last) are replaced with __elided__, keeping op
  // line numbers identical to the full print while dramatically shrinking the
  // text for large models.
  std::string fullText;
  {
    llvm::raw_string_ostream os(fullText);
    module->print(os);
  }
  OwningOpRef<ModuleOp> dumpModule = parseMLIRText(context, fullText);

  std::string dumpText;
  {
    llvm::raw_string_ostream os(dumpText);
    mlir::OpPrintingFlags flags;
    flags.elideLargeElementsAttrs(32).elideLargeResourceString(64);
    module->print(os, flags);
  }

  runMainLoop(*module, *dumpModule, dumpText);
  return 0;
}
