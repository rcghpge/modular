//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// MLIR-lowering policy for targets serviced by a dynamically-loaded compiler
// plugin: a registered dispatcher that resolves a triple to the
// `TargetLowering` each loaded plugin hands back through its
// `M_KGEN_getTargetLowering` entry point. The dlopen'd plugin cannot register
// with this process's registry at static-init, so the dispatcher pulls the
// lowering at resolve time instead.
//
//===----------------------------------------------------------------------===//

#include "Target/TargetLowering.h"

#include "Mojo/Support/mcl/PluginUtils.h"

#include "llvm/TargetParser/Triple.h"

#include <vector>

namespace M::KGEN {
namespace {

/// Signature of a plugin's `M_KGEN_getTargetLowering` entry point.
using GetTargetLoweringFn = const TargetLowering *(*)();

/// Loaded plugins and the `TargetLowering` each contributes. The PluginManager
/// owns the plugins the lowerings point into, so it is kept alongside them in a
/// function-local static that lives for the process.
struct LoadedPlugins {
  PluginManager manager;
  std::vector<const TargetLowering *> lowerings;

  LoadedPlugins() {
    for (const std::unique_ptr<Plugin> &plugin : manager.getPlugins()) {
      ErrorOr<GetTargetLoweringFn> fnOr = getFunction<GetTargetLoweringFn>(
          plugin->getHandle(), "M_KGEN_getTargetLowering");
      if (fnOr.isError())
        continue;
      if (const TargetLowering *lowering = (*fnOr)())
        lowerings.push_back(lowering);
    }
  }
};

const LoadedPlugins &loadedPlugins() {
  static LoadedPlugins plugins;
  return plugins;
}

/// Dispatcher for plugin-contributed lowerings: never handles a target itself
/// (`matches()` is false); `resolve()` hands off to a plugin's lowering, which
/// the registry prefers over a direct match.
class PluginLowering final : public TargetLowering {
public:
  llvm::StringRef name() const override { return "plugin"; }

  bool matches(const llvm::Triple &) const override { return false; }

  const TargetLowering *resolve(const llvm::Triple &triple) const override {
    for (const TargetLowering *lowering : loadedPlugins().lowerings) {
      if (lowering->matches(triple))
        return lowering;
    }
    return nullptr;
  }

  // This is max only target
  bool isBaseTarget() const override { return false; }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetLowering<PluginLowering> registerPluginLowering;
#pragma GCC diagnostic pop

} // namespace
} // namespace M::KGEN
