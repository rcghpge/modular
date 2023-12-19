//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

def VisualizeDescription
    : EngineCLICommandDescription<
          "visualize",
          "Generate a file that can be used to visualize a model with "
          "Netron.app.",
          "Generate a file that can be used to visualize a model with "
          "Netron.app.",
          [UsageInfo<"options", "model path",
                     /*variadicInput=*/false>]>,
      Index<2>;
