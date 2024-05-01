//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

def VisualizeDescription
    : EngineCLICommandDescription<
          "visualize",
          "Generates a file you can use to visualize a model with Netron.app.",
          "Generates a `.maxviz` file from a given TorchScript "
          "or ONNX model, which you can then load into Netron.app. "
          "The maxviz file is a deconstructed version of the model before it "
          "is optimized in the MAX Engine, meaning that it represents the "
          "high-level graph layout using the \"MO\" dialect. For more detail, "
          "see the guide to "
          "[visualize a model](https://docs.modular.com/engine/visualize).",
          [UsageInfo<"options", "model path",
                     /*variadicInput=*/false>]>,
      Index<2>;
