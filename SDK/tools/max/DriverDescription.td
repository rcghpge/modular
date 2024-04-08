//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "Support/Driver/DriverCommand.td"

include "../../lib/EngineCLI/Common/DescriptionBase.td"
include "../../lib/EngineCLI/Benchmark/BenchmarkDescription.td"
include "../../lib/EngineCLI/Visualize/VisualizeDescription.td"

def : DriverDescription<"max", "CLI interface to MAX Engine.",
                        "Allows interacting with Modular MAX Engine through "
                        "the command line.",
                        [UsageInfo<"", /*inputName=*/"command",
                                   /*variadicInput=*/false>]>;
