//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "Support/Driver/CommonOptions.td"
include "Support/Driver/DriverCommand.td"

include "../Common/DescriptionBase.td"
include "VisualizeDescription.td"

def OutputOptionGroup : OptionGroup<"Output">, Index<1>;
def CompilationOptionGroup : OptionGroup<"Compilation options">, Index<3>;

def version : Flag<["--"], "version">,
              Group<CommonOptionGroup>,
              Index<1>,
              HelpText<"Displays version information.">;

defm output_directory
    : CommandOption<["--"], "output-directory", OutputOptionGroup, Index<1>,
                    MetaVarName<"DIR">,
                    HelpText<"Directory to produce outputs in.">>;
def : Separate<["-"], "o">,
      Group<OutputOptionGroup>,
      Index<2>,
      MetaVarName<"DIR">,
      Alias<output_directory>;

defm input_generation_schema
    : CommandOption<
          ["--"], "input-generation-schema", CompilationOptionGroup, Index<1>,
          MetaVarName<"PATH">,
          HelpText<"Path to input generation schema file "
                   "(only used for compile-time shape specifications).">>;

defm custom_ops_path
    : CommandOption<["--"], "custom-ops-path", CompilationOptionGroup, Index<2>,
                    MetaVarName<"PATH">, HelpText<"Path to custom ops.">>;
