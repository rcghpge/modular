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

defm output
    : CommandOption<["--"], "output", OutputOptionGroup, Index<1>,
                    MetaVarName<"PATH">,
                    HelpText<"Path / directory to save visualization spec. "
                             "If a similar filename already exists on the specified path, "
                             "we will overwrite it with an updated visualization spec.">>;
def : Separate<["-"], "o">,
      Group<OutputOptionGroup>,
      Index<2>,
      MetaVarName<"PATH">,
      Alias<output>;

defm input_generation_schema
    : CommandOption<
          ["--"], "input-data-schema", CompilationOptionGroup, Index<1>,
          MetaVarName<"PATH">,
          HelpText<"Path to input data schema file "
                   "(only used for compile-time shape specifications).">>;

defm custom_ops_path
    : CommandOption<["--"], "custom-ops-path", CompilationOptionGroup, Index<2>,
                    MetaVarName<"PATH">, HelpText<"Path to custom ops.">>;
