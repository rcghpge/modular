//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DriverDescription.td"
include "Support/Driver/CommonOptions.td"

def version : Flag<["--"], "version">,
              Group<CommonOptionGroup>,
              Index<1>,
              HelpText<"Displays version information.">;

def v : Flag<["-"], "v">,
        Group<CommonOptionGroup>,
        Alias<version>,
        Index<0>;
