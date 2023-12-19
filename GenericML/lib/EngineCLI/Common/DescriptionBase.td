//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

class EngineCLICommandDescription<string subcommand, string summary,
                                  string description, list<UsageInfo> usages>
    : CommandDescription</*executable=*/"max", subcommand, summary, description,
                         usages>;
