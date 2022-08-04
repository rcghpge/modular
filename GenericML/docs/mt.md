# `mt` <u>m</u>odular <u>t</u>ool

`mt` is meant to be a driver for functionally *any* framework that we want to
use. It uses an API defined in [DriverAPI.h](../include/GenericML/Driver/DriverAPI.h)
to call into frameworks built as dynamic libraries and provide the basic commands
we expect from a driver like `-execute`, `-inspect`, etc.

The tool is designed with extensibility at the forefront - anyone who defines
the correct API can export a dynamic library that `mt` can load and take
advantage of the tooling it provides.

## How does `mt` work?
The first step of `mt` is to sniff the format of the input file. Depending on
what is inside the file, `mt` will choose a framework to load or stop with an
error if the correct framework cannot be determined. The user can also specify
a framework manually with the `--load` command, this will stop `mt` from trying
to sense the input format and will force it to use the provided framework. If
the input file is not in the expected format for the loaded framework, `mt`
will provide an error message and exit.

> NOTE: The way that stock-* frameworks are supported is by specifying the
> `--load=<prefix>stock-<framework>-plugin.<suffix>` option. Otherwise, the file
> would be auto-sensed and the modular version of the framework would be
> loaded.

The next step is to perform any prerequisites to the command specified. For
example, if the input file is a TFLite flatbuffer file and the command was
`-execute`, `mt` will load the `tfl` framework and ask it to compile the input
into a BEF file, then ask the framework to execute the resulting BEF file.

The tool has a set of command line options that it recognizes and parses into
a C++ structure. These are the set of options that can be found in
[DriverCLOptions.h](../include/GenericML/Driver/DriverCLOptions.h). These
options are passed into the loaded framework. For example:
```shell
$  mt --execute .derived/models/tflite/MobileBert-float-384-20200602.tflite \
      --load=libstock-tflite-plugin.dylib \
      --model-specification=Models/TFLite/MobileBert-float-384-20200602.yaml \
      --delegate=xnnpack
```
`mt` clearly doesn't know anything about the `--delegate` option, as it only has
meaning for the `stock-tflite-plugin` framework. This exact pattern should be
used for all framework-specific options.

### Consequences
The main consequence of this design (that is loading dynamic libraries at runtime)
is that you now have to have at least the dynamic libraries you care about along
with the `mt` binary itself. The table below contains the binaries for each
framework, along with their CMake target to simplify building the target.

> NOTE: The dynamic library prefixes/suffixes have been normalized to the Linux
> standard for simplicity - please insert your platform's dynamic library
> prefix/suffix where appropriate.

| CMake target | Binary | Framework |
|--------------|--------|-----------|
| `tfl` | `libtfl.so` | Modular TFLite |
| `stock-tflite-plugin` | `libstock-tflite-plugin.so` | Stock TFLite |

Another consequence is that `mt --help` cannot work the way many users expect
it to. Passing the help option to `mt` will result in getting the `mt` help,
not any framework options. In order to get the framework-specific help options you
can do (for example):
```shell
$ mt --load=tfl --framework-help
USAGE: mt [options] <input file>

OPTIONS:

Color Options:

  --color                         - Use colors in output (default=autodetect)

General options:

  --allocator=<value>             - Specify allocator type:
    =malloc                       -   System malloc/free
    =leak-checker                 -   Allocator with leak checking
    =profiler                     -   Allocator with profiling and leak checking
  --benchmark                     - Measure each invocation of the model inference.
  ...
  --save-temps                    - Store the usual 'temporary' intermediate files permanently in the directory specified by -temps-dir (defaults to the output directory); name them as auxiliary output files. This option only changes the behavior of the `compile` and `execute` commands.
  --temps-dir=<string>            - The directory in which to store 'temporary' intermediate files. No files will be saved here unless `-save-temps` is also specified.
  --verify-diagnostics            - Check that emitted diagnostics match expected-* lines on the corresponding line
  --workqueue=<value>             - Specify workqueue type:
    =default                      -   Auto-select based on # threads
    =single-thread                -   Work queue that only ever uses one thread
    =thread-pool                  -   Default threaded work queue based on std::thread
    =sharded-semaphore            -   Thread pool work queue with sharded semaphore (experimental)
    =nine                         -   'No Interesting Name' Experimental Work Queue
```

## Scenarios
`mt` supports running what's called a 'scenario'. This means a group of options
that will be passed to the loaded framework before a call to `M_runBinaryFile`.
This enables the user to do things like:
```shell
mt -execute --scenario="--num-threads=2" --scenario="--num-threads=4" /path/to/file.tflite
```
This command would compile the file once, then execute it twice with the two
different scenarios, in this case, LLCL would receive a 2-thread threadpool
and then a 4-thread threadpool.

Another thing you might want is the ability to continue in case of failure.
This can be useful when you're not worried about the first error, for example.
In that case, you can do something like:
```shell
mt -execute --scenario="nofail:--flaky-option=true" --scenario="--flaky-option=off" /path/to/file.tflite
```
This command would ignore an error in the first scenario, but would not ignore
any error in the second. If `nofail:` is provided in the last scenario, the
return code of the tool will be EXIT_SUCCESS (0).

## Graph Verification Tooling
WIP
