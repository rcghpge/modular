//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/API/python/Types.h"
#include "Support/ErrorOr.h"
#include "gtest/gtest.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using namespace M;

/// A scoped helper class to initialize and finalize the Python interpreter.
///
/// This is defined here instead of using pybind11::scoped_interpreter in order
/// to initialize the interpreter with a custom PyConfig that has an absolute
/// path to the Python interpreter.
/// Otherwise, CPython will search for an executable path for the embedded
/// interpreter using an algorithm specified in Modules/getpath.py.
/// However, the getpath.py algorithm is fragile and dependent on environment
/// variables such as PATH that are used for many other purposes besides
/// running this test.
/// To remove reliance on external factors as much as possible, we use
/// CPython's facility to directly specify the path to the embedded Python
/// interpreter by initializing the interpreter with a PyConfig instead.
///
/// See the CPython API documentation on initializing an interpreter with
/// PyConfig:
/// https://docs.python.org/3/c-api/init_config.html#initialization-with-pyconfig.
class ScopedInterpreter {
public:
  static ErrorOr<ScopedInterpreter>
  getInterpreter(bool initSignalHandlers = true, int argc = 0,
                 const char *const *argv = nullptr,
                 bool addProgramDirToPath = true);

  ScopedInterpreter(const ScopedInterpreter &) = delete;
  ScopedInterpreter(ScopedInterpreter &&other) noexcept {
    isValid = std::exchange(other.isValid, false);
  }
  ScopedInterpreter &operator=(const ScopedInterpreter &) = delete;
  ScopedInterpreter &operator=(ScopedInterpreter &&) = delete;

  ~ScopedInterpreter() {
    if (isValid)
      pybind11::finalize_interpreter();
  }

private:
  ScopedInterpreter() = default;

  bool isValid{true};
};

ErrorOr<ScopedInterpreter>
ScopedInterpreter::getInterpreter(bool initSignalHandlers, int argc,
                                  const char *const *argv,
                                  bool addProgramDirToPath) {
  PyConfig config;
  PyConfig_InitIsolatedConfig(&config);
  // Specify isolated Python from the system, since we are embedding Python.
  config.isolated = 0;
  // Ignore environment variables that influence Python's behaviour, such as
  // PYTHONHOME and PYTHONPATH.
  config.use_environment = 0;
  // Initialize signal handlers such as SIGINT for keyboard interrupt?
  config.install_signal_handlers = initSignalHandlers ? 1 : 0;

  if (auto pythonEnvVar = std::getenv("MODULAR_PYTHON_EXECUTABLE")) {
    auto modularPythonExe = std::string(pythonEnvVar);

    PyStatus status = PyConfig_SetBytesString(&config, &config.executable,
                                              modularPythonExe.c_str());
    if (PyStatus_Exception(status)) {
      PyConfig_Clear(&config);
      return Error("Failed to set python executable");
    }
  }

  pybind11::initialize_interpreter(&config, argc, argv, addProgramDirToPath);

  return ScopedInterpreter{};
}

struct InferenceSessionTest : ::testing::Test {
  // Fail here, in SetUp(), to avoid exceptions in the constructor.
  void SetUp() override {
    if (const char *error = interpreterOr.getError())
      FAIL() << error;
  }

  ErrorOr<ScopedInterpreter> interpreterOr{ScopedInterpreter::getInterpreter()};
};

TEST_F(InferenceSessionTest, DTypePybindConversionTest) {
  EXPECT_EQ(core::mapPybind11Type(core::mapDType(M_BOOL)), M_BOOL);
  EXPECT_EQ(core::mapPybind11Type(core::mapDType(M_INT16)), M_INT16);
  EXPECT_EQ(core::mapPybind11Type(core::mapDType(M_INT32)), M_INT32);
  EXPECT_EQ(core::mapPybind11Type(core::mapDType(M_INT64)), M_INT64);
  EXPECT_EQ(core::mapPybind11Type(core::mapDType(M_UINT8)), M_UINT8);
  EXPECT_EQ(core::mapPybind11Type(core::mapDType(M_UINT16)), M_UINT16);
  EXPECT_EQ(core::mapPybind11Type(core::mapDType(M_UINT64)), M_UINT64);
  EXPECT_EQ(core::mapPybind11Type(core::mapDType(M_FLOAT32)), M_FLOAT32);
  EXPECT_EQ(core::mapPybind11Type(core::mapDType(M_DOUBLE)), M_DOUBLE);

  // we convert M_FLOAT16 to a pybind::dtype::of<float> which gets converted to
  // M_FLOAT32 this is a stop-gap until we have full float16 support. See issue
  // https://github.com/modularml/modular/issues/10844
  EXPECT_EQ(core::mapPybind11Type(core::mapDType(M_FLOAT16)), M_FLOAT32);
  EXPECT_THROW(core::mapDType(M_UNKNOWN), std::runtime_error);
}
