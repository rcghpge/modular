//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/API/modular/c/common.h"
#include "GenericML/API/modular/c/context.h"
#include "GenericML/API/modular/c/model.h"
#include "GenericML/API/modular/c/tensor.h"
#include "GenericML/API/modular/cpp/aliases.h"
#include "GenericML/Driver/DriverCLOptions.h"
#include "Support/CommonCLOptions.h"
#include "Support/ErrorOr.h"
#include "Support/ML/TensorShape.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdlib>
#include <mutex>
#include <numeric>
#include <vector>

using namespace M;

namespace {

class APICLOptions : public CLOptionsBase {
public:
  using CLOptionsBase::CLOptionsBase;

  cl::list<std::string> paths{"paths", cl::desc("Paths of models")};
  cl::opt<std::string> cacheDir{"cache-dir",
                                cl::desc("Filesystem path of CAS cache"),
                                llvm::cl::Optional};
  cl::opt<size_t> numThreads{"num-threads", cl::desc("Number of threads"),
                             cl::init(0)};

  cl::list<std::string> inputNames{
      "input-names", cl::desc("Specify input names. This is always required "
                              "and it must match the model's input names")};

  cl::list<TensorShape, bool, TensorShapeParser> inputShapes{
      "input-shapes", cl::desc("Specify input shapes. This is always required "
                               "and it must match the model's input tensors in "
                               "the same order as `input-names`")};

  cl::opt<bool> threaded{"threaded",
                         cl::desc("If true, 'concurrent' requests are run on "
                                  "separate foreign threads."),
                         cl::init(false)};
  cl::opt<size_t> numInitialized{
      "num-initialized",
      cl::desc(
          "Number of concurrent 'initialized' models for each model in paths."),
      cl::init(1)};
  cl::opt<size_t> numConcurrentRequesters{
      "num-concurrent-requesters",
      cl::desc("Number of concurrent 'execute' request "
               "threads to fire on each 'initialized' model. CAUTION: This "
               "is currently not safe for tf and tfl models."),
      cl::init(1)};
  cl::opt<size_t> numSequentialRequests{
      "num-sequential-requests",
      cl::desc("Number of sequential 'executes' to "
               "perform in each concurrent request thread."),
      cl::init(1)};
};

/// serializes console output
std::mutex &consoleMutex() {
  static std::mutex *mu = new std::mutex();
  return *mu;
}

/// Thread friendly logger.
template <typename... Arg>
void log(bool threaded, const char *fmt, const Arg &...args) {
  std::string ostr;
  llvm::raw_string_ostream os(ostr);
  if (threaded)
    os << llvm::get_threadid() << ": ";
  os << llvm::format(fmt, args...);
  os << "\n";
  {
    std::lock_guard<std::mutex> l(consoleMutex());
    llvm::outs() << ostr;
    llvm::outs().flush();
  }
}

/// Logs a brief summary of buffer's contents.
void log(bool threaded, ArrayRef<float> buffer) {
  std::string ostr;
  llvm::raw_string_ostream os(ostr);
  if (threaded)
    os << llvm::get_threadid() << ": ";
  os << "Output: [";
  const size_t extract = std::min(static_cast<size_t>(2ul), buffer.size());
  llvm::interleaveComma(buffer.slice(0, extract), os);
  os << " ... ";
  llvm::interleaveComma(buffer.slice(buffer.size() - extract, extract), os);
  os << "]\n";
  {
    std::lock_guard<std::mutex> l(consoleMutex());
    llvm::outs() << ostr;
    llvm::outs().flush();
  }
}

/// Fork-join wrapper.
class TaskGroup final {
public:
  TaskGroup(size_t reserved) { tasks.reserve(reserved); }

  void add_task(std::function<void()> task) {
    tasks.emplace_back(std::move(task));
  }

  void run(bool threaded) {
    if (threaded) {
      std::vector<std::thread> threads;
      threads.reserve(tasks.size());
      for (const auto &task : tasks)
        threads.emplace_back(task);
      for (auto &thread : threads)
        thread.join();
    } else {
      for (const auto &task : tasks)
        task();
    }
  }

  /// Threadsafe method to set an error message if not currently in an error
  /// state.
  void setError(std::string_view msg) {
    std::lock_guard<std::mutex> l(errorMutex);
    if (!hasError) {
      hasError = true;
      errorMessage = msg;
    }
  }

  const std::string &getError() const {
    std::lock_guard<std::mutex> l(errorMutex);
    return errorMessage;
  }

  /// This should only be called from a threadsafe context since it isn't
  /// protected by a mutex.
  bool didErrorOccur() const { return hasError; }

private:
  std::vector<std::function<void()>> tasks;
  std::string errorMessage;
  mutable std::mutex errorMutex;
  bool hasError{false};
};

/// Convenience macros for working with M_Status.
#define RETURN_IF_ERROR(status)                                                \
  do {                                                                         \
    if (M_isError((status).get()))                                             \
      return Error(M_getError((status).get()));                                \
  } while (false)

/// We have a runtime context, but not yet any compiled model.
struct RuntimeContextPhase {
  static ErrorOr<RuntimeContextPhase>
  createRuntimeContext(const APICLOptions &clOptions) {
    RuntimeContextPhase result;
    result.runtimeConfig = M_newRuntimeConfig();
    M_setNumThreads(result.runtimeConfig, clOptions.numThreads);
    StatusPtr status(M_newStatus(), M_freeStatus);
    if (!clOptions.cacheDir.empty()) {
      M_setArtifactPath(result.runtimeConfig, clOptions.cacheDir.c_str(),
                        status.get());
      RETURN_IF_ERROR(status);
    }

    result.context = M_newRuntimeContext(result.runtimeConfig, status.get());
    RETURN_IF_ERROR(status);

    return result;
  }

  RuntimeContextPhase() = default;
  ~RuntimeContextPhase() {
    if (context) {
      M_freeRuntimeContext(context);
      context = nullptr;
    }
    if (runtimeConfig) {
      M_freeRuntimeConfig(runtimeConfig);
      runtimeConfig = nullptr;
    }
  }

  RuntimeContextPhase(const RuntimeContextPhase &that) = delete;
  RuntimeContextPhase &operator=(const RuntimeContextPhase &that) = delete;

  RuntimeContextPhase(RuntimeContextPhase &&that) { swap(that); }
  RuntimeContextPhase &operator=(RuntimeContextPhase &&that) {
    swap(that);
    return *this;
  }

  void swap(RuntimeContextPhase &that) {
    std::swap(runtimeConfig, that.runtimeConfig);
    std::swap(context, that.context);
  }

  M_RuntimeConfig *runtimeConfig = nullptr;
  M_RuntimeContext *context = nullptr;
};

// Pair a model path with its expected input shapes.
// TODO: Better would be to extract the expected tensor specs from a model
// yaml description.
struct ExternalModel {
  std::string path;
  std::vector<TensorShape> inputShapes;
  std::vector<std::string> inputNames;
};

/// We have compiled a model, but not yet initialized it.
struct CompiledModelPhase {
  static ErrorOr<CompiledModelPhase>
  createCompiledModel(bool threaded, const RuntimeContextPhase *runtimeContext,
                      const ExternalModel *externalModel) {
    CompiledModelPhase result;
    result.runtimeContext = runtimeContext;
    result.externalModel = externalModel;

    result.compileConfig = M_newCompileConfig();
    M_setModelPath(result.compileConfig, externalModel->path.c_str());

    // TODO: We need to make framework loading thread safe before we can overlap
    // compilation of models.
    log(threaded, "INFO: Compiling %s", externalModel->path.c_str());
    StatusPtr status(M_newStatus(), M_freeStatus);
    result.compiledModel = M_compileModel(runtimeContext->context,
                                          result.compileConfig, status.get());
    RETURN_IF_ERROR(status);
    M_waitForCompilation(result.compiledModel, status.get());
    RETURN_IF_ERROR(status);

    return result;
  }

  CompiledModelPhase() = default;

  ~CompiledModelPhase() {
    if (compiledModel) {
      M_freeCompiledModel(compiledModel);
      compiledModel = nullptr;
    }
    if (compileConfig) {
      M_freeCompileConfig(compileConfig);
      compileConfig = nullptr;
    }
  }

  CompiledModelPhase(const CompiledModelPhase &that) = delete;
  CompiledModelPhase &operator=(const CompiledModelPhase &that) = delete;

  CompiledModelPhase(CompiledModelPhase &&that) { swap(that); }
  CompiledModelPhase &operator=(CompiledModelPhase &&that) {
    swap(that);
    return *this;
  }

  void swap(CompiledModelPhase &that) {
    std::swap(runtimeContext, that.runtimeContext);
    std::swap(externalModel, that.externalModel);
    std::swap(compileConfig, that.compileConfig);
    std::swap(compiledModel, that.compiledModel);
  }

  const RuntimeContextPhase *runtimeContext = nullptr;
  const ExternalModel *externalModel = nullptr;
  M_CompileConfig *compileConfig = nullptr;
  M_AsyncCompiledModel *compiledModel = nullptr;
};

/// We have an initialized model, but not yet invoked execute upon it.
struct InitializedModelPhase {
  static ErrorOr<InitializedModelPhase>
  createInitializedModel(bool threaded,
                         const CompiledModelPhase *compiledModel) {
    InitializedModelPhase result;
    result.compiledModel = compiledModel;

    StatusPtr status(M_newStatus(), M_freeStatus);
    log(threaded, "INFO: Initializing %s",
        compiledModel->externalModel->path.c_str());
    result.model = M_initModel(compiledModel->runtimeContext->context,
                               compiledModel->compiledModel, status.get());
    RETURN_IF_ERROR(status);

    M_waitForModel(result.model, status.get());
    RETURN_IF_ERROR(status);

    return result;
  }

  InitializedModelPhase() = default;

  ~InitializedModelPhase() {
    if (model) {
      M_freeModel(model);
      model = nullptr;
    }
  }

  InitializedModelPhase(const InitializedModelPhase &that) = delete;
  InitializedModelPhase &operator=(const InitializedModelPhase &that) = delete;

  InitializedModelPhase(InitializedModelPhase &&that) { swap(that); }
  InitializedModelPhase &operator=(InitializedModelPhase &&that) {
    swap(that);
    return *this;
  }

  void swap(InitializedModelPhase &that) {
    std::swap(compiledModel, that.compiledModel);
    std::swap(model, that.model);
  }

  const CompiledModelPhase *compiledModel = nullptr;
  M_AsyncModel *model = nullptr;
};

/// We are invoking an initialized model.
struct ExecutePhase {
  static ErrorOr<ExecutePhase>
  createExecute(const InitializedModelPhase *initializedModel) {
    ExecutePhase result;
    result.initializedModel = initializedModel;

    const std::vector<TensorShape> &inputShapes =
        initializedModel->compiledModel->externalModel->inputShapes;

    const std::vector<std::string> &inputNames =
        initializedModel->compiledModel->externalModel->inputNames;

    size_t numInputs = inputShapes.size();
    result.inputToModel = M_newAsyncTensorMap(
        initializedModel->compiledModel->runtimeContext->context);

    result.inputSpecs.reserve(numInputs);
    result.inputTensors.reserve(numInputs);

    StatusPtr status(M_newStatus(), M_freeStatus);
    for (auto [inputShape, inputName] : llvm::zip(inputShapes, inputNames)) {
      SmallVector<int64_t> shape{};
      shape.reserve(inputShape.getRank());
      llvm::copy(inputShape, std::back_inserter(shape));
      // TODO: Currently assume f32 dtypes for all input tensors.
      M_TensorSpec *inputSpec = M_newTensorSpec(shape.data(), std::size(shape),
                                                M_FLOAT32, inputName.c_str());
      result.inputSpecs.push_back(inputSpec);

      // TODO: eventually allow controlling the input values too.
      std::vector<float> &inputTensor =
          result.inputTensors.emplace_back(inputShape.getNumElements(), 1.0);
      M_borrowTensorInto(result.inputToModel, inputTensor.data(), inputSpec,
                         status.get());
      RETURN_IF_ERROR(status);
    }

    return result;
  }

  ErrorOrSuccess execute(bool threaded) {
    // Currently, there's no way for users to allocate: the outputs are
    // allocated internally in the `execute` call for now.
    StatusPtr status(M_newStatus(), M_freeStatus);
    log(threaded, "INFO: Executing %s",
        initializedModel->compiledModel->externalModel->path.c_str());
    outputs = M_executeModelSync(
        initializedModel->compiledModel->runtimeContext->context,
        initializedModel->model, inputToModel, status.get());
    RETURN_IF_ERROR(status);
    size_t numInputs = M_getNumModelInputs(
        initializedModel->compiledModel->compiledModel, status.get());
    RETURN_IF_ERROR(status);
    size_t numOutputs = M_getNumModelOutputs(
        initializedModel->compiledModel->compiledModel, status.get());
    RETURN_IF_ERROR(status);

    log(threaded, "NumInputs: %d", numInputs);
    log(threaded, "NumOutputs: %d", numOutputs);

    if (numInputs != inputSpecs.size())
      return Error(Twine("expecting ") + Twine(numInputs) +
                   " --inputSpecs to match model arity");
    M_TensorNameArray *outputTensorNames = M_getOutputNames(
        initializedModel->compiledModel->compiledModel, status.get());
    for (size_t i = 0; i < numOutputs; ++i) {
      const char *outputTensorName = M_getTensorNameAt(outputTensorNames, i);
      M_AsyncTensor *tensor =
          M_getTensorByNameFrom(outputs, outputTensorName, status.get());
      RETURN_IF_ERROR(status);

      size_t numElements = M_getTensorNumElements(tensor);
      log(threaded, "NumElements: %d", numElements);

      // TODO: read output dtype value to know how to decode the data
      // correctly for other dtypes
      auto *tensorData = static_cast<const float *>(M_getTensorData(tensor));
      log(threaded, ArrayRef<float>(tensorData, numElements));
      M_freeTensor(tensor);
    }

    return success();
  };

  ExecutePhase() = default;
  ~ExecutePhase() {
    if (outputs) {
      M_freeAsyncTensorMap(outputs);
      outputs = nullptr;
    }
    for (auto inputSpec : inputSpecs)
      M_freeTensorSpec(inputSpec);
    inputSpecs.clear();
    if (inputToModel) {
      M_freeAsyncTensorMap(inputToModel);
      inputToModel = nullptr;
    }
  }

  ExecutePhase(const ExecutePhase &that) = delete;
  ExecutePhase &operator=(const ExecutePhase &that) = delete;

  ExecutePhase(ExecutePhase &&that) { swap(that); }
  ExecutePhase &operator=(ExecutePhase &&that) {
    swap(that);
    return *this;
  }

  void swap(ExecutePhase &that) {
    std::swap(initializedModel, that.initializedModel);
    std::swap(inputToModel, that.inputToModel);
    std::swap(inputSpecs, that.inputSpecs);
    std::swap(inputTensors, that.inputTensors);
    std::swap(outputs, that.outputs);
  }

  const InitializedModelPhase *initializedModel = nullptr;
  M_AsyncTensorMap *inputToModel = nullptr;
  std::vector<M_TensorSpec *> inputSpecs;
  // We'll hold backing buffers for all the input tensors for the duration
  // of the execute call, and only pass 'borrowed buffers' into the model's
  // 'execute' entry point.
  std::vector<std::vector<float>> inputTensors;
  M_AsyncTensorMap *outputs = nullptr;
};

#define EXIT_IF_ERROR(x, e)                                                    \
  auto errOr##x = (e);                                                         \
  if (errOr##x.isError()) {                                                    \
    log(threaded, "ERROR: %s", errOr##x.getError());                           \
    exit(EXIT_FAILURE);                                                        \
  }                                                                            \
  auto(x) = std::move(*errOr##x)

#define THREAD_RETURN_IF_ERROR(tg, x, e)                                       \
  auto errOr##x = (e);                                                         \
  if (errOr##x.isError()) {                                                    \
    log(threaded, "ERROR: %s", errOr##x.getError());                           \
    (tg).setError(errOr##x.getError());                                        \
    return;                                                                    \
  }                                                                            \
  auto(x) = std::move(*errOr##x)

} // namespace

int main(int argc, char **argv) {
  APICLOptions clOptions(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv, "generic C API test driver\n");
  const size_t numPaths = clOptions.paths.size();
  const bool threaded = clOptions.threaded;
  const std::vector<TensorShape> inputShapes(clOptions.inputShapes.begin(),
                                             clOptions.inputShapes.end());

  const std::vector<std::string> inputNames(clOptions.inputNames.begin(),
                                            clOptions.inputNames.end());
  // Collect the 'ExternalModels'.
  // TODO: We probably should take paths to model specification YAMLs so
  // as to extract all the metadata we need.
  std::vector<ExternalModel> externalModels;
  externalModels.reserve(numPaths);
  for (const auto &path : clOptions.paths) {
    ExternalModel &externalModel = externalModels.emplace_back();
    externalModel.path = path;
    externalModel.inputShapes.resize(inputShapes.size());
    llvm::copy(inputShapes, externalModel.inputShapes.begin());

    assert(
        inputNames.size() == inputShapes.size() &&
        "number of input tensor names does not match number of tensor shapes");

    externalModel.inputNames.resize(inputNames.size());
    llvm::copy(inputNames, externalModel.inputNames.begin());
  }

  std::mutex mu; // protects vectors during concurrent execution.

  EXIT_IF_ERROR(runtimeContext,
                RuntimeContextPhase::createRuntimeContext(clOptions));
  // Compile all the models, sequentially and on the main thread.
  // TODO: Appears we have issues with loading libtf or libtensorflow_cc
  // outside of the main thread, so always run sequentially.
  std::vector<CompiledModelPhase> compiledModels;
  compiledModels.reserve(numPaths);
  {
    TaskGroup tg(numPaths);
    for (const auto &externalModel : externalModels) {
      tg.add_task([&]() {
        THREAD_RETURN_IF_ERROR(tg, compiledModel,
                               CompiledModelPhase::createCompiledModel(
                                   threaded, &runtimeContext, &externalModel));
        std::lock_guard<std::mutex> l(mu);
        compiledModels.emplace_back(std::move(compiledModel));
      });
    }
    tg.run(/*threaded=*/false);

    if (tg.didErrorOccur()) {
      log(threaded, "TaskGroup ERROR: %s", tg.getError().c_str());
      return EXIT_FAILURE;
    }
  }

  // Initialize all the models, concurrently if threaded is true.
  std::vector<InitializedModelPhase> initializedModels;
  initializedModels.reserve(numPaths * clOptions.numInitialized);
  {
    TaskGroup tg(numPaths * clOptions.numInitialized);
    for (auto &compiledModel : compiledModels) {
      for (size_t i = 0; i < clOptions.numInitialized; ++i) {
        tg.add_task([&]() {
          THREAD_RETURN_IF_ERROR(tg, initializedModel,
                                 InitializedModelPhase::createInitializedModel(
                                     threaded, &compiledModel));
          std::lock_guard<std::mutex> l(mu);
          initializedModels.emplace_back(std::move(initializedModel));
        });
      }
    }
    tg.run(threaded);

    if (tg.didErrorOccur()) {
      log(threaded, "TaskGroup ERROR: %s", tg.getError().c_str());
      return EXIT_FAILURE;
    }
  }

  // Execute all the models, concurrently over initialized models (if threaded
  // is true), concurrently over multiple 'request' threads per initialized
  // model (if threaded is true), and sequentially over each 'request' thread.
  if (threaded && clOptions.numConcurrentRequesters > 1) {
    // Most models DO NOT support concurrent execution over the
    // same initialized model due to explicitly or implicitly mutable
    // state slot entries in their contexts. This option is supported just
    // to show where the problems are rather than demonstrate expected use
    // of the C API.
    log(threaded, "WARNING: Concurrent request threads is undefined behaviour");
  }
  {
    TaskGroup tg(numPaths * clOptions.numInitialized *
                 clOptions.numConcurrentRequesters);
    for (const auto &initializedModel : initializedModels) {
      for (size_t i = 0; i < clOptions.numConcurrentRequesters; ++i) {
        tg.add_task([&]() {
          for (size_t j = 0; j < clOptions.numSequentialRequests; ++j) {
            THREAD_RETURN_IF_ERROR(
                tg, executingModel,
                ExecutePhase::createExecute(&initializedModel));
            THREAD_RETURN_IF_ERROR(tg, unit, executingModel.execute(threaded));
            (void)unit;
          }
        });
      }
    }
    tg.run(threaded);

    if (tg.didErrorOccur()) {
      log(threaded, "TaskGroup ERROR: %s", tg.getError().c_str());
      return EXIT_FAILURE;
    }
  }

  return 0;
}
