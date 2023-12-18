//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_ENGINECLI_BENCHMARK_MLPERF_INTEGRATION_H
#define GENERICML_ENGINECLI_BENCHMARK_MLPERF_INTEGRATION_H

#include "Support/ForwardDecls.h"
#include "mlperf-loadgen/query_sample_library.h"
#include "mlperf-loadgen/system_under_test.h"
#include "modular/c/types.h"
#include <atomic>

namespace M::BenchmarkCLI {

class GeneratedQuery;
class QueryGenerator;

class QuerySampleLibrary final : public mlperf::QuerySampleLibrary {
public:
  QuerySampleLibrary(M_RuntimeContext *runtimeContext,
                     std::unique_ptr<QueryGenerator> queryGenerator);
  ~QuerySampleLibrary();

  std::string &Name() const override;
  size_t TotalSampleCount() override;
  size_t PerformanceSampleCount() override;

  void LoadSamplesToRam(
      const std::vector<mlperf::QuerySampleIndex> &samples) override;
  void UnloadSamplesFromRam(
      const std::vector<mlperf::QuerySampleIndex> &samples) override;

  GeneratedQuery *getLoadedSample(mlperf::QuerySampleIndex index) {
    return loadedSamples[index].load();
  }

private:
  void loadSample(mlperf::QuerySampleIndex index);
  void unloadSample(mlperf::QuerySampleIndex index);

  M_RuntimeContext *runtimeContext;
  size_t sampleCount;
  std::unique_ptr<QueryGenerator> queryGenerator;
  std::unique_ptr<std::atomic<GeneratedQuery *>[]> loadedSamples;
};

class SystemUnderTest final : public mlperf::SystemUnderTest {
public:
  explicit SystemUnderTest(QuerySampleLibrary *qsl,
                           M_RuntimeContext *runtimeContext,
                           M_AsyncModel *model);

  std::string &Name() const override;
  void IssueQuery(const std::vector<mlperf::QuerySample> &samples) override;
  void FlushQueries() override;

private:
  ErrorOrSuccess
  issueQueryFallible(const std::vector<mlperf::QuerySample> &samples);

  QuerySampleLibrary *qsl;
  M_RuntimeContext *runtimeContext;
  M_AsyncModel *model;
};

} // namespace M::BenchmarkCLI

#endif // GENERICML_ENGINECLI_BENCHMARK_MLPERF_INTEGRATION_H
