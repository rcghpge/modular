//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/EngineCLI/Benchmark/MLPerfIntegration.h"
#include "GenericML/EngineCLI/Benchmark/InputGenerator.h"
#include "modular/cpp/aliases.h"
#include "gtest/gtest.h"

using namespace M;
using namespace M::BenchmarkCLI;

namespace {

class TestQueryGenerator final : public QueryGenerator {
public:
  explicit TestQueryGenerator(
      std::function<ErrorOr<std::unique_ptr<GeneratedQuery>>(size_t queryIndex)>
          generateQueryFn)
      : generateQueryFn(std::move(generateQueryFn)) {}

  ssize_t getNumQueries() const override { return -1; }
  ErrorOr<std::unique_ptr<GeneratedQuery>>
  generateQuery(M_RuntimeContext *runtimeContext,
                size_t queryIndex) const override {
    return generateQueryFn(queryIndex);
  }

private:
  std::function<ErrorOr<std::unique_ptr<GeneratedQuery>>(size_t queryIndex)>
      generateQueryFn;
};

class LivenessTrackingQuery final : public GeneratedQuery {
public:
  LivenessTrackingQuery(size_t queryIndex,
                        std::vector<LivenessTrackingQuery *> &liveVec)
      : queryIndex(queryIndex), liveVec(liveVec) {
    liveVec.push_back(this);
  }
  LivenessTrackingQuery(const LivenessTrackingQuery &) = delete;
  LivenessTrackingQuery(LivenessTrackingQuery &&) = delete;
  ~LivenessTrackingQuery() {
    auto it = std::find(liveVec.begin(), liveVec.end(), this);
    EXPECT_NE(it, liveVec.end());
    liveVec.erase(it);
  }

  LivenessTrackingQuery &operator=(const LivenessTrackingQuery &) = delete;
  LivenessTrackingQuery &operator=(LivenessTrackingQuery &&) = delete;

  size_t getQueryIndex() const { return queryIndex; }

  M_AsyncTensorMap *getAsyncTensorMap() const override {
    ADD_FAILURE()
        << "LivenessTrackingQuery::getAsyncTensorMap should not be called";
    return nullptr;
  }

private:
  size_t queryIndex;
  std::vector<LivenessTrackingQuery *> &liveVec;
};

[[maybe_unused]] llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                               const LivenessTrackingQuery &q) {
  return os << "<query idx=" << q.getQueryIndex() << ">";
}

} // namespace

TEST(QuerySampleLibrary, basic) {
  int generateCalls = 0;
  size_t lastQueryIndex = 0;
  std::vector<LivenessTrackingQuery *> liveQueries;
  StatusPtr status(M_newStatus(), M_freeStatus);
  ASSERT_NE(nullptr, status);
  RuntimeConfigPtr config(M_newRuntimeConfig(), M_freeRuntimeConfig);
  ASSERT_NE(nullptr, config);
  RuntimeContextPtr runtimeContext(
      M_newRuntimeContext(config.get(), status.get()), M_freeRuntimeContext);
  EXPECT_FALSE(M_isError(status.get())) << M_getError(status.get());
  ASSERT_NE(nullptr, runtimeContext);
  QuerySampleLibrary qsl(
      runtimeContext.get(),
      std::make_unique<TestQueryGenerator>([&](size_t queryIndex) {
        ++generateCalls;
        lastQueryIndex = queryIndex;
        return std::make_unique<LivenessTrackingQuery>(queryIndex, liveQueries);
      }));
  EXPECT_EQ(0, generateCalls);
  EXPECT_EQ(size_t{0}, liveQueries.size());
  EXPECT_EQ(nullptr, qsl.getLoadedSample(3));
  qsl.LoadSamplesToRam({3, 4});
  EXPECT_EQ(2, generateCalls);
  ASSERT_EQ(size_t{2}, liveQueries.size());
  EXPECT_EQ(liveQueries.at(0), qsl.getLoadedSample(3));
  EXPECT_EQ(liveQueries.at(1), qsl.getLoadedSample(4));
  EXPECT_EQ(2, generateCalls);
  EXPECT_EQ(size_t{2}, liveQueries.size());
  qsl.LoadSamplesToRam({5});
  EXPECT_EQ(3, generateCalls);
  ASSERT_EQ(size_t{3}, liveQueries.size());
  EXPECT_EQ(liveQueries.at(2), qsl.getLoadedSample(5));
  EXPECT_EQ(3, generateCalls);
  EXPECT_EQ(size_t{3}, liveQueries.size());
  qsl.UnloadSamplesFromRam({3});
  EXPECT_EQ(3, generateCalls);
  ASSERT_EQ(size_t{2}, liveQueries.size());
  EXPECT_EQ(nullptr, qsl.getLoadedSample(3));
  EXPECT_EQ(liveQueries.at(0), qsl.getLoadedSample(4));
  qsl.UnloadSamplesFromRam({4, 5});
  EXPECT_EQ(3, generateCalls);
  ASSERT_EQ(size_t{0}, liveQueries.size());
  EXPECT_EQ(nullptr, qsl.getLoadedSample(5));
}

// SystemUnderTest not tested here since it is hard to test outside of an
// integration test.
