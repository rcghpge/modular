//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Metrics.h"
#include "Support/Nanobind/TypeCasters.h"
#include "_core/Graph/Bindings.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/vector.h"

namespace nb = nanobind;
using namespace M::Metrics;

void M::Graph::Python::bindMetricsModule(nb::module_ m) {
  m.doc() = R"nbdoc(
      In-process metrics collection for the Modular runtime.
  )nbdoc";

  nb::class_<Histogram::Snapshot>(m, "HistogramSnapshot", R"nbdoc(
      Statistics accumulated by a Histogram instrument between collect() calls.

      All fields reflect data points recorded since the previous collect().
  )nbdoc")
      .def_ro("count", &Histogram::Snapshot::count,
              "Number of data points recorded.")
      .def_ro("sum", &Histogram::Snapshot::sum, "Sum of all recorded values.")
      .def_ro("min", &Histogram::Snapshot::min, "Minimum recorded value.")
      .def_ro("max", &Histogram::Snapshot::max, "Maximum recorded value.");

  nb::class_<CollatedMetrics::CounterSample>(m, "CounterSample", R"nbdoc(
      A single counter reading from a collect() snapshot.

      Counter values are cumulative and never reset.
  )nbdoc")
      .def_ro("name", &CollatedMetrics::CounterSample::name, "Instrument name.")
      .def_ro("value", &CollatedMetrics::CounterSample::value,
              "Cumulative count.");

  nb::class_<CollatedMetrics::GaugeSample>(m, "GaugeSample", R"nbdoc(
      A single gauge value from an instantaneous read.
  )nbdoc")
      .def_ro("name", &CollatedMetrics::GaugeSample::name, "Instrument name.")
      .def_ro("value", &CollatedMetrics::GaugeSample::value,
              "Current gauge value.");

  nb::class_<CollatedMetrics::HistogramSample>(m, "HistogramSample", R"nbdoc(
      A single histogram reading from a collect() snapshot.

      The underlying accumulator is reset on each collect() call, so ``ss``
      reflects only the interval since the previous collect().
  )nbdoc")
      .def_ro("name", &CollatedMetrics::HistogramSample::name,
              "Instrument name.")
      .def_ro("ss", &CollatedMetrics::HistogramSample::ss,
              "Statistics for this collection interval.");

  nb::class_<CollatedMetrics>(m, "CollatedMetrics", R"nbdoc(
      Snapshot of all registered instruments, returned by collect().
  )nbdoc")
      .def_ro("counters", &CollatedMetrics::counters,
              "Counter readings, one entry per registered counter.")
      .def_ro("gauges", &CollatedMetrics::gauges,
              "Gauge readings, one entry per registered gauge.")
      .def_ro("histograms", &CollatedMetrics::histograms,
              "Histogram readings, one entry per registered histogram.");

  m.def("collect", []() { return M::Metrics::metrics().collect(); }, R"nbdoc(
      Snapshot all registered instruments.

      Returns a :class:`CollatedMetrics` containing the current reading of
      every counter and gauge, and a reset snapshot of every histogram.
      Histogram accumulators are cleared on each call; counter and gauge
      values are cumulative.
  )nbdoc");
}
