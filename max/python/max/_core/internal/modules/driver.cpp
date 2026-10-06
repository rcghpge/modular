//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
// Driver module: exports Buffer class (renamed from Tensor)
#include "MLRT/Driver/DeviceContext/LaunchTraceRegistry.h"
#include "Support/Nanobind/TypeCasters.h" // IWYU pragma: keep (type casters)
#include "_core/DLPack/Utils.h"
#include "_core/Driver/Buffer.h"
#include "_core/Driver/ChunkedStagingRegion.h"
#include "_core/Driver/CompletionFlag.h"
#include "_core/Driver/Device.h"
#include "_core/Graph/Bindings.h"
#include "_core/Profiler/Trace.h"
#include "nanobind/nanobind.h"
#include "nanobind/operators.h"
#include "nanobind/stl/shared_ptr.h" // IWYU pragma: keep (type casters)
#include "nanobind/stl/string.h"     // IWYU pragma: keep (type casters)
#include "nanobind/stl/vector.h"     // IWYU pragma: keep (type casters)
#include <dlpack/dlpack.h>
#include <stdexcept>

namespace nb = nanobind;

// For named argument shorthand.
// See:
// https://nanobind.readthedocs.io/en/stable/basics.html#keyword-and-default-arguments
using namespace nanobind::literals;
using namespace M;
using namespace Driver;

extern "C" const char *AsyncRT_DeviceContext_enableAllPeerAccess();

// This function wraps the internals of the overloads of Tensor.to().
// In each of these, the method will return the same Python object if the
// requested device is the same as the one the device is already on, otherwise
// it will return a new tensor on the target device.
nb::typed<nb::object, Python::Buffer> static tensorTo(
    nb::pointer_and_handle<Python::Buffer> self,
    Python::PyDeviceTy &currentDevice, Python::PyDeviceTy &targetDevice) {
  // Pinned host memory and device memory can have the same
  // device context, but they're different memory spaces (host RAM
  // vs GPU VRAM). Always copy if the source is pinned memory.
  if (*currentDevice->ctx == *targetDevice->ctx && !self.p->isPinned()) {
    return nb::borrow(self.h);
  } else {
    return nb::cast(self.p->to(targetDevice));
  }
}

void M::Graph::Python::bindDriverModule(nb::module_ m) {
  namespace Python = M::Driver::Python;
  m.doc() = R"nbdoc(
        MAX Driver Python bindings.

        Provides low-level access to hardware devices and memory management.
    )nbdoc";

  nb::class_<Python::Device>(m, "Device", R"nbdoc(
      Represents a compute device available for tensor operations.

      This is the base class for :class:`CPU` and :class:`Accelerator`.
      Do not instantiate this class directly; use :class:`CPU` for host
      devices and :class:`Accelerator` for any hardware accelerator.

      .. code-block:: python

          from max import driver

          cpu = driver.CPU()
          gpu = driver.Accelerator()
  )nbdoc")
      .def(
          "can_access",
          [](Python::Device &self, Python::Device &other) -> bool {
            return self.canAccess(other);
          },
          "other"_a,
          R"nbdoc(
         Checks if this device can directly access memory of another device.

         .. code-block:: python

              from max import driver

              gpu0 = driver.Accelerator(id=0)
              gpu1 = driver.Accelerator(id=1)

              if gpu0.can_access(gpu1):
                  print("GPU0 can directly access GPU1 memory.")

         Args:
             other (Device): The other device to check peer access against.

         Returns:
             bool: True if peer access is possible, False otherwise.
         )nbdoc")
      .def("synchronize", &Python::Device::synchronize, R"nbdoc(
        Ensures all operations on this device complete before returning.

        Raises:
            ValueError: If any enqueued operations had an internal error.
      )nbdoc")
      .def_prop_ro("is_host", &Python::Device::isHost, R"nbdoc(
        Whether this device is the CPU (host) device.

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.is_host
      )nbdoc")
      .def_prop_ro("is_host_unified", &Python::Device::isHostUnified, R"nbdoc(
        Whether this device and the host draw from one physical memory pool.

        Reports hardware topology, so it does not predict whether a particular
        buffer is readable from the host.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            device.is_host_unified
      )nbdoc")
      .def_prop_ro(
          "stats",
          [](Python::Device &self) {
            auto json = nb::module_::import_("json");
            return nb::borrow<nb::typed<nb::mapping, nb::str, nb::any>>(
                json.attr("loads")(self.getStats()));
          },
          R"nbdoc(
        Returns utilization data for the device.

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            stats = device.stats

        Returns:
            dict: A dictionary containing device utilization statistics.
      )nbdoc")
      .def_prop_ro("max_single_alloc_size",
                   &Python::Device::getMaxSingleAllocSize,
                   "Largest single contiguous allocation, in bytes.")
      .def_prop_ro("label", &Python::Device::getLabel, R"nbdoc(
        Returns device label.

        Possible values are:

        - ``cpu`` for host devices.
        - ``gpu`` for accelerators.

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.label
      )nbdoc")
      .def_prop_ro("api", &Python::Device::getApi, R"nbdoc(
        Returns the API used to program the device.

        Possible values are:

        - ``cpu`` for host devices.
        - ``cuda`` for NVIDIA GPUs.
        - ``hip`` for AMD GPUs.
        - ``metal`` for Apple GPUs.

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.api
      )nbdoc")
      .def_prop_ro("architecture_name", &Python::Device::getArchName,
                   R"nbdoc(
        Returns the architecture name of the device.

        Examples of possible values:

        - ``gfx90a``, ``gfx942`` for AMD GPUs.
        - ``sm_80``, ``sm_86`` for NVIDIA GPUs.
        - CPU devices raise an exception.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            device.architecture_name
      )nbdoc")
      .def_prop_ro("model_name", &Python::Device::getModelName, R"nbdoc(
        Returns the model name of the device.

        Examples of possible values:

        - ``NVIDIA H100 80GB HBM3`` for an H100.
        - ``NVIDIA B200`` for a B200.
        - ``AMD Instinct MI300X`` for an MI300X.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            device.model_name
      )nbdoc")
      .def_prop_ro("id", &Python::Device::getId, R"nbdoc(
        Returns a zero-based device id.

        For a CPU device this is always 0.
        For GPU accelerators this is the id of the device relative to this host.
        Along with the ``label``, an id can uniquely identify a device,
        e.g. ``gpu:0``, ``gpu:1``.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            device_id = device.id

        Returns:
            int: The device ID.
      )nbdoc")
      .def_prop_ro("default_queue", &Python::Device::defaultStream, R"nbdoc(
        Returns the default queue for this device.

        The default queue is initialized when the device object is created.

        Returns:
            DeviceQueue: The default execution queue for this device.
      )nbdoc")
      .def_prop_ro("is_compatible", &Python::Device::isCompatible, R"nbdoc(
        Returns whether this device is compatible with MAX.

        Returns:
            bool: True if the device is compatible with MAX, False otherwise.
      )nbdoc")
      .def(
          "__unsafe_enqueue_py_host_func",
          [](Python::Device &self, nb::callable &&fn) {
            self.defaultStream()->enqueueHostFunc(std::move(fn));
          },
          "fn"_a,
          R"nbdoc(
          Enqueues a Python callable to run on the host after preceding work.

          The callable runs on a driver thread once the device's default
          queue reaches this point, after all previously enqueued work has
          completed. It must not call any device APIs (per the
          ``cuLaunchHostFunc`` contract). Currently only supported on CUDA
          devices.

          This API is intentionally namespaced with a ``__unsafe_`` prefix
          to discourage casual use: there is no safety net for callbacks
          that capture state that outlives the compiled graph, and
          misuse can deadlock or corrupt memory.

          Args:
              fn (Callable[[], None]): A zero-argument callable.

          Raises:
              RuntimeError: If the underlying device does not support host
                  callbacks, or if the driver rejects the enqueue.
      )nbdoc")
      .def(
          "__unsafe_enqueue_async_py_host_func",
          [](Python::Device &self, nb::callable &&fn,
             const Python::CompletionFlag &flag, uint64_t value,
             const Python::CPU &cpu) {
            self.defaultStream()->enqueueAsyncHostFunc(std::move(fn), flag,
                                                       value, cpu);
          },
          "fn"_a, "flag"_a, "value"_a, "cpu"_a,
          R"nbdoc(
          Async kickoff variant of ``__unsafe_enqueue_py_host_func``.

          Like ``__unsafe_enqueue_py_host_func``, except the kickoff host
          node dispatches ``fn`` onto ``cpu``'s AsyncRT worker pool and
          returns immediately, so the GPU queue can proceed to
          subsequent nodes concurrently with ``fn`` running on an
          AsyncRT worker thread.

          When ``fn`` finishes, the worker atomic-stores ``value``
          (release ordering) to the 64-bit memory at ``flag``. Pair with
          ``DeviceQueue.wait_for_host_value(flag, value)`` on the same
          queue to gate the downstream consumer kernel.

          The trampoline keeps refcounts on ``flag``'s underlying MLRT
          allocation AND on ``cpu``'s AsyncRT CPUDevice, so neither can
          be released out from under the in-flight task even if the
          Python wrappers are GC'd between enqueue and signal.

          Args:
              fn (Callable[[], None]): A zero-argument callable.
              flag (CompletionFlag): The completion flag the worker will
                  signal when ``fn`` returns.
              value (int): The 64-bit value to store on completion
                  (matched against the
                  ``wait_for_host_value(flag, value)`` consumer).
              cpu (CPU): The CPU device whose AsyncRT worker pool will
                  execute ``fn``. Typically just ``max.driver.CPU()``;
                  callers needing fine-grained scheduling can pass a
                  specific CPU device.

          Raises:
              RuntimeError: If the underlying device does not support
                  host callbacks, if the supplied ``cpu`` has no
                  associated AsyncRT CPUDevice, or if the driver rejects
                  the enqueue.
      )nbdoc")
      .def("__str__", &Python::Device::getAsString)
      .def("__repr__", &Python::Device::getAsString)
      .def(nb::self == nb::self,
           // This gives a mypy warning if we don't replace the type of 'other'
           // with 'object'. See
           // https://nanobind.readthedocs.io/en/latest/typing.html#functions
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def("__hash__", &Python::Device::hash)
      .def(
          "_device_context_ptr",
          [](Python::Device &self) {
            auto *ctx = self.ctx.getPointer();
#if MODULAR_HAL_DEVICE_CONTEXT
            // Under the HAL backend the Mojo `DeviceContext` is the source of
            // truth and the C++ object merely wraps it. Mojo consumers
            // reconstruct via `DeviceContext._from_unsafe_cpp_handle`, which
            // expects the boxed Mojo handle - so hand out `mojoHandle()`
            // rather than the C++ wrapper.
            return reinterpret_cast<uintptr_t>(ctx->mojoHandle());
#else
            return reinterpret_cast<uintptr_t>(ctx);
#endif
          },
          "Gets the device context pointer.")
      .def_static("cpu", &Python::getCpuDevice, "id"_a = -1,
                  "Creates a CPU device. The id is ignored currently.");

  // Note the use of __new__ here instead of __init__. This allows us to use a
  // factory function on the C++ side to produce the Accelerator type from a
  // cache. We could also use __init__ here if we changed the C++
  // implementation.
  nb::class_<Python::Accelerator, Python::Device>(m, "Accelerator")
      .def(nb::new_(&Python::getAccelerator), "id"_a = -1,
           R"nbdoc(
        Creates an accelerator device with the specified ID and memory limit.

        Represents any hardware accelerator attached to the host: the
        graph compiler reaches CUDA, HIP, Metal and plugin-provided
        backends alike through this one device class. Use
        ``isinstance(device, Accelerator)`` to mean "any non-CPU device",
        and :attr:`api` to tell the concrete backends apart.

        Repeated instantiations with a previously-used device-id will still
        refer to the first such instance that was created. This is especially
        important when providing a different memory limit: only the value
        (implicitly or explicitly) provided in the first such instantiation
        is effective.

        .. code-block:: python

          from max import driver
          # Create default accelerator (usually first available GPU)
          device = driver.Accelerator()
          # Or specify GPU id
          device = driver.Accelerator(id=0)  # First GPU
          device = driver.Accelerator(id=1)  # Second GPU
          # Get device id
          device_id = device.id

        Args:
            id (int, optional): The device ID to use. Defaults to -1, which selects
                the first available accelerator.

        Returns:
            Accelerator: A new Accelerator device object.
        )nbdoc");

  nb::class_<Python::CPU, Python::Device>(m, "CPU").def(
      nb::new_(&Python::getCpuDevice), "id"_a = -1, R"nbdoc(
          Creates a CPU device.

          .. code-block:: python

              from max import driver
              # Create default CPU device
              device = driver.CPU()
              # Device id is always 0 for CPU devices
              device_id = device.id

          Args:
              id (int, optional): The device ID to use.
                  Defaults to -1.

          Returns:
              CPU: A new CPU device object.
       )nbdoc");

  nb::class_<Python::CompletionFlag>(m, "CompletionFlag", R"nbdoc(
          An 8-byte completion flag in pinned host memory mapped into a device's address space.

          Lets a CPU thread signal a GPU queue (or vice versa) by
          writing a 64-bit value to a single location that's visible to
          both. Pair with ``DeviceQueue.wait_for_host_value`` (added in
          a follow-on PR) or the ``mo.wait_host_value`` graph op to gate
          downstream GPU work on a host-produced result without a
          second queue or a blocking host callback.

          Currently requires a CUDA-backed ``Device``; constructing
          against any other backend raises ``RuntimeError``.

          .. code-block:: python

              from max.driver import Accelerator, CompletionFlag

              accel = Accelerator()
              flag = CompletionFlag(accel)
              assert flag.load() == 0  # initialized to zero

              # Subsequent PRs add the producer/consumer methods that
              # actually use the flag's device pointer.
       )nbdoc")
      .def(nb::init<const Python::Device &>(), "device"_a,
           R"nbdoc(
              Allocates a fresh device-mapped pinned u64 bound to ``device``.

              Args:
                  device: A CUDA-backed device. Other backends raise
                      ``RuntimeError``.
            )nbdoc")
      .def_prop_ro("device_ptr", &Python::CompletionFlag::devicePtr,
                   R"nbdoc(
              Device-visible 64-bit address of the 8-byte slot.

              Suitable for passing to graph ops or queue APIs that wait
              on a memory value.
            )nbdoc")
      .def("reset", &Python::CompletionFlag::reset,
           R"nbdoc(
              Clears the flag back to ``0`` with a relaxed atomic store.

              Safe to call before any consumer has observed the address.
            )nbdoc")
      .def("signal", &Python::CompletionFlag::signal, "value"_a,
           R"nbdoc(
              Release-ordered store of ``value`` to the flag.

              Pairs with the GPU-side ``cuStreamWaitValue64`` (or a
              host-side acquire ``load``).

              Primary intended use is priming the flag at setup time so
              the first captured-graph replay's ``mo.wait_host_value``
              passes immediately, before any async kickoff has run.
              Direct Python signalling on the hot path is usually a
              mistake -- prefer the async-host-func trampoline which
              signals from its AsyncRT worker.

              Args:
                  value: The 64-bit value to store.
            )nbdoc")
      .def("load", &Python::CompletionFlag::load,
           R"nbdoc(
              Acquire-ordered load of the current flag value.

              Pairs with a release-ordered store on the producer side.

              Returns:
                  int: Current 64-bit flag value.
            )nbdoc")
      .def_prop_ro(
          "_unsafe_ptr",
          [](const Python::CompletionFlag &self) {
            return reinterpret_cast<std::uintptr_t>(self.ref().getPointer());
          },
          R"nbdoc(
              Raw 64-bit address of the underlying ``M::Driver::CompletionFlag``.

              Intended for packing into a graph-op payload buffer
              (e.g. for ``mo.wait_host_value``); parallels the
              trampoline/user_data pointers returned by
              ``__unsafe_pack_py_host_func`` for ``mo.launch_host_func``.

              The caller must keep this ``CompletionFlag`` Python object
              alive for the duration of any graph execution that
              references the pointer; the underlying allocation is
              freed when the last ref to the wrapper is dropped. The
              leading underscore marks this as an escape hatch with no
              safety net.
            )nbdoc");

  nb::class_<Python::DeviceEvent>(m, "DeviceEvent", R"nbdoc(
          Provides access to an event object.

          An event can be used to wait for the GPU execution to reach a certain
          point on the given queue.

          .. code-block:: python

              from max import driver
              # Create a default accelerator device
              device = driver.Accelerator()
              # Create an event on the device
              event = driver.DeviceEvent(device)
              # Record an event on the device (default queue)
              device.default_queue.record_event(event)
              # Wait for execution on the default queue to reach the event
              event.synchronize()
  )nbdoc")
      .def(nb::init<Python::Device &, bool>(), "device"_a,
           "enable_timing"_a = false, R"nbdoc(
        Creates an event for synchronization on the specified device.

        Args:
            device (Device): The device on which to create the event.
            enable_timing (bool): If True, enable GPU timing on this event.
                Events created with ``enable_timing=True`` can be used with
                :meth:`elapsed_time` to measure GPU execution time.
                Defaults to False.

        Raises:
            ValueError: If event creation failed.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            event = driver.DeviceEvent(device)
            timed_event = driver.DeviceEvent(device, enable_timing=True)
      )nbdoc")
      .def("synchronize", &Python::DeviceEvent::synchronize, R"nbdoc(
          Ensures all operations on this queue complete before returning.

          Raises:
              ValueError: If any enqueued operations had an internal error.
      )nbdoc")
      .def("is_ready", &Python::DeviceEvent::isReady, R"nbdoc(
        Returns whether this event is ready.

        Returns:
            bool: True if the event is complete, otherwise false.

        Raises:
            ValueError: If querying the event status returned an error.
      )nbdoc")
      .def("elapsed_time", &Python::DeviceEvent::elapsedTime, "end_event"_a,
           R"nbdoc(
        Returns the elapsed GPU time in milliseconds between this event and ``end_event``.

        Both events must have been created with ``enable_timing=True``
        and recorded on a queue before calling this method. The end
        event must be synchronized before calling this method.

        Args:
            end_event (DeviceEvent): The ending event.

        Returns:
            float: Elapsed time in milliseconds.

        Raises:
            RuntimeError: If either event was not created with timing
                enabled, or if the events have not been recorded.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            start = driver.DeviceEvent(device, enable_timing=True)
            end = driver.DeviceEvent(device, enable_timing=True)

            queue = device.default_queue
            queue.record_event(start)
            # ... GPU work ...
            queue.record_event(end)
            end.synchronize()

            elapsed_ms = start.elapsed_time(end)
      )nbdoc")
      .def("__str__", &Python::DeviceEvent::getAsString)
      .def("__repr__", &Python::DeviceEvent::getAsString)
      .def(nb::self == nb::self,
           // This gives a mypy warning if we don't replace the type of 'other'
           // with 'object'. See
           // https://nanobind.readthedocs.io/en/latest/typing.html#functions
           nb::sig("def __eq__(self, arg: object, /) -> bool"));
  auto launchTraceEntry = nb::class_<M::Driver::KernelLaunchMetadata>(
      m, "LaunchTraceEntry", R"nbdoc(
          One operation recorded by ``max.driver.begin_launch_trace``.

          Describes a kernel launch, memory copy, or memset enqueued on a
          device stream. Only the field group matching ``kind`` is
          meaningful; the other groups hold zero values. ``stream_index``
          identifies which stream enqueued the operation, so entries from
          different streams can be told apart in the single enqueue-ordered
          list. ``semantic_hash`` is a deterministic hash of the launch
          parameters that excludes memory addresses, so it is stable across
          runs and suitable for change-detection in tests.
  )nbdoc");
  nb::enum_<M::Driver::KernelLaunchMetadata::OperationKind>(
      launchTraceEntry, "OperationKind",
      "The kind of operation a trace entry describes.")
      .value("KERNEL_LAUNCH",
             M::Driver::KernelLaunchMetadata::OperationKind::KernelLaunch)
      .value("MEMCPY", M::Driver::KernelLaunchMetadata::OperationKind::Memcpy)
      .value("MEMSET", M::Driver::KernelLaunchMetadata::OperationKind::Memset);
  nb::enum_<M::Driver::KernelLaunchMetadata::MemcpyKind>(
      launchTraceEntry, "MemcpyKind", "The direction of a memcpy entry.")
      .value("NONE", M::Driver::KernelLaunchMetadata::MemcpyKind::None)
      .value("HTOD", M::Driver::KernelLaunchMetadata::MemcpyKind::HtoD)
      .value("DTOH", M::Driver::KernelLaunchMetadata::MemcpyKind::DtoH)
      .value("DTOD", M::Driver::KernelLaunchMetadata::MemcpyKind::DtoD);
  launchTraceEntry
      .def_ro("kind", &M::Driver::KernelLaunchMetadata::kind,
              "The kind of operation this entry represents.")
      .def_ro("name", &M::Driver::KernelLaunchMetadata::kernelName,
              "The kernel name, or the driver API name for copies/memsets"
              " (e.g. ``cuMemcpyHtoD``).")
      .def_ro("semantic_hash", &M::Driver::KernelLaunchMetadata::semanticHash,
              "Deterministic, address-free hash of the operation parameters.")
      .def_ro("grid_x", &M::Driver::KernelLaunchMetadata::gridX)
      .def_ro("grid_y", &M::Driver::KernelLaunchMetadata::gridY)
      .def_ro("grid_z", &M::Driver::KernelLaunchMetadata::gridZ)
      .def_ro("block_x", &M::Driver::KernelLaunchMetadata::blockX)
      .def_ro("block_y", &M::Driver::KernelLaunchMetadata::blockY)
      .def_ro("block_z", &M::Driver::KernelLaunchMetadata::blockZ)
      .def_ro("shared_mem_bytes",
              &M::Driver::KernelLaunchMetadata::sharedMemBytes)
      .def_ro("stream_index", &M::Driver::KernelLaunchMetadata::streamIndex,
              "Identifies the stream this operation was enqueued on, assigned"
              " in first-seen order within a trace.")
      .def_ro("memcpy_kind", &M::Driver::KernelLaunchMetadata::memcpyKind,
              "The copy direction; ``NONE`` unless ``kind`` is ``MEMCPY``.")
      .def_ro("memcpy_byte_size",
              &M::Driver::KernelLaunchMetadata::memcpyByteSize)
      .def_ro("memset_byte_size",
              &M::Driver::KernelLaunchMetadata::memsetByteSize)
      .def_ro("memset_value", &M::Driver::KernelLaunchMetadata::memsetValue)
      .def_ro("memset_value_size",
              &M::Driver::KernelLaunchMetadata::memsetValueSize)
      .def("__repr__", [](const M::Driver::KernelLaunchMetadata &self) {
        const char *kind = "KERNEL_LAUNCH";
        if (self.kind == M::Driver::KernelLaunchMetadata::OperationKind::Memcpy)
          kind = "MEMCPY";
        else if (self.kind ==
                 M::Driver::KernelLaunchMetadata::OperationKind::Memset)
          kind = "MEMSET";
        return std::string("LaunchTraceEntry(kind=") + kind +
               ", name=" + self.kernelName +
               ", stream_index=" + std::to_string(self.streamIndex) + ")";
      });
  nb::class_<Python::DeviceStream>(m, "DeviceQueue", R"nbdoc(
          Provides access to a queue of execution on a device.

          A queue represents a sequence of operations that will be executed in order.
          Multiple queues on the same device can execute concurrently.

          .. code-block:: python

              from max import driver
              # Create a default accelerator device
              device = driver.Accelerator()
              # Get the default queue for the device
              queue = device.default_queue
              # Create a new queue of execution on the device
              new_queue = driver.DeviceQueue(device)
  )nbdoc")
      .def(nb::new_(&Python::DeviceStream::createStream), "device"_a, R"nbdoc(
          Creates a new queue of execution associated with the device.

          Args:
              device (Device): The device to create the queue on.

          Returns:
              DeviceQueue: A new queue of execution.
      )nbdoc")
      .def("synchronize", &Python::DeviceStream::synchronize, R"nbdoc(
          Ensures all operations on this queue complete before returning.

          Raises:
              ValueError: If any enqueued operations had an internal error.
      )nbdoc")
      .def("record_event",
           static_cast<Python::PyDeviceEventTy (Python::DeviceStream::*)()>(
               &Python::DeviceStream::enqueueEvent),
           R"nbdoc(
          Records an event on this queue.

          Returns:
              DeviceEvent: A new event that will be signaled when all operations
                  submitted to this queue before this call have completed.

          Raises:
              ValueError: If recording the event failed.
      )nbdoc")
      .def("record_event",
           static_cast<void (Python::DeviceStream::*)(
               const Python::DeviceEvent &) const>(
               &Python::DeviceStream::enqueueEvent),
           "event"_a,
           R"nbdoc(
          Records an existing event on this queue.

          Args:
              event (DeviceEvent): The event to record on this queue.

          Raises:
              ValueError: If recording the event failed.
      )nbdoc")
      .def("wait_for", &Python::DeviceStream::waitFor, "stream"_a, R"nbdoc(
          Ensures all operations on the other queue complete before future work submitted to this queue is scheduled.

          Args:
              stream (DeviceQueue): The queue to wait for.
      )nbdoc")
      .def("wait_for", &Python::DeviceStream::waitForDevice, "device"_a,
           R"nbdoc(
          Ensures all operations on device's default queue complete before future work submitted to this queue is scheduled.

          Args:
              device (Device): The device whose default queue to wait for.
      )nbdoc")
      .def("wait_for_host_value", &Python::DeviceStream::waitForHostValue,
           "flag"_a, "value"_a,
           R"nbdoc(
          Stalls the queue until ``flag``'s 64-bit value equals ``value``.

          Wraps the MLRT ``DeviceStream::enqueueWaitOnHostValue`` primitive
          (CUDA's ``cuStreamWaitValue64``). Typically paired with
          ``Device.__unsafe_enqueue_async_py_host_func`` to gate
          downstream GPU work on a host-side AsyncRT task that signals
          ``flag`` when it finishes -- a queue-internal sync that
          avoids a host ``synchronize()`` and captures cleanly into a
          CUDA graph as a wait-value node.

          Args:
              flag (CompletionFlag): The completion flag to wait on.
                  The queue observes ``flag.device_ptr`` via the
                  pinned device-mapped alias.
              value (int): The 64-bit value to wait for (equality).

          Raises:
              RuntimeError: If the underlying device does not support
                  stream memory ops, or if the driver rejects the
                  enqueue.
      )nbdoc")
      .def(
          "__unsafe_enqueue_async_py_host_func",
          [](Python::DeviceStream &self, nb::callable &&fn,
             const Python::CompletionFlag &flag, uint64_t value,
             const Python::CPU &cpu) {
            self.enqueueAsyncHostFunc(std::move(fn), flag, value, cpu);
          },
          "fn"_a, "flag"_a, "value"_a, "cpu"_a,
          R"nbdoc(
          Queue-targeted variant of ``Device.__unsafe_enqueue_async_py_host_func``.

          Enqueues a kickoff host node on **this** queue that dispatches ``fn``
          onto ``cpu``'s AsyncRT worker pool and returns immediately. When ``fn``
          finishes, the worker atomic-stores ``value`` (release ordering) to the
          64-bit memory at ``flag``. Pair with
          ``DeviceQueue.wait_for_host_value(flag, value)`` on a consumer queue
          to gate downstream GPU work.

          Use this overload when you need the host callback to run on a side
          queue concurrently with the model queue's forward pass; the
          ``Device`` overload always targets the default queue and therefore
          serializes against any other default-queue work. As of this
          writing no production caller dispatches via this queue
          overload -- ``StructuredOutputOverlapState.enqueue_async_callback``
          intentionally lands on the device default queue so the
          trampoline's ``flag.reset()`` is naturally ordered against the
          next iter's captured-graph wait. The queue variant is exposed
          as future-facing API and exercised by the GPU integration test
          (``test_structured_output_overlap_gpu.py``).

          Args:
              fn (Callable[[], None]): A zero-argument callable.
              flag (CompletionFlag): The completion flag the worker will
                  signal when ``fn`` returns.
              value (int): The 64-bit value to store on completion.
              cpu (CPU): The CPU device whose AsyncRT worker pool will
                  execute ``fn``.

          Raises:
              RuntimeError: If the underlying device does not support host
                  callbacks, if the supplied ``cpu`` has no associated AsyncRT
                  CPUDevice, or if the driver rejects the enqueue.
      )nbdoc")
      .def_prop_ro("device", &Python::DeviceStream::getDevice,
                   "The device this queue is executing on.")
      .def(
          "_device_context_ptr",
          [](Python::DeviceStream &self) {
            auto *ctx = self.stream->ctx.getPointer();
#if MODULAR_HAL_DEVICE_CONTEXT
            // See Device._device_context_ptr: hand out the boxed Mojo handle.
            return reinterpret_cast<uintptr_t>(ctx->mojoHandle());
#else
            return reinterpret_cast<uintptr_t>(ctx);
#endif
          },
          "Gets the AsyncRT DeviceContext pointer for this specific queue.")
      .def_prop_ro("native_stream_handle",
                   &Python::DeviceStream::nativeStreamHandle, R"nbdoc(
        The native stream handle as an integer, or ``0`` if there is none.

        The handle is the CUDA ``CUstream`` / HIP ``hipStream_t``; ``0`` means
        the stream has no native handle (e.g. a CPU device). Lets native code
        outside MLRT order its own work against this stream -- for example,
        record a CUDA event on it. The handle remains owned by this stream; do
        not destroy it.

        Returns:
            int: The native stream handle, or ``0`` if there is none.
      )nbdoc")
      .def("__str__", &Python::DeviceStream::getAsString)
      .def("__repr__", &Python::DeviceStream::getAsString)
      .def(nb::self == nb::self,
           // This gives a mypy warning if we don't replace the type of 'other'
           // with 'object'. See
           // https://nanobind.readthedocs.io/en/latest/typing.html#functions
           nb::sig("def __eq__(self, arg: object, /) -> bool"));

  m.def("accelerator_count", &Python::acceleratorCount,
        "Returns number of accelerator devices available.");

  m.def(
      "begin_launch_trace",
      []() {
        nb::gil_scoped_release release;
        M::Driver::beginGlobalLaunchTrace();
      },
      R"nbdoc(
      Starts a process-global recording of enqueued device operations.

      Records kernel launches, memory copies, and memsets across **all**
      streams into one enqueue-ordered list, clearing any previous trace.
      No stream or device handle is needed, so work enqueued on streams the
      caller does not hold (e.g. a compiled graph's internal stream) is still
      captured. Only CUDA and HIP devices record entries; on other devices the
      trace is always empty. Intended for tests and debugging: pair with
      ``take_launch_trace`` to assert which device work a code path enqueues
      and on which stream.
  )nbdoc");

  m.def(
      "take_launch_trace",
      []() {
        std::vector<M::Driver::KernelLaunchMetadata> trace;
        {
          nb::gil_scoped_release release;
          trace = M::Driver::takeGlobalLaunchTrace();
        }
        return trace;
      },
      R"nbdoc(
      Stops the global recording and returns the recorded entries.

      Returns:
          list[LaunchTraceEntry]: The operations enqueued since
              ``begin_launch_trace``, in enqueue order across all streams.
              Each entry's ``stream_index`` identifies its stream.
  )nbdoc");

  m.def("__unsafe_pack_py_host_func", &Python::packHostFunc, "fn"_a,
        R"nbdoc(
      Packs a Python callable into a `(trampoline_ptr, user_data_ptr)` pair.

      The returned integers are suitable for passing to
      ``MLRT::DeviceStream::enqueueHostFunc`` (and, by extension, to the
      ``mo.launch_host_func`` custom op). The trampoline is a C ABI
      ``void(void*)`` function; ``user_data_ptr`` owns a heap allocation
      holding the Python callable.

      Ownership transfers to the caller: the user-data pointer MUST be
      consumed by exactly one ``enqueue_host_func`` invocation, which
      causes the trampoline to free it after calling the callable.
      Discarding the pair without enqueueing leaks the callable.

      Args:
          fn (Callable[[], None]): A zero-argument callable.

      Returns:
          tuple[int, int]: ``(trampoline_ptr, user_data_ptr)`` as integers.
      )nbdoc");

  m.def("set_virtual_device_count", &M::AsyncRT::setVirtualDeviceCount,
        "count"_a,
        R"nbdoc(
        Sets the number of virtual devices for device creation.

        When count is greater than 0, Device::create() will return VirtualDevice
        instances instead of real hardware devices for GPU APIs, and
        Device::numberOfDevices() will return this count. This allows creating
        devices for GPU configurations that don't match the current hardware.

        Args:
            count (int): The number of virtual devices. Set to 0 to disable
                virtual device mode.
    )nbdoc");

  m.def("get_virtual_device_count", &M::AsyncRT::getVirtualDeviceCount, R"nbdoc(
        Gets the current virtual device count.

        Returns:
            int: The number of virtual devices, or 0 if virtual device mode
                is disabled.
    )nbdoc");

  m.def("is_virtual_device_mode", &M::AsyncRT::isVirtualDeviceMode, R"nbdoc(
        Checks if virtual device mode is currently enabled.

        Returns:
            bool: True if virtual device mode is enabled (count > 0), False otherwise.
    )nbdoc");

  m.def(
      "enable_all_peer_access",
      []() {
#if MODULAR_HAL_DEVICE_CONTEXT
  // The HAL DeviceContext manages a single device, so there are no
  // peers to enable (mirroring the Mojo-side no-op).
#else
        const char *err = AsyncRT_DeviceContext_enableAllPeerAccess();
        if (err)
          throw std::runtime_error(err);
#endif
      },
      R"nbdoc(
        Enables peer-to-peer memory access between all available GPU pairs.

        This must be called before any collective operations (allreduce,
        broadcast, etc.) that require direct GPU-to-GPU memory access.
        It is safe to call multiple times; the underlying runtime caches
        the result after the first successful enablement.

        Raises:
            RuntimeError: If P2P access cannot be enabled between any GPU pair.
    )nbdoc");

  m.def("set_virtual_device_api", &M::AsyncRT::setVirtualDeviceAPI, "api"_a,
        R"nbdoc(
        Sets the target API for virtual devices in compile-only mode.

        This specifies which GPU API (e.g., "cuda", "hip", "metal") virtual
        devices will use for compilation. Must be called before creating
        virtual devices via set_virtual_device_count().

        Args:
            api (str): The target API string (e.g., "cuda" for NVIDIA,
                "hip" for AMD, "metal" for Apple).
    )nbdoc");

  m.def("get_virtual_device_api", &M::AsyncRT::getVirtualDeviceAPI,
        R"nbdoc(
        Gets the current target API for virtual devices.

        Returns:
            str: The target API string, or empty string if not set.
    )nbdoc");

  m.def("set_virtual_device_target_arch",
        &M::AsyncRT::setVirtualDeviceTargetArch, "arch"_a,
        R"nbdoc(
        Sets the target GPU architecture for virtual devices in compile-only mode.

        This specifies the GPU architecture (e.g., "sm_80", "sm_90") that virtual
        devices will target when compiling code. Must be called before creating
        virtual devices via set_virtual_device_count().

        Args:
            arch (str): The target GPU architecture string (e.g., "sm_80" for
                Ampere/A100, "sm_90" for Hopper/H100).
    )nbdoc");

  m.def("get_virtual_device_target_arch",
        &M::AsyncRT::getVirtualDeviceTargetArch,
        R"nbdoc(
        Gets the current target GPU architecture for virtual devices.

        Returns:
            str: The target GPU architecture string, or empty string if not set.
    )nbdoc");

  m.def("set_virtual_cpu_target", &M::Driver::setVirtualCpuTarget, "cpu"_a,
        R"nbdoc(
        Sets the CPU target for host-independent kernel codegen.

        When set before any CPU kernel compilation (e.g. before importing
        ``max._interpreter_ops``), CPU kernels compile for this fixed target
        instead of the build host's CPU, so the kernel cache can ship to and be
        reused on a different host. Mirrors
        :func:`set_virtual_device_target_arch` for GPUs.

        Args:
            cpu (str): An LLVM target-CPU name (e.g. "x86-64-v3",
                "neoverse-n1"), or "generic" for the most-portable baseline of
                the host arch family ("x86-64" on x86_64, the armv8-a baseline
                on AArch64; other families raise an error). Empty string
                restores host-CPU codegen. "native" is rejected because it
                would re-leak the build host's CPU.
    )nbdoc");

  m.def("get_virtual_cpu_target", &M::Driver::getVirtualCpuTarget,
        R"nbdoc(
        Gets the current virtual CPU target.

        Returns:
            str: The CPU target string, or empty string if not set (host CPU).
    )nbdoc");

  nb::enum_<Python::Usage>(m, "Usage", nb::is_flag(), R"nbdoc(
          Allocation-intent descriptor for :obj:`Buffer`.

          Flags compose with ``|`` and are tested with ``in``. ``max.driver``
          owns the flag set and its per-backend mapping.
      )nbdoc")
      .value("DEFAULT", Python::Usage::Default,
             "The allocation Buffer performs today: device memory for a "
             "non-host device, ordinary host memory for the CPU.")
      .value("STAGING", Python::Usage::Staging,
             "Host memory for staging transfers to and from the given "
             "device. May be page-locked, depending on the backend.")
      .value("UNTRACKED", Python::Usage::Untracked,
             "Memory opting out of host side hazard tracking. Useless unless "
             "paired with STAGING.");

  nb::exception<Python::HostHazardError>(m, "HostHazardError",
                                         PyExc_RuntimeError);

  nb::class_<Python::HostHazardCompletion>(m, "HostHazardCompletion", R"nbdoc(
          A completion shared by host hazard tracked resources.

          Internal to max.driver's hazard tracking. Facilitates ordering of host
          access to resources.

          .. code-block:: python

              import numpy as np

              from max import driver
              from max.dtype import DType

              gpu = driver.Accelerator()
              staging = driver.Buffer(
                  DType.float32, [4], device=gpu, usage=driver.Usage.STAGING
              )
              host = np.zeros(4, dtype=np.float32)
              source = driver.Buffer.from_numpy(host).to(gpu)

              # Record a token on the queue that will signal the copy, then
              # stamp it onto every buffer that submission touched.
              staging.inplace_copy_from(source)

              # The host read waits on that token instead of draining `gpu`.
              value = staging[0].item()
    )nbdoc")
      .def_static("record_on", &Python::HostHazardCompletion::recordOn,
                  "queue"_a, nb::call_guard<nb::gil_scoped_release>(),
                  R"nbdoc(
          Records an event on ``queue`` for the work already submitted to it.

          Args:
              queue (DeviceQueue): The queue whose submitted work to record.

          Returns:
              HostHazardCompletion: A completion that resolves when that work
              finishes.
      )nbdoc")
      .def_static("poisoned", &Python::HostHazardCompletion::poisoned,
                  "queue"_a, "error"_a, R"nbdoc(
          Creates a completion that raises instead of waiting.

          For a producer that failed after enqueuing work, so a later host
          access reports the failure rather than trusting unfinished bytes.

          Args:
              queue (DeviceQueue): The queue the failed work was submitted to.
              error (str): The producer's rendered error.

          Returns:
              HostHazardCompletion: A poisoned completion.
      )nbdoc")
      .def("is_ready", &Python::HostHazardCompletion::isReady, R"nbdoc(
          Returns whether the recorded work has finished.

          A poisoned completion reports ready: waiting on it raises rather
          than blocking, so it can never hold up a caller.

          Returns:
              bool: Whether the work has finished.
      )nbdoc")
      .def("wait", &Python::HostHazardCompletion::wait, R"nbdoc(
          Blocks until the recorded work finishes.

          Raises:
              HostHazardError: If the completion is poisoned, carrying the
                  producer's error.
      )nbdoc")
      .def("poison", &Python::HostHazardCompletion::poison, "message"_a,
           R"nbdoc(
          Marks the completion failed, so every later wait raises.

          Args:
              message (str): The producer's error, reported by the raise.
      )nbdoc");

  nb::class_<Python::Buffer>(m, "Buffer", R"nbdoc(
          Device-resident buffer representation.

          Allocates memory onto a given device with the provided shape and dtype.
          Buffers can be sliced to provide strided views of the underlying memory,
          but any buffers input into model execution must be contiguous.

          Supports numpy-style slicing but does not currently support setting
          items across multiple indices.

          .. code-block:: python

              from max import driver
              from max.dtype import DType

              # Create a buffer on CPU
              cpu_buffer = driver.Buffer(shape=[2, 3], dtype=DType.float32)

              # Create a buffer on GPU
              gpu = driver.Accelerator()
              gpu_buffer = driver.Buffer(shape=[2, 3], dtype=DType.float32, device=gpu)

          Args:
              dtype (DType): Data type of buffer elements.
              shape (Sequence[int]): Tuple of positive, non-zero integers denoting the buffer shape.
              device (Device, optional): Device to allocate buffer onto. Defaults to the CPU.
              stream (DeviceQueue, optional): Queue to associate the buffer with.
              usage (Usage, optional): Allocation intent, see :obj:`Usage`.
                  Defaults to ``Usage.DEFAULT``.
      )nbdoc")
      // Constructors
      .def(nb::init<M::DType, const std::vector<size_t> &,
                    std::optional<Python::PyDeviceTy>, Python::Usage>(),
           "dtype"_a, "shape"_a, "device"_a = nb::none(),
           "usage"_a = Python::Usage::Default)
      .def(nb::init<M::DType, const std::vector<size_t> &,
                    Python::PyDeviceStreamTy, Python::Usage>(),
           "dtype"_a, "shape"_a, "stream"_a, "usage"_a = Python::Usage::Default)
      .def(nb::init<nb::ndarray<nb::ro>, Python::PyDeviceTy>(), "shape"_a,
           "device"_a)
      // Properties
      .def_prop_ro("device", &Python::Buffer::getDevice,
                   "Device on which tensor is resident.")
      .def_prop_ro("stream", &Python::Buffer::getStream,
                   "Stream to which tensor is bound.")
      .def_prop_ro("dtype", &Python::Buffer::getDType,
                   "DType of constituent elements in tensor.")
      .def_prop_ro(
          "element_size",
          [](Python::Buffer &self) { return self.getDType().getSizeInBytes(); },
          "Return the size of the element type in bytes.")
      .def_prop_ro("is_contiguous", &Python::Buffer::isContiguous, R"nbdoc(
            Whether or not buffer is contiguously allocated in memory.

            Returns false if the buffer is a non-contiguous slice.

            Currently, we consider certain situations that are contiguous as
            non-contiguous for the purposes of our engine, such as when a buffer
            has negative steps.
      )nbdoc")
      .def_prop_ro("strides", &Python::Buffer::getStrides, R"nbdoc(
            How far apart consecutive elements are, per dimension.

            Counted in elements, not bytes.

            Slicing does not move any data, so a slice keeps its parent's
            strides. Narrow a 10-wide buffer to 6 columns and it still takes
            10 elements to reach the next row:

            .. code-block:: python

                parent = driver.Buffer(shape=[8, 10], dtype=DType.uint8)
                rows = parent[:, :6]
                rows.shape          # (8, 6)
                rows.strides        # (10, 1) -- still 10 to the next row
                rows.is_contiguous  # False

            A buffer must be contiguous to be passed into model execution.
          )nbdoc")
      .def_prop_ro("is_host", &Python::Buffer::isHost, R"nbdoc(
            Whether or not buffer is host-resident.

            Returns false for GPU buffers, true for CPU buffers.

            .. code-block:: python

                from max import driver
                from max.dtype import DType

                cpu_buffer = driver.Buffer(shape=[2, 3], dtype=DType.bfloat16, device=driver.CPU())

                print(cpu_buffer.is_host)
      )nbdoc")
      .def_prop_ro("num_elements", &Python::Buffer::getNumElements, R"nbdoc(
            Returns the number of elements in this buffer.

            Rank-0 buffers have 1 element by convention.
      )nbdoc")
      .def_prop_ro("rank", &Python::Buffer::getRank, "Buffer rank.")
      .def_prop_ro("shape", &Python::Buffer::getShape, "Shape of buffer.")
      // Methods (including statics)
      .def(
          "contiguous", [](Python::Buffer &) { /* Replaced in Python */ },
          nb::sig("def contiguous(self) -> Buffer"),
          "Creates a contiguous copy of the buffer.")
      .def(
          "copy",
          [](Python::Buffer &self, Python::PyDeviceStreamTy stream) {
            return self.copyTo(std::move(stream));
          },
          "stream"_a,
          R"nbdoc(
        Creates a deep copy on the device associated with the queue.

        Args:
            stream (DeviceQueue): The queue to associate the new buffer with.

        Returns:
            Buffer: A new buffer that is a copy of this buffer.
      )nbdoc")
      .def(
          "copy",
          [](Python::Buffer &self,
             std::optional<Python::PyDeviceTy> maybeDevice) {
            return self.copyTo(maybeDevice ? std::move(maybeDevice.value())
                                           : self.getDevice());
          },
          "device"_a = nb::none(),
          R"nbdoc(
        Creates a deep copy on an optionally given device.

        If device is None (default), a copy is created on the same device.

        .. code-block:: python

            from max import driver
            from max.dtype import DType

            cpu_buffer = driver.Buffer(shape=[2, 3], dtype=DType.bfloat16, device=driver.CPU())
            cpu_copy = cpu_buffer.copy()

            # Copy to GPU
            gpu = driver.Accelerator()
            gpu_copy = cpu_buffer.copy(device=gpu)

        Args:
            device (Device, optional): The device to create the copy on.
                Defaults to None (same device).

        Returns:
            Buffer: A new buffer that is a copy of this buffer.
      )nbdoc")
      .def_static(
          "mmap", []() { /* Replaced in Python */ },
          nb::sig("def mmap("
                  "filename: os.PathLike, "
                  "dtype: max._core.dtype.DType, "
                  "shape: Sequence[int], "
                  "mode: numpy._MemMapModeKind = \"copyonwrite\", "
                  "offset: int = 0) -> Buffer"),
          R"nbdoc(
          Creates a memory-mapped buffer from a binary file on disk.

          The constructor argument semantics follow that of np.memmap.
      )nbdoc")
      .def(
          "inplace_copy_from",
          [](Python::Buffer &) { /* Replaced in Python */ },
          nb::sig("def inplace_copy_from(self, src: Buffer) -> None"),
          R"nbdoc(
          Copies the contents of another buffer into this one.

          These buffers may be on different devices.
          Requires that both buffers are contiguous and have same size.
      )nbdoc")
      .def_static(
          "from_dlpack", []() { /* Replaced in Python */ },
          nb::sig("def from_dlpack(array: Any, *, copy: bool | None = None) -> "
                  "Buffer"),
          R"nbdoc(
        Creates a buffer from an object implementing the dlpack protocol.

        This usually does not result in a copy, and the producer of the object
        retains ownership of the underlying memory.

        Args:
            array (Any): An object that implements the dlpack protocol.
            copy (bool, optional): Whether to create a copy of the data.
                Defaults to None.

        Returns:
            Buffer: A new buffer that views or copies the dlpack data.
      )nbdoc")
      .def_static(
          "from_numpy", []() { /* Replaced in Python */ },
          nb::sig("def from_numpy(arr: numpy.ndarray) -> Buffer"),
          R"nbdoc(
        Creates a buffer from a provided numpy array on the host device.

        The underlying data is not copied unless the array is noncontiguous. If
        it is, a contiguous copy will be returned.

        Args:
            arr (numpy.ndarray): The numpy array to convert.

        Returns:
            Buffer: A new buffer that views or copies the numpy array data.
      )nbdoc")
      .def("item", &Python::Buffer::item, R"nbdoc(
        Returns the scalar value at a given location.

        Currently implemented only for zero-rank buffers. The return type is
        converted to a Python built-in type.
      )nbdoc")
      .def_static(
          "scalar",
          [](nb::any value, M::DType dtype,
             std::optional<Python::PyDeviceTy> device) {
            auto cpu = Python::getCpuDevice(-1);
            Python::Buffer t{dtype, {}, cpu};
            t.set(nb::cast(0), value);
            if (device && device.value() != cpu) {
              // We can't directly set GPU memory, so
              // we just have to copy the tensor over.
              return t.copyTo(std::move(device.value()));
            } else {
              return t;
            }
          },
          "value"_a, "dtype"_a, "device"_a = nb::none(), R"nbdoc(
        Creates a scalar value of a given dtype and value.

        If device is None (default), the buffer will be allocated on the CPU.
      )nbdoc")
      .def(
          "to",
          [](nb::pointer_and_handle<Python::Buffer> self,
             Python::PyDeviceTy device) {
            return tensorTo(self, self.p->getStream()->stream, device);
          },
          "device"_a,
          R"nbdoc(
        Returns a buffer that's guaranteed to be on the given device.

        The buffer is only copied if the requested device is different from the
        device upon which the buffer is already resident.
      )nbdoc")
      .def(
          "to",
          [](nb::pointer_and_handle<Python::Buffer> self,
             Python::PyDeviceStreamTy stream)
              -> nb::typed<nb::object, Python::Buffer> {
            // Variation of tensorTo() that avoids the copy if the current and
            // target devices are the same. After all, this is a
            // stream-to-stream transfer.
            Python::PyDeviceTy &currentDevice =
                self.p->getStream()->stream; // yeah, I know!
            Python::PyDeviceTy &targetDevice = stream->stream;
            M::Driver::DeviceContext const &srcCtx = *currentDevice->ctx;
            M::Driver::DeviceContext const &dstCtx = *targetDevice->ctx;
            if (srcCtx == dstCtx) {
              // Same contexts: noop
              return nb::borrow(self.h);
            } else if (*(self.p->getDevice()) == *(stream->getDevice())) {
              // Different contexts on same device.
              // Transfer ownership with proper synchronization.
              // Applies to pinned buffers too.
              return nb::cast(self.p->to(stream));
            } else {
              // Different devices. Copy.
              return nb::cast(self.p->to(targetDevice));
            }
          },
          "stream"_a,
          R"nbdoc(
        Returns a buffer that's guaranteed to be on the given device and associated with the given queue.

        The buffer is only copied if the requested device is different from the
        device upon which the buffer is already resident. If the destination
        queue is on the same device, then a new reference to the same buffer is
        returned.
      )nbdoc")
      .def(
          "to",
          [](nb::pointer_and_handle<Python::Buffer> self,
             std::vector<Python::PyDeviceTy> devices) {
            std::vector<nb::typed<nb::object, Python::Buffer>> tensors;
            for (Python::PyDeviceTy &device : devices) {
              tensors.emplace_back(
                  tensorTo(self, self.p->getStream()->stream, device));
            }
            return tensors;
          },
          "devices"_a,
          R"nbdoc(
            Returns a list of buffers that are guaranteed to be on the given devices.

            The buffers are only copied if the requested devices are different from the
            device upon which the buffer is already resident.
          )nbdoc")
      .def(
          "to",
          [](nb::pointer_and_handle<Python::Buffer> &self,
             std::vector<Python::PyDeviceStreamTy> streams) {
            std::vector<nb::typed<nb::object, Python::Buffer>> tensors;
            for (Python::PyDeviceStreamTy &stream : streams) {
              tensors.emplace_back(
                  tensorTo(self, self.p->getStream()->stream, stream->stream));
            }
            return tensors;
          },
          "streams"_a,
          R"nbdoc(
            Returns a list of buffers that are guaranteed to be on the given queues.

            The buffers are only copied if the requested queues are different from the
            queue upon which the buffer is already resident.
          )nbdoc")
      .def(
          "to_numpy", [](Python::Buffer &) { /* Replaced in Python */ },
          nb::sig("def to_numpy(self) -> numpy.ndarray"),
          R"nbdoc(
        Converts the buffer to a numpy array.

        If the buffer is on the host (CPU), the numpy array aliases the existing memory.
        Otherwise, it is copied to the host device.

        Returns:
            numpy.ndarray: A numpy array containing the buffer data.
      )nbdoc")
      .def_prop_ro(
          "pinned", &Python::Buffer::isPinned,
          "Whether the allocation landed in the device's host memory space. "
          "Ask ``usage`` for what was requested.")
      .def_prop_ro(
          "usage", &Python::Buffer::getUsage,
          "Allocation intent. Slices and views report their parent's usage.")
      .def_prop_ro(
          "_host_hazard_tracked", &Python::Buffer::isHostHazardTracked,
          "Whether the allocation is instrumented for host hazard tracking.")
      .def(
          "view", [](Python::Buffer &) { /* Replaced in Python */ },
          nb::sig("def view(self, dtype: max._core.dtype.DType, shape: "
                  "Sequence[int] | None = None) -> Buffer"),
          R"nbdoc(
        Returns a new buffer with the given type and shape that shares the underlying memory.

        If the shape is not given, it will be deduced if possible, or a
        ValueError is raised.
      )nbdoc")
      .def_static(
          "zeros",
          [](const std::vector<size_t> &shape, M::DType dtype,
             std::optional<Python::PyDeviceTy> device, Python::Usage usage) {
            Python::Buffer t{dtype, shape, std::move(device), usage};
            t.zeros();
            return t;
          },
          "shape"_a, "dtype"_a, "device"_a = nb::none(),
          "usage"_a = Python::Usage::Default,
          R"nbdoc(
        Allocates a buffer with all elements initialized to zero.

        Args:
            shape (Sequence[int]): The shape of the buffer.
            dtype (DType): The data type of the buffer.
            device (Device, optional): The device to allocate the buffer on.
                Defaults to None (CPU).
            usage (Usage, optional): Allocation intent, see :obj:`Usage`.
                Defaults to ``Usage.DEFAULT``.

        Returns:
            Buffer: A new buffer filled with zeros.
      )nbdoc")
      // Dunders
      .def("__dlpack__", &Python::Buffer::dlPack, nb::kw_only(),
           "stream"_a = nb::none(), "kwargs"_a,
           "Implements part of the dlpack contract.")
      .def("__dlpack_device__", &Python::Buffer::dlPackDevice,
           "Implements part of the dlpack contract.")
#define ShapeType "int | slice | Sequence[int | slice]"
      .def("__getitem__", &Python::Buffer::get, "index"_a, R"nbdoc(
          Gets a buffer slice.

          Supports full numpy-style slicing. Invocations
          using only integer-based indexes will return zero-rank buffers.
      )nbdoc",
           nb::sig("def __getitem__(self, idx: " ShapeType ") -> Buffer"))
      .def("__setitem__", &Python::Buffer::set, "index"_a, "value"_a,
           "Sets an item in the buffer.",
           nb::sig("def __setitem__(self, idx: " ShapeType
                   ", value: Any) -> None"))
#undef ShapeType
      // Private methods
      .def(
          "_aligned",
          [](Python::Buffer &self, std::optional<int> maybe_align) {
            const auto alignment =
                maybe_align ? *maybe_align : self.getDType().getAlignment();
            return self.aligned(alignment);
          },
          "alignment"_a = nb::none(),
          "Returns whether the buffer is aligned to the desired alignment.")
      .def_static("_from_dlpack", &Python::Buffer::fromDLPack)
      .def_static("_from_dlpack",
                  [](nb::capsule dlpack, Python::PyDeviceTy device,
                     nb::int_ externalStream) {
                    if (!PyCapsule_IsValid(dlpack.ptr(), "dltensor")) {
                      throw std::runtime_error(
                          "malformed capsule in dlpack implementation");
                    }
                    PyCapsule_SetName(dlpack.ptr(), "used_dltensor");
                    return Python::Buffer::fromDLManagedTensor(
                        DLPack::DLManagedTensorWrapper{
                            static_cast<DLManagedTensor *>(dlpack.data())},
                        device, nb::cast<intptr_t>(externalStream));
                  })
      .def_static("_iterate_indices",
                  [](Python::Buffer &) { /* Replaced in Python*/ })
      .def("_view", &Python::Buffer::view, "dtype"_a, "shape"_a)
      .def("_inplace_copy_from", &Python::Buffer::inplaceCopyFrom, "src"_a)
      .def(
          "_data_ptr",
          [](Python::Buffer &self) {
            return reinterpret_cast<uintptr_t>(self.data());
          },
          "Gets the memory address of the buffer data. Internal use only.")
      .def("_stamp_read", &Python::Buffer::stampRead, "completion"_a, R"nbdoc(
          Add a read completion to the buffer. Internal use only.
      )nbdoc")
      .def("_stamp_write", &Python::Buffer::stampWrite, "completion"_a, R"nbdoc(
          Add a write completion to the buffer. Internal use only.
      )nbdoc");

  m.def(
      "_batch_inplace_copy",
      [](nb::sequence dsts, nb::sequence srcs) {
        std::vector<Python::Buffer *> dstPtrs, srcPtrs;
        for (nb::handle h : dsts)
          dstPtrs.push_back(nb::cast<Python::Buffer *>(h));
        for (nb::handle h : srcs)
          srcPtrs.push_back(nb::cast<Python::Buffer *>(h));
        Python::Buffer::batchInplaceCopy(dstPtrs, srcPtrs);
      },
      "dsts"_a, "srcs"_a,
      nb::sig("def _batch_inplace_copy(dsts: Sequence[Buffer], srcs: "
              "Sequence[Buffer]) -> None"),
      R"nbdoc(
      Batched copy of ``srcs`` into ``dsts``.

      Sources may be host, pinned, same-device or peer memory in any mix. One
      submission only orders the writes on its own stream, so destinations are
      grouped by device and submitted one batch per device. Identical pairs
      (``dst is src``) are skipped. All buffers must have matching sizes.
  )nbdoc");

  nb::class_<Python::ChunkedStagingRegion>(m, "_ChunkedStagingRegion",
                                           R"nbdoc(
      A host mapping made device-accessible in chunks.

      Page-locking a multi-hundred-GiB range in one call is slow, and above
      1 TiB it fails outright. This maps the range once, faults it in across
      parallel workers, and registers it in chunks while they run ahead.

      Not a :obj:`Buffer`: a ``Buffer`` promises a range the device addresses
      in a single transfer, and a transfer crossing a chunk boundary here
      fails. Callers work from :obj:`address` and keep each transfer inside
      one ``row_bytes`` row, which no chunk boundary splits.

      Releases on garbage collection: drains the device, unregisters every
      chunk, then unmaps.

      Internal to ``max``; no compatibility guarantee.

      Args:
          byte_size (int): Total bytes to map. Must be a positive multiple of ``row_bytes``.
          row_bytes (int): Row width no chunk boundary may fall inside.
          device (Device): GPU/Accelerator the range is made accessible to. Must not be CPU.
          threads (int, optional): Parallel page-touch workers. Defaults to 16.
          chunk_bytes (int, optional): Registration granularity, rounded up to whole rows and pages. Defaults to 512 MiB.

      Raises:
          ValueError: If ``device`` is a CPU device, any size is zero, or ``byte_size`` is not a multiple of ``row_bytes``.
  )nbdoc")
      .def(nb::init<size_t, size_t, Python::PyDeviceTy, unsigned, size_t>(),
           "byte_size"_a, "row_bytes"_a, "device"_a,
           "threads"_a = Python::kDefaultTouchThreads,
           "chunk_bytes"_a = Python::kDefaultChunkBytes)
      .def_prop_ro("address", &Python::ChunkedStagingRegion::address,
                   "Base host address, valid until this is released.")
      .def_prop_ro("byte_size", &Python::ChunkedStagingRegion::byteSize)
      .def_prop_ro("row_bytes", &Python::ChunkedStagingRegion::rowBytes)
      .def_prop_ro("chunk_bytes", &Python::ChunkedStagingRegion::chunkBytes,
                   "The requested ``chunk_bytes`` rounded up to whole rows "
                   "and pages.")
      .def_prop_ro("num_chunks", &Python::ChunkedStagingRegion::numChunks)
      .def_prop_ro("device", &Python::ChunkedStagingRegion::getDevice);

  m.def("_release_buffers_to_borrowed", &Python::Buffer::releaseAllToBorrowed,
        "buffers"_a,
        "Convert owning buffers into borrowed wrappers over the same storage.");

  M::Profiler::Python::addTracingToNanobindModuleClassMethods(
      m, "max._core.driver.");
}
