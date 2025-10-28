# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from .bencher import Bench, Bencher, BenchId, ThroughputMeasure


struct QuickBench:
    """Defines a struct to facilitate benchmarking and avoiding `Bencher` boilerplate.
    """

    var m: Bench
    """Bench object to collect the results."""

    @always_inline
    fn __init__(out self) raises:
        """Initializes the `Bench` object.

        Raises:
            If the operation fails.
        """
        self.m = Bench()

    @always_inline
    fn dump_report(mut self) raises:
        """Prints out the report from a Benchmark execution collected in Bench object.

        Raises:
            If the operation fails.
        """
        self.m.dump_report()

    @always_inline
    fn run[
        T_out: AnyTrivialRegType
    ](
        mut self,
        func: fn () -> T_out,
        *,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """
        Benchmark function `func` with no input arguments and return type `T_out`.

        Parameters:
            T_out: Output type of func.

        Args:
            func: The function to be benchmarked (run in benchmark iterations).
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.

        Raises:
            If the operation fails.
        """

        @parameter
        @always_inline
        fn bench_iter(mut b: Bencher):
            @parameter
            @always_inline
            fn call_func():
                var x = func()
                keep(x)

            b.iter[call_func]()

        self.m.bench_function[bench_iter](bench_id, measures=measures)

    @always_inline
    fn run[
        T0: AnyTrivialRegType, /, T_out: AnyTrivialRegType
    ](
        mut self,
        func: fn (T0) -> T_out,
        x0: T0,
        *,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """
        Benchmark function `func` with 1 input argument and return type `T_out`.

        Parameters:
            T0: Type of the 1st argument of func.
            T_out: Output type of func.

        Args:
            func: The function to be benchmarked (run in benchmark iterations).
            x0: The 1st argument of func.
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.

        Raises:
            If the operation fails.
        """

        @parameter
        @always_inline
        fn bench_iter(mut b: Bencher):
            @parameter
            @always_inline
            fn call_func():
                var x = func(x0)
                keep(x)

            b.iter[call_func]()

        self.m.bench_function[bench_iter](bench_id, measures=measures)

    @always_inline
    fn run[
        T0: AnyTrivialRegType,
        T1: AnyTrivialRegType,
        /,
        T_out: AnyTrivialRegType,
    ](
        mut self,
        func: fn (T0, T1) -> T_out,
        x0: T0,
        x1: T1,
        *,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """
        Benchmark function `func` with 2 input argument and return type `T_out`.

        Parameters:
            T0: Type of the 1st argument of func.
            T1: Type of the 2nd argument of func.
            T_out: Output type of func.

        Args:
            func: The function to be benchmarked (run in benchmark iterations).
            x0: The 1st argument of func.
            x1: The 2nd argument of func.
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.

        Raises:
            If the operation fails.
        """

        @parameter
        @always_inline
        fn bench_iter(mut b: Bencher):
            @parameter
            @always_inline
            fn call_func():
                var x = func(x0, x1)
                keep(x)

            b.iter[call_func]()

        self.m.bench_function[bench_iter](bench_id, measures=measures)

    @always_inline
    fn run[
        T0: AnyTrivialRegType,
        T1: AnyTrivialRegType,
        T2: AnyTrivialRegType,
        /,
        T_out: AnyTrivialRegType,
    ](
        mut self,
        func: fn (T0, T1, T2) -> T_out,
        x0: T0,
        x1: T1,
        x2: T2,
        *,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """
        Benchmark function `func` with 3 input argument and return type `T_out`.

        Parameters:
            T0: Type of the 1st argument of func.
            T1: Type of the 2nd argument of func.
            T2: Type of the 3rd argument of func.
            T_out: Output type of func.

        Args:
            func: The function to be benchmarked (run in benchmark iterations).
            x0: The 1st argument of func.
            x1: The 2nd argument of func.
            x2: The 3rd argument of func.
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.

        Raises:
            If the operation fails.
        """

        @parameter
        @always_inline
        fn bench_iter(mut b: Bencher):
            @parameter
            @always_inline
            fn call_func():
                var x = func(x0, x1, x2)
                keep(x)

            b.iter[call_func]()

        self.m.bench_function[bench_iter](bench_id, measures=measures)

    @always_inline
    fn run[
        T0: AnyTrivialRegType,
        T1: AnyTrivialRegType,
        T2: AnyTrivialRegType,
        T3: AnyTrivialRegType,
        /,
        T_out: AnyTrivialRegType,
    ](
        mut self,
        func: fn (T0, T1, T2, T3) -> T_out,
        x0: T0,
        x1: T1,
        x2: T2,
        x3: T3,
        *,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """
        Benchmark function `func` with 4 input argument and return type `T_out`.

        Parameters:
            T0: Type of the 1st argument of func.
            T1: Type of the 2nd argument of func.
            T2: Type of the 3rd argument of func.
            T3: Type of the 4th argument of func.
            T_out: Output type of func.

        Args:
            func: The function to be benchmarked (run in benchmark iterations).
            x0: The 1st argument of func.
            x1: The 2nd argument of func.
            x2: The 3rd argument of func.
            x3: The 4th argument of func.
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.

        Raises:
            If the operation fails.
        """

        @parameter
        @always_inline
        fn bench_iter(mut b: Bencher):
            @parameter
            @always_inline
            fn call_func():
                var x = func(x0, x1, x2, x3)
                keep(x)

            b.iter[call_func]()

        self.m.bench_function[bench_iter](bench_id, measures=measures)

    @always_inline
    fn run[
        T0: AnyTrivialRegType,
        T1: AnyTrivialRegType,
        T2: AnyTrivialRegType,
        T3: AnyTrivialRegType,
        T4: AnyTrivialRegType,
        /,
        T_out: AnyTrivialRegType,
    ](
        mut self,
        func: fn (T0, T1, T2, T3, T4) -> T_out,
        x0: T0,
        x1: T1,
        x2: T2,
        x3: T3,
        x4: T4,
        *,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """
        Benchmark function `func` with 5 input argument and return type `T_out`.

        Parameters:
            T0: Type of the 1st argument of func.
            T1: Type of the 2nd argument of func.
            T2: Type of the 3rd argument of func.
            T3: Type of the 4th argument of func.
            T4: Type of the 5th argument of func.
            T_out: Output type of func.

        Args:
            func: The function to be benchmarked (run in benchmark iterations).
            x0: The 1st argument of func.
            x1: The 2nd argument of func.
            x2: The 3rd argument of func.
            x3: The 4th argument of func.
            x4: The 5th argument of func.
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.

        Raises:
            If the operation fails.
        """

        @parameter
        @always_inline
        fn bench_iter(mut b: Bencher):
            @parameter
            @always_inline
            fn call_func():
                var x = func(x0, x1, x2, x3, x4)
                keep(x)

            b.iter[call_func]()

        self.m.bench_function[bench_iter](bench_id, measures=measures)

    @always_inline
    fn run[
        T0: AnyTrivialRegType,
        T1: AnyTrivialRegType,
        T2: AnyTrivialRegType,
        T3: AnyTrivialRegType,
        T4: AnyTrivialRegType,
        T5: AnyTrivialRegType,
        /,
        T_out: AnyTrivialRegType,
    ](
        mut self,
        func: fn (T0, T1, T2, T3, T4, T5) -> T_out,
        x0: T0,
        x1: T1,
        x2: T2,
        x3: T3,
        x4: T4,
        x5: T5,
        *,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """
        Benchmark function `func` with 6 input argument and return type `T_out`.

        Parameters:
            T0: Type of the 1st argument of func.
            T1: Type of the 2nd argument of func.
            T2: Type of the 3rd argument of func.
            T3: Type of the 4th argument of func.
            T4: Type of the 5th argument of func.
            T5: Type of the 6th argument of func.
            T_out: Output type of func.

        Args:
            func: The function to be benchmarked (run in benchmark iterations).
            x0: The 1st argument of func.
            x1: The 2nd argument of func.
            x2: The 3rd argument of func.
            x3: The 4th argument of func.
            x4: The 5th argument of func.
            x5: The 6th argument of func.
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.

        Raises:
            If the operation fails.
        """

        @parameter
        @always_inline
        fn bench_iter(mut b: Bencher):
            @parameter
            @always_inline
            fn call_func():
                var x = func(x0, x1, x2, x3, x4, x5)
                keep(x)

            b.iter[call_func]()

        self.m.bench_function[bench_iter](bench_id, measures=measures)

    @always_inline
    fn run[
        T0: AnyTrivialRegType,
        T1: AnyTrivialRegType,
        T2: AnyTrivialRegType,
        T3: AnyTrivialRegType,
        T4: AnyTrivialRegType,
        T5: AnyTrivialRegType,
        T6: AnyTrivialRegType,
        /,
        T_out: AnyTrivialRegType,
    ](
        mut self,
        func: fn (T0, T1, T2, T3, T4, T5, T6) -> T_out,
        x0: T0,
        x1: T1,
        x2: T2,
        x3: T3,
        x4: T4,
        x5: T5,
        x6: T6,
        *,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """
        Benchmark function `func` with 7 input argument and return type `T_out`.

        Parameters:
            T0: Type of the 1st argument of func.
            T1: Type of the 2nd argument of func.
            T2: Type of the 3rd argument of func.
            T3: Type of the 4th argument of func.
            T4: Type of the 5th argument of func.
            T5: Type of the 6th argument of func.
            T6: Type of the 7th argument of func.
            T_out: Output type of func.

        Args:
            func: The function to be benchmarked (run in benchmark iterations).
            x0: The 1st argument of func.
            x1: The 2nd argument of func.
            x2: The 3rd argument of func.
            x3: The 4th argument of func.
            x4: The 5th argument of func.
            x5: The 6th argument of func.
            x6: The 7th argument of func.
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.

        Raises:
            If the operation fails.
        """

        @parameter
        @always_inline
        fn bench_iter(mut b: Bencher):
            @parameter
            @always_inline
            fn call_func():
                var x = func(x0, x1, x2, x3, x4, x5, x6)
                keep(x)

            b.iter[call_func]()

        self.m.bench_function[bench_iter](bench_id, measures=measures)

    @always_inline
    fn run[
        T0: AnyTrivialRegType,
        T1: AnyTrivialRegType,
        T2: AnyTrivialRegType,
        T3: AnyTrivialRegType,
        T4: AnyTrivialRegType,
        T5: AnyTrivialRegType,
        T6: AnyTrivialRegType,
        T7: AnyTrivialRegType,
        /,
        T_out: AnyTrivialRegType,
    ](
        mut self,
        func: fn (T0, T1, T2, T3, T4, T5, T6, T7) -> T_out,
        x0: T0,
        x1: T1,
        x2: T2,
        x3: T3,
        x4: T4,
        x5: T5,
        x6: T6,
        x7: T7,
        *,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """
        Benchmark function `func` with 8 input argument and return type `T_out`.

        Parameters:
            T0: Type of the 1st argument of func.
            T1: Type of the 2nd argument of func.
            T2: Type of the 3rd argument of func.
            T3: Type of the 4th argument of func.
            T4: Type of the 5th argument of func.
            T5: Type of the 6th argument of func.
            T6: Type of the 7th argument of func.
            T7: Type of the 8th argument of func.
            T_out: Output type of func.

        Args:
            func: The function to be benchmarked (run in benchmark iterations).
            x0: The 1st argument of func.
            x1: The 2nd argument of func.
            x2: The 3rd argument of func.
            x3: The 4th argument of func.
            x4: The 5th argument of func.
            x5: The 6th argument of func.
            x6: The 7th argument of func.
            x7: The 8th argument of func.
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.

        Raises:
            If the operation fails.
        """

        @parameter
        @always_inline
        fn bench_iter(mut b: Bencher):
            @parameter
            @always_inline
            fn call_func():
                var x = func(x0, x1, x2, x3, x4, x5, x6, x7)
                keep(x)

            b.iter[call_func]()

        self.m.bench_function[bench_iter](bench_id, measures=measures)

    @always_inline
    fn run[
        T0: AnyTrivialRegType,
        T1: AnyTrivialRegType,
        T2: AnyTrivialRegType,
        T3: AnyTrivialRegType,
        T4: AnyTrivialRegType,
        T5: AnyTrivialRegType,
        T6: AnyTrivialRegType,
        T7: AnyTrivialRegType,
        T8: AnyTrivialRegType,
        /,
        T_out: AnyTrivialRegType,
    ](
        mut self,
        func: fn (T0, T1, T2, T3, T4, T5, T6, T7, T8) -> T_out,
        x0: T0,
        x1: T1,
        x2: T2,
        x3: T3,
        x4: T4,
        x5: T5,
        x6: T6,
        x7: T7,
        x8: T8,
        *,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """
        Benchmark function `func` with 9 input argument and return type `T_out`.

        Parameters:
            T0: Type of the 1st argument of func.
            T1: Type of the 2nd argument of func.
            T2: Type of the 3rd argument of func.
            T3: Type of the 4th argument of func.
            T4: Type of the 5th argument of func.
            T5: Type of the 6th argument of func.
            T6: Type of the 7th argument of func.
            T7: Type of the 8th argument of func.
            T8: Type of the 9th argument of func.
            T_out: Output type of func.

        Args:
            func: The function to be benchmarked (run in benchmark iterations).
            x0: The 1st argument of func.
            x1: The 2nd argument of func.
            x2: The 3rd argument of func.
            x3: The 4th argument of func.
            x4: The 5th argument of func.
            x5: The 6th argument of func.
            x6: The 7th argument of func.
            x7: The 8th argument of func.
            x8: The 9th argument of func.
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.

        Raises:
            If the operation fails.
        """

        @parameter
        @always_inline
        fn bench_iter(mut b: Bencher):
            @parameter
            @always_inline
            fn call_func():
                var x = func(x0, x1, x2, x3, x4, x5, x6, x7, x8)
                keep(x)

            b.iter[call_func]()

        self.m.bench_function[bench_iter](bench_id, measures=measures)

    @always_inline
    fn run[
        T0: AnyTrivialRegType,
        T1: AnyTrivialRegType,
        T2: AnyTrivialRegType,
        T3: AnyTrivialRegType,
        T4: AnyTrivialRegType,
        T5: AnyTrivialRegType,
        T6: AnyTrivialRegType,
        T7: AnyTrivialRegType,
        T8: AnyTrivialRegType,
        T9: AnyTrivialRegType,
        /,
        T_out: AnyTrivialRegType,
    ](
        mut self,
        func: fn (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9) -> T_out,
        x0: T0,
        x1: T1,
        x2: T2,
        x3: T3,
        x4: T4,
        x5: T5,
        x6: T6,
        x7: T7,
        x8: T8,
        x9: T9,
        *,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """
        Benchmark function `func` with 10 input argument and return type `T_out`.

        Parameters:
            T0: Type of the 1st argument of func.
            T1: Type of the 2nd argument of func.
            T2: Type of the 3rd argument of func.
            T3: Type of the 4th argument of func.
            T4: Type of the 5th argument of func.
            T5: Type of the 6th argument of func.
            T6: Type of the 7th argument of func.
            T7: Type of the 8th argument of func.
            T8: Type of the 9th argument of func.
            T9: Type of the 10th argument of func.
            T_out: Output type of func.

        Args:
            func: The function to be benchmarked (run in benchmark iterations).
            x0: The 1st argument of func.
            x1: The 2nd argument of func.
            x2: The 3rd argument of func.
            x3: The 4th argument of func.
            x4: The 5th argument of func.
            x5: The 6th argument of func.
            x6: The 7th argument of func.
            x7: The 8th argument of func.
            x8: The 9th argument of func.
            x9: The 10th argument of func.
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.

        Raises:
            If the operation fails.
        """

        @parameter
        @always_inline
        fn bench_iter(mut b: Bencher):
            @parameter
            @always_inline
            fn call_func():
                var x = func(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)
                keep(x)

            b.iter[call_func]()

        self.m.bench_function[bench_iter](bench_id, measures=measures)
