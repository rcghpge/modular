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


trait Iterable:
    """The `Iterator` trait describes a type that can be turned into an
    iterator.
    """

    alias IteratorType[mut: Bool, //, origin: Origin[mut]]: Iterator

    fn __iter__(ref self) -> Self.IteratorType[__origin_of(self)]:
        ...


trait Iterator(ExplicitlyCopyable, Movable):
    """The `Iterator` trait describes a type that can be used as an
    iterator, e.g. in a `for` loop.
    """

    alias Element: ExplicitlyCopyable & Movable

    fn __has_next__(self) -> Bool:
        ...

    fn __next__(mut self) -> Self.Element:
        ...


@always_inline
fn iter[
    IterableType: Iterable
](ref iterable: IterableType) -> IterableType.IteratorType[
    __origin_of(iterable)
]:
    return iterable.__iter__()


@always_inline
fn next[
    IteratorType: Iterator
](mut iterator: IteratorType) -> IteratorType.Element:
    return iterator.__next__()


struct _Enumerate[InnerIteratorType: Iterator](
    ExplicitlyCopyable, Iterable, Iterator, Movable
):
    """The `enumerate` function returns an iterator that yields tuples of the
    index and the element of the original iterator.
    """

    alias Element = Tuple[Int, InnerIteratorType.Element]
    alias IteratorType[mut: Bool, //, origin: Origin[mut]]: Iterator = Self
    var _inner: InnerIteratorType
    var _count: Int

    fn __iter__(ref self) -> Self.IteratorType[__origin_of(self)]:
        return self.copy()

    fn __init__(out self, var iterator: InnerIteratorType, *, start: Int = 0):
        self._inner = iterator^
        self._count = start

    fn __has_next__(self) -> Bool:
        return self._inner.__has_next__()

    fn __next__(mut self) -> Self.Element:
        var count = self._count
        self._count += 1
        return count, next(self._inner)


@always_inline
fn enumerate[
    IterableType: Iterable
](ref iterable: IterableType, *, start: Int = 0) -> _Enumerate[
    IterableType.IteratorType[__origin_of(iterable)]
]:
    """The `enumerate` function returns an iterator that yields tuples of the
    index and the element of the original iterator.

    Args:
        iterable: An iterable object (e.g., list, string, etc.).
        start: The starting index for enumeration (default is 0).

    Example:

    ```mojo
    var l = ["hey", "hi", "hello"]
    for i, elem in enumerate(l):
        print(i, elem)
    ```
    """
    return _Enumerate(iter(iterable), start=start)
