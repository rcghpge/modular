# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from std.collections import Optional
from std.ffi import c_int, get_errno
from std.io import FileDescriptor, RawFileDescriptor
from std.sys import size_of
from std.sys._io import stdin, stdout, stderr
from std.sys._libc import FcntlCommands, fcntl, pipe

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)


def test_value_is_c_int() raises:
    assert_true(type_of(stdout._value) == c_int)
    assert_equal(stdout._value, 1)
    assert_equal(stderr._value, 2)
    assert_equal(RawFileDescriptor()._value, 1)


def test_isatty_with_standard_descriptors() raises:
    assert_true(type_of(stdin.isatty()) == Bool)
    assert_true(type_of(stdout.isatty()) == Bool)
    assert_true(type_of(stderr.isatty()) == Bool)


def test_optional_uses_fd_niche() raises:
    comptime assert (
        size_of[Optional[RawFileDescriptor]]() == size_of[RawFileDescriptor]()
    )
    comptime assert size_of[RawFileDescriptor]() == size_of[c_int]()

    var fd: Optional[RawFileDescriptor] = RawFileDescriptor(7)
    assert_true(fd)
    assert_equal(fd[]._value, 7)

    fd = None
    assert_false(fd)

    fd = RawFileDescriptor(3)
    assert_true(fd)
    assert_equal(fd[]._value, 3)


def test_isatty_with_invalid_fd() raises:
    # A descriptor that is not open is not a tty.
    assert_false(RawFileDescriptor(9999).isatty())


def _make_pipe() raises -> Tuple[c_int, c_int]:
    var fds = [c_int(-1), c_int(-1)]
    if pipe(fds.unsafe_ptr()) != 0:
        raise Error("pipe failed err: ", String(get_errno()))
    return (fds[0], fds[1])


def _is_open(fd: c_int) -> Bool:
    return fcntl(fd, FcntlCommands.F_GETFD) != -1


def _read_exact(mut fd: FileDescriptor, count: Int) raises -> String:
    # A single `read` may return fewer bytes than requested.
    var buffer = List[Byte](length=count, fill=0)
    var filled = 0
    while filled < count:
        var n = fd.read_bytes(Span(buffer)[filled:])
        if n == 0:
            raise Error("unexpected EOF after ", filled, " bytes")
        filled += n
    return String(StringSlice(unsafe_from_utf8=Span(buffer)))


def test_close_closes_fd() raises:
    var read_fd, write_fd = _make_pipe()
    var owned = FileDescriptor(unsafe_raw_fd=RawFileDescriptor(write_fd))
    var was_open = _is_open(write_fd)
    owned^.close()
    assert_true(was_open)
    assert_false(_is_open(write_fd))
    # A second close fails with EBADF, so `close()` closed it exactly once.
    with assert_raises():
        RawFileDescriptor(write_fd).close()
    RawFileDescriptor(read_fd).close()


def test_write_then_read_through_pipe() raises:
    var read_fd, write_fd = _make_pipe()
    var writer = FileDescriptor(unsafe_raw_fd=RawFileDescriptor(write_fd))
    writer.write_string("hello ")
    try:
        writer.write_bytes_exact("fd".as_bytes())
    finally:
        # Closing the write end lets the reader observe EOF.
        writer^.close()

    var reader = FileDescriptor(unsafe_raw_fd=RawFileDescriptor(read_fd))
    var buffer = List[Byte](length=1, fill=0)
    var text: String
    var eof_read: Int
    try:
        text = _read_exact(reader, 8)
        eof_read = reader.read_bytes(buffer)
    finally:
        reader^.close()
    assert_equal(text, "hello fd")
    assert_equal(eof_read, 0)


def test_move_only() raises:
    assert_true(conforms_to(FileDescriptor, Movable))
    assert_false(conforms_to(FileDescriptor, Copyable))


def test_optional_owned_fd_uses_niche() raises:
    comptime assert (
        size_of[Optional[FileDescriptor]]() == size_of[FileDescriptor]()
    )


def test_read_bytes_raises_on_invalid_fd() raises:
    var fd = FileDescriptor(unsafe_raw_fd=RawFileDescriptor(9999))
    var buffer = List[Byte](length=4, fill=0)
    var read_raised = False
    try:
        _ = fd.read_bytes(buffer)
    except:
        read_raised = True
    # Nothing owns descriptor 9999, so closing it fails too.
    with assert_raises():
        fd^.close()
    assert_true(read_raised)


def test_write_bytes_returns_count() raises:
    var read_fd, write_fd = _make_pipe()
    var writer = FileDescriptor(unsafe_raw_fd=RawFileDescriptor(write_fd))
    var n: Int
    try:
        n = writer.write_bytes("abc".as_bytes())
    finally:
        writer^.close()

    var reader = FileDescriptor(unsafe_raw_fd=RawFileDescriptor(read_fd))
    var text: String
    try:
        text = _read_exact(reader, n)
    finally:
        reader^.close()
    assert_equal(n, 3)
    assert_equal(text, "abc")


def test_write_bytes_raises_on_invalid_fd() raises:
    var fd = FileDescriptor(unsafe_raw_fd=RawFileDescriptor(9999))
    var write_raised = False
    var exact_raised = False
    try:
        _ = fd.write_bytes("x".as_bytes())
    except:
        write_raised = True
    try:
        fd.write_bytes_exact("x".as_bytes())
    except:
        exact_raised = True
    with assert_raises():
        fd^.close()
    assert_true(write_raised)
    assert_true(exact_raised)


def test_read_bytes_honors_buffer_length() raises:
    var read_fd, write_fd = _make_pipe()
    var writer = FileDescriptor(unsafe_raw_fd=RawFileDescriptor(write_fd))
    writer.write_string("abcdef")
    writer^.close()

    var reader = FileDescriptor(unsafe_raw_fd=RawFileDescriptor(read_fd))
    var buffer = List[Byte](length=4, fill=0)
    var n: Int
    var tail: String
    try:
        n = reader.read_bytes(buffer)
        tail = _read_exact(reader, 6 - n)
    finally:
        reader^.close()
    assert_true(n > 0 and n <= 4)
    assert_equal(
        String(StringSlice(unsafe_from_utf8=Span(buffer)[:n])) + tail, "abcdef"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
