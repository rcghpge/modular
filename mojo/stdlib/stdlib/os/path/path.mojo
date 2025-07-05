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
"""Provides a set of operating-system independent functions for manipulating
file system paths.

You can import these APIs from the `os.path` package. For example:

```mojo
from os.path import isdir
```
"""

from pwd import getpwuid
from stat import S_ISDIR, S_ISLNK, S_ISREG
from sys import CompilationTarget, os_is_linux, os_is_macos, os_is_windows


from .. import PathLike
from .._linux_aarch64 import _lstat as _lstat_linux_arm
from .._linux_aarch64 import _stat as _stat_linux_arm
from .._linux_x86 import _lstat as _lstat_linux_x86
from .._linux_x86 import _stat as _stat_linux_x86
from .._macos import _lstat as _lstat_macos
from .._macos import _stat as _stat_macos
from ..env import getenv
from ..fstat import stat
from ..os import sep


# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #
fn _constrain_unix():
    constrained[
        not os_is_windows(), "operating system must be Linux or macOS"
    ]()


@always_inline
fn _get_stat_st_mode(var path: String) raises -> Int:
    @parameter
    if os_is_macos():
        return Int(_stat_macos(path^).st_mode)
    elif CompilationTarget.has_neon():
        return Int(_stat_linux_arm(path^).st_mode)
    else:
        return Int(_stat_linux_x86(path^).st_mode)


@always_inline
fn _get_lstat_st_mode(var path: String) raises -> Int:
    @parameter
    if os_is_macos():
        return Int(_lstat_macos(path^).st_mode)
    elif CompilationTarget.has_neon():
        return Int(_lstat_linux_arm(path^).st_mode)
    else:
        return Int(_lstat_linux_x86(path^).st_mode)


# ===----------------------------------------------------------------------=== #
# expanduser
# ===----------------------------------------------------------------------=== #


fn _user_home_path(path: String) -> String:
    @parameter
    if os_is_windows():
        return getenv("USERPROFILE")
    else:
        var user_end = path.find(sep, 1)
        if user_end < 0:
            user_end = len(path)
        # Special POSIX syntax for ~[user-name]/path
        if len(path) > 1 and user_end > 1:
            try:
                return pwd.getpwnam(path[1:user_end]).pw_dir
            except:
                return ""
        else:
            var user_home = getenv("HOME")
            # Fallback to password database if `HOME` not set
            if not user_home:
                try:
                    user_home = pwd.getpwuid(getuid()).pw_dir
                except:
                    return ""
            return user_home


fn expanduser[PathLike: os.PathLike, //](path: PathLike) raises -> String:
    """Expands a tilde "~" prefix in `path` to the user's home directory.

    For example, `~/folder` becomes `/home/current_user/folder`. On macOS and
    Linux a path starting with `~user/` will expand to the specified user's home
    directory, so `~user/folder` becomes `/home/user/folder`.

    If the home directory cannot be determined, or the `path` is not prefixed
    with "~", the original path is returned unchanged.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path that is being expanded.

    Returns:
        The expanded path.
    """
    var fspath = path.__fspath__()
    if not fspath.startswith("~"):
        return fspath
    var userhome = _user_home_path(fspath)
    if not userhome:
        return fspath
    var path_split = fspath.split(os.sep, 1)
    # If there is a properly formatted separator, return expanded fspath.
    if len(path_split) == 2:
        return os.path.join(userhome, path_split[1])
    # Path was a single `~` character, return home path
    return userhome


# ===----------------------------------------------------------------------=== #
# isdir
# ===----------------------------------------------------------------------=== #
fn isdir[PathLike: os.PathLike, //](path: PathLike) -> Bool:
    """Return True if path is an existing directory. This follows
    symbolic links, so both islink() and isdir() can be true for the same path.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the directory.

    Returns:
        True if the path is a directory or a link to a directory and
        False otherwise.
    """
    _constrain_unix()
    var fspath = path.__fspath__()
    try:
        var st_mode = _get_stat_st_mode(fspath)
        if S_ISDIR(st_mode):
            return True
        return S_ISLNK(st_mode) and S_ISDIR(_get_lstat_st_mode(fspath^))
    except:
        return False


# ===----------------------------------------------------------------------=== #
# isfile
# ===----------------------------------------------------------------------=== #


fn isfile[PathLike: os.PathLike, //](path: PathLike) -> Bool:
    """Test whether a path is a regular file.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the directory.

    Returns:
        Returns True if the path is a regular file.
    """
    _constrain_unix()
    var fspath = path.__fspath__()
    try:
        var st_mode = _get_stat_st_mode(fspath)
        if S_ISREG(st_mode):
            return True
        return S_ISLNK(st_mode) and S_ISREG(_get_lstat_st_mode(fspath))
    except:
        return False


# ===----------------------------------------------------------------------=== #
# islink
# ===----------------------------------------------------------------------=== #
fn islink[PathLike: os.PathLike, //](path: PathLike) -> Bool:
    """Return True if path refers to an existing directory entry that is a
    symbolic link.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the directory.

    Returns:
        True if the path is a link to a directory and False otherwise.
    """
    _constrain_unix()
    try:
        return S_ISLNK(_get_lstat_st_mode(path.__fspath__()))
    except:
        return False


# ===----------------------------------------------------------------------=== #
# dirname
# ===----------------------------------------------------------------------=== #


fn dirname[PathLike: os.PathLike, //](path: PathLike) -> String:
    """Returns the directory component of a pathname.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to a file.

    Returns:
        The directory component of a pathname.
    """
    var fspath = path.__fspath__()
    var i = fspath.rfind(os.sep) + 1
    var head = fspath[:i]
    if head and head != os.sep * len(head):
        return String(head.rstrip(os.sep))
    return head


# ===----------------------------------------------------------------------=== #
# exists
# ===----------------------------------------------------------------------=== #


fn exists[PathLike: os.PathLike, //](path: PathLike) -> Bool:
    """Return True if path exists.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the directory.

    Returns:
        Returns True if the path exists and is not a broken symbolic link.
    """
    _constrain_unix()
    try:
        _ = _get_stat_st_mode(path.__fspath__())
        return True
    except:
        return False


# ===----------------------------------------------------------------------=== #
# lexists
# ===----------------------------------------------------------------------=== #


fn lexists[PathLike: os.PathLike, //](path: PathLike) -> Bool:
    """Return True if path exists or is a broken symlink.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the directory.

    Returns:
        Returns True if the path exists or is a broken symbolic link.
    """
    _constrain_unix()
    try:
        _ = _get_lstat_st_mode(path.__fspath__())
        return True
    except:
        return False


# ===----------------------------------------------------------------------=== #
# getsize
# ===----------------------------------------------------------------------=== #


fn getsize[PathLike: os.PathLike, //](path: PathLike) raises -> Int:
    """Return the size, in bytes, of the specified path.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the file.

    Returns:
        The size of the path in bytes.
    """
    return stat(path.__fspath__()).st_size


# ===----------------------------------------------------------------------=== #
# is_absolute
# ===----------------------------------------------------------------------=== #


fn is_absolute[PathLike: os.PathLike, //](path: PathLike) -> Bool:
    """Return True if `path` is an absolute path name.
    On Unix, that means it begins with a slash.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to check.

    Returns:
        Return `True` if path is an absolute path name.
    """
    _constrain_unix()
    return path.__fspath__().startswith(sep)


# ===----------------------------------------------------------------------=== #
# join
# ===----------------------------------------------------------------------=== #


# TODO(MOCO-1532):
#   Use StringSlice here once param inference bug for empty variadic
#   list of parameterized types is fixed.
fn join(var path: String, *paths: String) -> String:
    """Join two or more pathname components, inserting '/' as needed.
    If any component is an absolute path, all previous path components
    will be discarded.  An empty last part will result in a path that
    ends with a separator.

    Args:
        path: The path to join.
        paths: The paths to join.

    Returns:
        The joined path.
    """
    var joined_path = path

    for cur_path in paths:
        if cur_path.startswith(sep):
            joined_path = cur_path
        elif not joined_path or path.endswith(sep):
            joined_path += cur_path
        else:
            joined_path += sep + cur_path

    return joined_path^


# ===----------------------------------------------------------------------=== #
# split
# ===----------------------------------------------------------------------=== #


def split[PathLike: os.PathLike, //](path: PathLike) -> (String, String):
    """
    Split a given pathname into two components: head and tail. This is useful
    for separating the directory path from the filename. If the input path ends
    with a separator, the tail component will be empty. If there is no separator
    in the path, the head component will be empty, and the entire path will be
    considered the tail. Trailing separators in the head are stripped unless the
    head is the root directory.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to be split.

    Returns:
        A tuple containing two strings: (head, tail).
    """
    var fspath = path.__fspath__()
    var i = fspath.rfind(os.sep) + 1
    var head, tail = fspath[:i], fspath[i:]
    if head and head != String(os.sep) * len(head):
        head = String(head.rstrip(sep))
    return head, tail


fn basename[PathLike: os.PathLike, //](path: PathLike) -> String:
    """Returns the tail section of a path.

    ```mojo
    from os.path import basename

    basename("a/path/foo.txt")  # returns "foo.txt"
    ```

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to retrieve the basename from.

    Returns:
        The basename from the path.
    """
    var fspath = path.__fspath__()
    var i = fspath.rfind(os.sep) + 1
    var head = fspath[i:]
    if head and head != os.sep * len(head):
        return String(head.rstrip(os.sep))
    return head


# TODO uncomment this when unpacking is supported
# fn join[PathLike: os.PathLike](path: PathLike, *paths: PathLike) -> String:
#     """Join two or more pathname components, inserting '/' as needed.
#     If any component is an absolute path, all previous path components
#     will be discarded.  An empty last part will result in a path that
#     ends with a separator.

#     Parameters:
#       PathLike: The type conforming to the os.PathLike trait.

#     Args:
#       path: The path to join.
#       paths: The paths to join.

#     Returns:
#       The joined path.
#     """
#     var paths_str= List[String]()

#     for cur_path in paths:
#         paths_str.append(cur_path[].__fspath__())

#     return join(path.__fspath__(), *paths_str)

# ===----------------------------------------------------------------------=== #
# split_extension
# ===----------------------------------------------------------------------=== #


# TODO: Move this to a generic path module when Windows is supported.
# As it can be used for both Windows and Unix-like systems.
fn _split_extension(
    path: StringSlice,
    sep: StringSlice,
    alt_sep: StringSlice,
    extension_sep: StringSlice,
) raises -> Tuple[String, String]:
    """Splits `path` into the root and extension.

    Args:
        path: The path to be split.
        sep: The separator used in the path.
        alt_sep: The alternative separator used in the path.
        extension_sep: The extension separator used in the path.

    Returns:
        A tuple containing two strings: (root, extension).
    """
    # Find the last extension separator after the last separator.
    var head_end = path.rfind(sep)
    if alt_sep:
        head_end = max(head_end, path.rfind(alt_sep))

    var file_end = path.rfind(extension_sep)
    if file_end > head_end:
        # skip all leading dots
        var file_start = head_end + 1
        while file_start < file_end:
            if path[file_start].as_string_slice() != extension_sep:
                return String(path[:file_end]), String(path[file_end:])
            file_start += 1

    return String(path), ""


fn split_extension[
    PathLike: os.PathLike, //
](path: PathLike) raises -> Tuple[String, String]:
    """Splits `path` into the root and extension.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to be split.

    Returns:
        A tuple containing two strings: (root, extension).
    """

    @parameter
    if os_is_windows():
        return _split_extension(path.__fspath__(), "\\", "/", ".")
    return _split_extension(path.__fspath__(), sep, "", ".")


# ===----------------------------------------------------------------------=== #
# splitroot
# ===----------------------------------------------------------------------=== #


fn splitroot[
    PathLike: os.PathLike, //
](path: PathLike) -> Tuple[String, String, String]:
    """Splits `path` into drive, root and tail. The tail contains anything after the root.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to be split.

    Returns:
        A tuple containing three strings: (drive, root, tail).
    """
    var p = path.__fspath__()
    alias empty = ""

    # Relative path, e.g.: 'foo'
    if p[:1] != sep:
        return empty, empty, p

    # Absolute path, e.g.: '/foo', '///foo', '////foo', etc.
    elif p[1:2] != sep or p[2:3] == sep:
        return empty, String(sep), p[1:]

    # Precisely two leading slashes, e.g.: '//foo'. Implementation defined per POSIX, see
    # https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap04.html#tag_04_13
    else:
        return empty, p[:2], p[2:]


# ===----------------------------------------------------------------------=== #
# expandvars
# ===----------------------------------------------------------------------=== #


fn _is_shell_special_variable(byte: Byte) -> Bool:
    """Checks if `$` + `byte` identifies a special shell variable, such as `$@`.

    Args:
        byte: The byte to check.

    Returns:
        True if the byte is a special shell variable and False otherwise.
    """
    alias shell_variables = InlineArray[Int, 17](
        ord("*"),
        ord("#"),
        ord("$"),
        ord("@"),
        ord("!"),
        ord("?"),
        ord("-"),
        ord("0"),
        ord("1"),
        ord("2"),
        ord("3"),
        ord("4"),
        ord("5"),
        ord("6"),
        ord("7"),
        ord("8"),
        ord("9"),
    )
    return Int(byte) in shell_variables


fn _is_alphanumeric(byte: Byte) -> Bool:
    """Checks if `byte` is an ASCII letter, number, or underscore.

    Args:
        byte: The byte to check.

    Returns:
        True if the byte is an ASCII letter, number, or underscore and False otherwise.
    """
    var b = Int(byte)
    return (
        b == ord("_")
        or ord("0") <= b
        and b <= ord("9")
        or ord("a") <= b
        and b <= ord("z")
        or ord("A") <= b
        and b <= ord("Z")
    )


fn _parse_variable_name[
    immutable: ImmutableOrigin
](bytes: Span[Byte, immutable]) -> Tuple[StringSlice[immutable], Int]:
    """Returns the environment variable name and the byte count required to extract it.
    For `${}` expansions, two additional bytes are added to the byte count to account for the braces.

    Args:
        bytes: The bytes to extract the environment variable name from.

    Returns:
        The environment variable name and the byte count required to extract it.
    """
    if bytes[0] == ord("{"):
        if (
            len(bytes) > 2
            and _is_shell_special_variable(bytes[1])
            and bytes[2] == ord("}")
        ):
            return StringSlice(unsafe_from_utf8=bytes[1:2]), 3

        # Scan until the closing brace or the end of the bytes.
        var i = 1
        while i < len(bytes):
            if bytes[i] == ord("}"):
                return StringSlice(unsafe_from_utf8=bytes[1:i]), i + 1
            i += 1
        return StringSlice(unsafe_from_utf8=bytes[1:i]), i + 1
    elif _is_shell_special_variable(bytes[0]):
        return StringSlice(unsafe_from_utf8=bytes[0:1]), 1

    # Scan until we hit an invalid character in environment variable names.
    var i = 0
    while i < len(bytes) and _is_alphanumeric(bytes[i]):
        i += 1

    return StringSlice(unsafe_from_utf8=bytes[:i]), i


fn expandvars[PathLike: os.PathLike, //](path: PathLike) -> String:
    """Replaces `${var}` or `$var` in the path with values from the current environment variables.
    Malformed variable names and references to non-existing variables are left unchanged.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path that is being expanded.

    Returns:
        The expanded path.
    """
    var path_str = path.__fspath__()
    var bytes = path_str.as_bytes()
    var buf = String()

    # Byte scanning should be fine, ${} is ASCII.
    i = 0
    j = 0
    while j < len(bytes):
        if bytes[j] == ord("$") and j + 1 < len(bytes):
            if not buf:
                buf.reserve(new_capacity=2 * len(bytes))
            buf.write_bytes(bytes[i:j])

            var name, length = _parse_variable_name(bytes[j + 1 :])

            # Invalid syntax (`${}` or `${`) or $ was not followed by a name; write as is.
            if name.startswith("{") or name == "":
                buf.write_bytes(bytes[j : j + length + 1])
            # Shell variable (eg `$@` or `$*`); write as is.
            elif _is_shell_special_variable(name.as_bytes()[0]):
                buf.write_bytes(bytes[j : j + 2])
            # Environment variable; expand it. If no value, write as is.
            else:
                value = os.getenv(String(name))
                if value != "":
                    buf.write(value)
                else:
                    buf.write_bytes(bytes[j : j + length + 1])

            j += length
            i = j + 1
        j += 1

    if not buf:
        return path_str

    buf.write_bytes(bytes[i:])
    return buf
