# Binary Executable Format

<!-- TOC -->

This document describes the goals and rationale for the in-memory representation
the GraphRT executor uses to run primitives, known as the Binary Executable
Format "BEF".

We have the following goals:

-   We want this to be simple and relatively stable - though it is versioned
    just in case (tm). We don't want this to contain domain specific
    abstractions that will require changing it frequently over time.

-   This format is designed to be usable as an immutable region in memory,
    allowing it to be built offline and conveniently `mmap`'d in (or stored in
    the `.rodata` segment of an executable, etc).

-   This is aligned with the abstractions used by the GraphRT executor and
    includes key abstractions that it can directly use to make it speedy and
    memory efficient. The design puts more burden on the encoder of a BEF file
    (e.g. requiring multiple passes) to make the decoder fast.

-   This needs to be producible online e.g. in training or interactive use
    cases.

-   We need to represent location information in order to reflect runtime errors
    (e.g. shape errors) back to the user source program.

-   We should follow normal best practices for binary encodings, e.g. having
    tools to round trip them to a textual form (MLIR in our case), support
    arbitrary sized files (not limited to 4GB), be efficient to read, make
    reasonable attempts to be dense, etc.

In addition to describing the format, this document describes some of the
rationale for the design decisions as well as TODO items that need to be
completed.

## Fundamental Concepts

**Grammar**

```none
  BYTE                  ::= `0x00`...`0xFF`
  VBR                   ::= (`0x80`...`0xFF`)* (`0x00`...`0x7F`)
  FIXED32               ::= BYTE BYTE BYTE BYTE
  NUL_TERMINATED_STRING ::= (`0x01`...`0xFF`)* `0x00`
  OFFSET                ::= VBR
  INDEX                 ::= VBR
```

A BEF file is formed as a byte stream whose top-level structure is a list of
"sections" that hold various sorts of data. It uses a few fundamental concepts:

*   *VBRs* are encoded in a "Variable Byte Rate" (VBR) encoding, allowing
    small integers (less than 2^7) to be stored in one byte, integers up to 2^14
    to be stored in two bytes, etc. This is done by using the high bit of each
    byte in the stream to indicate "more bytes are coming".

*   *Fixed Integers* are unsigned integers with fixed bit width. The values are
    stored in little-endian byte order.

*   *NUL Terminated Strings* are terminated with the ASCII `NUL` character
    (whose byte value is zero).  These are not used in cases where a string may
    contain an embedded `NUL` character. 

*   *Offsets* are VBRs that demarcate a byte offset in the stream from a
    fixed position (typically the start of a section). These are useful when the
    section contains a bunch of variable length things, e.g. when referring to a
    string in the string table. Because variable sized integers are used
    pervasively in BEF files, offsets are very common.

*   *Indexes* are VBRs that demarcate an entry in a table, e.g. a register
    or primitives in a BEFFunction. These are typically used when indexing into
    a table where the structure is expected to be decoded and held in memory by
    the reader, e.g. as is the case with the primitives and types tables.

BEF files can be created by an arbitrary producer, but the standard ways is to
use the [MLIRToBEF](../lib/MLIRToBEF)
translation library (or standalone tool), which turns an MLIR representation of
a program into a BEF file.

The `BEFToMLIR` translation library performs the reverse translation from BEF
file to MLIR representation of the program. It is useful for writing BEF
encoder testcases and for dealing with that "random file someone sent you".

## Sections

**Grammar**

```none
  SECTION                ::= SECTION_HEADER SECTION_DATA
  SECTION_HEADER         ::= SECTION_ID VBR<LENGTH_AND_ALIGNMENT>
                             SECTION_BODY_ALIGNMENT?
  SECTION_ID             ::= BYTE
  LENGTH_AND_ALIGNMENT   ::= (SECTION_LENGTH << 1) | (SECTION_ALIGNMENT_FLAG)
  SECTION_BODY_ALIGNMENT ::= BYTE<"Alignment"> BYTE<"AlignmentPadding">*
```

*Sections* are the top level entities in the file (after the header). Each
section contains a *Section ID*, a length (allowing the section to be skipped
over entirely by the reader) and an optional alignment for the contents of the
section.

Section IDs and other fundamental constants are defined in
[`BEFEncoding.h`](../include/GraphRT/BEF/BEFEncoding.h),
and utilities for decoding the basic file structures like VBR integers are
defined in
[`BEFReader.h`](../include/GraphRT/BEF/BEFReader.h).

The LENGTH_AND_ALIGNMENT contains SECTION_LENGTH and SECTION_ALIGNMENT_FLAG. The
SECTION_LENGTH contains one bit shifted value of the section length and the
SECTION_ALIGNMENT_FLAG (0 bit) indicates if SECTION_BODY_ALIGNMENT exists or
not; 0 means the section body starts immediately and 1 means
SECTION_BODY_ALIGNMENT follows.

## Top Level Structure

**Grammar**

```none
  BEF_FILE     ::= `0xBE` `0xF0` SECTION*

  SECTION_DATA ::= STRINGS_SECTION
  SECTION_DATA ::= ATTRIBUTES_SECTION
  SECTION_DATA ::= PRIMITIVES_SECTION
  SECTION_DATA ::= TYPES_SECTION
  SECTION_DATA ::= BEF_FUNCTION_INDEX_SECTION
  SECTION_DATA ::= BEF_FUNCTIONS_SECTION
  SECTION_DATA ::= LOCATION_FILENAMES_SECTION
  SECTION_DATA ::= LOCATIONS_SECTION
  SECTION_DATA ::= ATTRIBUTE_TYPES_SECTION
  SECTION_DATA ::= ATTRIBUTE_NAMES_SECTION
  SECTION_DATA ::= REGISTER_TYPES_SECTION

  // Unknown section.
  SECTION_DATA ::= BYTE*
```

The top level structure of the file is a two-byte "magic number" of `0xBEF0`
and a list of sections.  The `0xBEF0` magic number allows us to encode version
numbers, e.g. the following would be `0xBEF1`.

The reader skips over unknown sections, which could be useful for future
evolution of the format, e.g. if we want to store extra metadata in the BEF
format for some purpose, or store binary tensor blobs.

### Strings Section

**Grammar**

```none
  STRINGS_SECTION ::= NUL_TERMINATED_STRING*
```

The Strings section contains a list of `NUL` terminated strings used by the
program (e.g. for type and primitive names). Entries in this section are
referenced by an Offset from the start of the section.  This section does not
support embedded `NUL` in strings.

### Attributes Section

**Grammar**

```none
  ATTRIBUTES_SECTION ::= BYTE*
```

The Attributes section contains the value of attributes used by primitives in
the BEF program. They are stored on the natural alignment boundary of the type,
and the address of the attribute is directly passed into the primitive
implementation C/C++ function as an const pointer.

BEF files only support a subset of MLIR attributes, currently including:

*   i1 integers: stored as 1-byte integers.
*   i32 integers: stored as 4-byte little endian integers.
*   i64 integers: stored as 8-byte little endian integers.
*   f32 floats: stored as IEEE single precision floats.
*   f64 floats: stored as IEEE double precision floats.
*   type enums: stored as 1-byte integers.
*   strings: stored as arrays of bytes, not necessarily NUL terminated.
*   dense elements: stored as shape dtype, shape rank, elements count, followed
    by shape elements and elements themselves. Each element can be any of the
    integer and float format above.
*   arrays: all elements of which are of the same type and fixed in width (eg.
    i32, f32, type).
*   aggregates: stored as array of i32 integers, which are offsets to other
    constants in Attributes Section. These nested constants can be of any
    supported attribute type including aggregates. Unlike arrays, an aggregate
    can contain a mix of different attribute types.

TODO: Support 8/16-bit integers.

***Rationale***

The GraphRT executor needs to refer to attribute values, and since the BEF file
is designed to be memory mapped in, we can directly use the encoding in memory.
This means that primitives will need to bswap attributes for big-endian systems,
but they generally have bswap'ing loads anyway so this shouldn't impose a
performance penalty.

### Primitive Names Section

**Grammar**

```none
  PRIMITIVE_NAMES_SECTION ::= VBR<"PrimitiveCount"> OFFSET<"PrimitiveNameOffset">*
```

The Primitive names section defines a table of primitive names, directly
corresponding to the names in the MLIR GraphRT executor program. This allows
references from the BEF Functions section to use dense indexes into the table.

The format of this section is a VBR count of the number of primitives in
the table, followed by 'count' VBR values which are offsets into the string
table for the primitive name.

***Rationale***

We want dense indexes from primitives in the BEF Functions section, and the
GraphRT Executor wants to resolve names to primitive implementations at startup
time anyway, as such, it makes sense for the GraphRT executor to build its own
mutable array in memory for primitives.

### Types Section

**Grammar**

```none
  TYPES_SECTION ::= VBR<"TypeCount"> OFFSET<"TypeNameOffset">*
```

The Types section defines a table of type names used by the program, and is
laid out exactly the same as the [Primitive Names section](#primitive-names-section). The
[BEF Functions section](#bef-functions-section) uses Indexes into this section to
specify types of registers.

### BEFFunctionIndex Section

**Grammar**

```none
  BEF_FUNCTION_INDEX_SECTION ::= VBR<"NumBEFFunctions"> BEF_FUNCTION_ENTRY*
  BEF_FUNCTION_ENTRY         ::= OFFSET<"BEFFunction"> OFFSET<"Name">
                                 VBR<"NumArguments"> INDEX<"Type">* 
                                 VBR<"NumResults"> INDEX<"Type">*
```

The BEFFunctionIndex section defines a table of "functions" in the BEF file, 
including an Offset into the [BEF Functions Section](#bef-functions-section), a
name (an Offset
into the [Strings section](#strings-section), which may be an empty string), and
a list of argument and result types.

***Rationale***

This defines a symbol table for the BEF file, allowing clients to look up
BEF functions by name.  While we could intersperse this information into the
[BEF Functions section](#bef-functions-section) itself, doing so would require the
reader to make a pass over all of the BEF functions ahead of time. We'd prefer to
have a quick index that the reader can scan at load time, deferring processing
of any individual function until it is needed.

### BEF Functions Section

**Grammar**

```none
  BEF_FUNCTIONS_SECTION ::= BEF_FUNCTION*
```

The BEF Functions section is a list of BEF Function records emitted to the byte
stream and then addressed by an Offset indicated by the
[BEFFunctionIndex section](#bef-functionindex-section). This section is 4-byte
aligned.

***Rationale***

Regions in MLIR are a generalization of a unit of computation. The BEF format
and GraphRT executor support a subset of MLIR region features that are core to
the abstractions we need to model - in particular, while MLIR regions can have
multiple basic blocks in them, BEF Functions only support a single block - this
guarantees that they are always be a DAG of computation.

BEF Functions are used for top-level MLIR `func` operations (indexed by the
function name table) as well as in nested positions for control flow and other
concepts that occur with MLIR regions.

### BEF Function Definition

**Grammar**

```none
  BEF_FUNCTION    ::= OFFSET<"Location"> REGISTER_TABLE PRIMITIVE_TABLE
                      RESULT_REGS BYTE<"AlignmentPadding">* PRIMITIVE+

  REGISTER_TABLE  ::= VBR<"NumRegs"> REGISTER_ENTRY*
  REGISTER_ENTRY  ::= VBR<"NumUses">

  PRIMITIVE_TABLE ::= VBR<"NumPrimitives"> PRIMITIVE_ENTRY*
  PRIMITIVE_ENTRY ::= OFFSET<"PrimitiveOffset"> VBR<"NumOperands">

  RESULT_REGS     ::= INDEX<"Register">*
```

Each BEF Function is defined by a location (an offset into the
[LocationPositions section](#locationpositions-section)), a register table, a
primitive table, a list of result registers, a list of primitives with 4-byte
alignment, and ends with a FIXED32 integer of value zero.

The Register Table is a count of registers, and an entry for each register -
indicating the number of primitives in this section that use the register.

The Primitive Table for a BEF Function is a count of primitives, an offset
(from the end of the Primitive Table) of the start of the primitive and the
number of operands that the primitive has.

The primitive list that is following the Primitive Table contains all the
primitives used in this BEF function. Note that every BEF function has a
pseudo-primitive that is the single entry point to the rest of the primitives.
Specifially, a pseudo-primitive defines registers for BEF function arguments and a
pseudo-register that is conceptually used by primitives that takes no arguments.

The result registers specify the register values to return, and must align with
the function result types from the
[BEF FunctionIndex section](#bef-functionindex-section).

***Rationale***

These two tables are key to allowing the GraphRT executor to efficiently
dispatch and destroy primitives in a lock-free way. At startup time, the
executor inflates these two tables into arrays, resolving the type descriptors
for the types, and builds a table of ready counts for the primitives. Having a
table of primitives allows the use of primitive indexes.

### Primitive Definition

**Grammar**

```none
  PRIMITIVE        ::= PRIMITIVE_HEADER PRIMITIVE_RESULT_TABLE PRIMITIVE_BODY

  PRIMITIVE_HEADER ::= FIXED32<"PrimitiveCode"> FIXED32<"PrimitiveLocation">
                       FIXED32<"NumArguments"> FIXED32<"NumAttributes">
                       FIXED32<"NumFunctions"> FIXED32<"NumResults">

  PRIMITIVE_RESULT_TABLE
                   ::= FIXED32<"NumUsedBys">*

  PRIMITIVE_BODY   ::= FIXED32<PrimitiveArgument>* FIXED32<PrimitiveAttribute>*
                       FIXED32<PrimitiveFunction>* FIXED32<PrimitiveResult>*
                       FIXED32<PrimitiveUsedBy>*
```

Each instance of a primitive includes a primitive header, a result table and a
primitive body. The primitive header consists of a opcode (an index into the
Primitives table, defined by the Primitives names section), a location (an offset into
the [LocationPositions section](#locationpositions-section)) the numbers of
arguments, attributes, functions and results in the primitive body.

The result table contains NumResults FIXED32 integers, indicating the number of
users for each corresponding result. The primitive body consists of zero or more
inputs (indexes into the Register Table for the enclosing BEF Function), zero or more
constants (offsets into the [Attributes section](#attributes-section)), zero or
more functions (these are regions to the primitive, stored as indexes into the
[FunctionIndex section](#bef-functionindex-section)), zero or more results (indexes into the
Register table for the enclosing BEF function), and zero or more 'used by' records (which are
indexes into the Primitive Table for this function). 'used by' records for the
same result are grouped together and these groups are in the same order of
'result' records. For example, if there are two results, A and B, and A has one
user (a0) and B has two users (b0, b1), then FIXED32<"NumResults"> will be
`0x00000002`, followed by two FIXED32<"NumUsedBys">, `0x00000001` `0x00000002`;
and in the primitive body there will be three "used by" records, a0, b0 and b1,
consecutively.

***Rationale***

This record allows us to efficiently form the array of arguments and results
that get passed to a C/C++ function that implements the primitive. When the primitive
completes, the UsedBy records allow us to efficiently trigger execution of data
dependent primitives if they are ready, by decrementing the "NumOperands" field
for the primitive.

### LocationFilenames Section

**Grammar**

```none
  LOCATION_FILENAMES_SECTION ::= NUL_TERMINATED_STRING*
```

This section contains a list of NUL terminated strings used by locations
section. A filename field of FileLineCol location and name field of Name
location are stored in this section.

***Rationale***

We choose to store these separately from the string table, because these
locations are only ever decoded in the case of an error. There is no reason to
dirty data cache lines with them, and we expect no reuse with other general
strings.

### LocationPositions Section

**Grammar**

```none
  LOCATIONS_SECTION ::= LOCATION*
  LOCATION ::= OFFSET<"Filename"> VBR<"LineNum"> VBR<"ColumnNum">
```

This section contains a list of "locations", which are a VBR offset into the
LocationFilenames section, and VBRs for the line and column of the position.

### AttributeTypes Section

**Grammar**

```none
  ATTRIBUTE_TYPES_SECTION   ::= VBR<"NumAttributes"> ATTRIBUTE_TYPE_ENTRY*
  ATTRIBUTE_TYPE_ENTRY      ::= OFFSET<"Attributes"> BYTE<"AttributeType">
```

The AttributeTypes section is an optional section which is not needed for BEF
execution, but this section is necessary for BEFToMLIR conversion. This
section describes the type information for each attribute (specified by an
Offset into [Attributes section](#attributes-section)). An attribute type is an
one-byte enum described in [Attributes section](#attributes-section) and defined
in [`BEFEncoding.h`](../include/GraphRT/BEF/BEFEncoding.h).

### AttributeNames Section

**Grammar**

```none
  ATTRIBUTE_NAMES_SECTION ::= VBR<"NumFunctions"> PRIMITIVE_TABLE*
  PRIMITIVE_TABLE         ::= VBR<"NumPrimitives"> OFFSET<"AttributeName">*
```

The AttributeNames sections is an optional section which is not needed for BEF
execution, but this section is necessary for BEFToMLIR conversion. This
section describes the attribute names used by each primitive. There is a 1:1
mapping between Primitive Table in this section and BEF Function Entries in
[BEF FunctionIndex section](#bef-functionindex-section). And there is a 1:1 mapping
between primitive entries in this section and primitive entries in Functions
section's primitive table. Each primitive entry contains any number of
AttributeNames. AttributeName is an offset to Strings section that specifies the
name of the attribute used by this primitive.

### RegisterTypes Section

**Grammar**

```none
  REGISTER_TYPES_SECTION ::= VBR<"NumFunctions"> REGISTER_TYPE_TABLE
  REGISTER_TYPE_TABLE    ::= VBR<"NumRegs"> INDEX<"Types">*
```

The RegisterTypes section is an optional section which is not needed for BEF
execution, but this section is necessary for BEFToMLIR conversion. This
section describes the type information for registers in each function. The
RegisterType Table is a count of registers, and an entry for each register
indicating its index into [Types section](#types-section). There is a 1:1
mapping between RegisterType Tables and BEF Function Entries in
[BEF FunctionIndex section](#bef-functionindex-section). And it is a 1:1 mapping between
types in a RegisterType Table and registers in the RegisterTable of the
corresponding BEF Function.
