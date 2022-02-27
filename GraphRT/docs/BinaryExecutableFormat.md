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
rationale for the design decisions.

## Fundamental Concepts

**Grammar**

```none
  BYTE                  ::= `0x00`...`0xFF`
  VBR                   ::= (`0x80`...`0xFF`)* (`0x00`...`0x7F`)
  FIXED16               ::= BYTE BYTE
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
  SECTION_DATA ::= BEF_FUNCTION_INDEX_SECTION
  SECTION_DATA ::= BEF_FUNCTIONS_SECTION
  SECTION_DATA ::= LOCATION_FILENAMES_SECTION
  SECTION_DATA ::= LOCATIONS_SECTION
  SECTION_DATA ::= ATTRIBUTE_KIND_MD_SECTION
  SECTION_DATA ::= ATTRIBUTE_NAMES_MD_SECTION
  SECTION_DATA ::= MLIRTYPE_STRINGS_MD_SECTION
  SECTION_DATA ::= MLIRTYPE_MD_SECTION

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

*   i1/i8 integers: stored as 1-byte integers.
*   i16/i32/i64 integers: stored as 2/4/8-byte little endian integers.
*   f32 floats: stored as IEEE single precision floats.
*   f64 floats: stored as IEEE double precision floats.
*   type enums: stored as 1-byte integers.
*   strings: stored as arrays of bytes, not necessarily NUL terminated.
*   arrays: all elements of which are of the same type and fixed in width (eg.
    i32, f32, type).

***Rationale***

The GraphRT executor needs to refer to attribute values, and since the BEF file
is designed to be memory mapped in, we can directly use the encoding in memory.
This means that primitives will need to bswap attributes for big-endian systems,
but they generally have bswap'ing loads anyway so this shouldn't impose a
performance penalty.

### Primitive Names Section

**Grammar**

```none
  PRIMITIVE_NAMES_SECTION ::= VBR<NumPrimitiveNames>
                              OFFSET<"PrimitiveNameOffset">*
```

The Primitive names section defines a table of primitive names, directly
corresponding to the names in the MLIR GraphRT executor program. This allows
references from the BEF Functions section to use dense indexes into the table.

This section contains one VBR value for each primitive, which is an offset into
the string table for the primitive name.

***Rationale***

We want dense indexes from primitives in the BEF Functions section, and the
GraphRT Executor wants to resolve names to primitive implementations at startup
time anyway, as such, it makes sense for the GraphRT executor to build its own
mutable array in memory for primitives.

We encode the count of primitive names so the executor can allocate one block
of memory up front, rather than use a vector-style resizing mechanism.

### BEFFunctionIndex Section

**Grammar**

```none
  BEF_FUNCTION_INDEX_SECTION ::= VBR<"NumBEFFunctions"> BEF_FUNCTION_ENTRY*
  BEF_FUNCTION_ENTRY         ::= OFFSET<"BEFFunction"> OFFSET<"Name">
                                 OFFSET<"Location"> 
                                 VBR<"NumArguments"> VBR<"NumResults"> 
```

The BEFFunctionIndex section defines a table of "functions" in the BEF file, 
including an Offset into the [BEF Functions Section](#bef-functions-section), a
name (an Offset
into the [Strings section](#strings-section), which may be an empty string), and
a count of argument and result types.

***Rationale***

This defines a symbol table for the BEF file, allowing clients to look up
BEF functions by name.  While we could intersperse this information into the
[BEF Functions section](#bef-functions-section) itself, doing so would require
the reader to make a pass over all of the BEF functions ahead of time. We prefer
to have a quick index that the reader can scan at load time, deferring
processing of any individual function until it is needed.

We encode the count of function names so the executor can allocate one block
of memory up front, rather than use a vector-style resizing mechanism.

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
  BEF_FUNCTION    ::= VBR<"NumPrimitives"> REGISTER_TABLE PRIMITIVE_TABLE
                      RESULT_REGS BYTE<"AlignmentPadding">* PRIMITIVE+

  REGISTER_TABLE  ::= VBR<"NumRegs"> REGISTER_ENTRY*
  REGISTER_ENTRY  ::= VBR<"NumUses">

  PRIMITIVE_TABLE ::= PRIMITIVE_ENTRY*
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
  PRIMITIVE              ::= PRIMITIVE_HEADER
                             PRIMITIVE_RESULT_TABLE
                             PRIMITIVE_BODY

  PRIMITIVE_HEADER       ::= FIXED32<"PrimitiveCode">
                             FIXED32<"PrimitiveLocation">
                             FIXED16<"NumOperands">
                             FIXED16<"NumAttributes">
                             FIXED16<"NumFunctions">
                             FIXED16<"NumResults">
                             
  PRIMITIVE_RESULT_TABLE ::= FIXED32<"NumUsedBys">*

  PRIMITIVE_BODY         ::= FIXED32<PrimitiveOperand>*
                             FIXED32<PrimitiveAttribute>*
                             FIXED32<PrimitiveFunction>*
                             FIXED32<PrimitiveResult>*
                             FIXED32<PrimitiveUsedBy>*
```

Each instance of a primitive includes a primitive header, a result table and a
primitive body. The primitive header consists of a opcode (an index into the
Primitives table, defined by the Primitives names section), a location (an offset into
the [LocationPositions section](#locationpositions-section)) the numbers of
operands, attributes, functions and results in the primitive body.

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

This record allows us to efficiently form the array of operands and results
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

### ATTRIBUTE_KIND_MD Section

**Grammar**

```none
  ATTRIBUTE_KIND_MD_SECTION ::= VBR<"NumAttributes"> ATTRIBUTE_KIND_ENTRY*
  ATTRIBUTE_KIND_ENTRY      ::= OFFSET<"Attributes"> BYTE<"AttributeKind">
```

The ATTRIBUTE_KIND section specifies the kind of each attribute emitted to the
attribute section, allowing BEFToMLIR conversion (e.g allowing us to know the
difference between an i1/i8 attribute or an i32/f32). The kind is specified by a
one-byte enum defined in [`BEFEncoding.h`](../include/GraphRT/BEF/BEFEncoding.h).

### ATTRIBUTE_NAMES_MD Section

**Grammar**

```none
  ATTRIBUTE_NAMES_SECTION ::= PRIMITIVE_TABLE*
  PRIMITIVE_TABLE         ::= OFFSET<"AttributeName">*
```

The AttributeNames sections is an optional section which is not needed for BEF
execution, but which makes BEFToMLIR conversion much nicer. This
section describes the attribute names used by each primitive. There is a 1:1
mapping between Primitive Table in this section and BEF Function Entries in
[BEF FunctionIndex section](#bef-functionindex-section). Each attribute and
region for a primitive has an entry here specifying the name.  The offset is
into the STRINGS_MD section (not the STRINGS section).

***Rationale***

Attribute names are strings that are not needed by the BEF Executor and have
nothing to do with types or other metadata so we put them in their own section
looked up lazily by offset.

### MLIRTYPE_STRINGS_MD Section

**Grammar**

```none
  MLIRTYPE_STRINGS_MD_SECTION ::= NUL_TERMINATED_STRING*
```

The MLIRTYPE_STRINGS_MD section defines a table of MLIR type names used by the
program, in a format that can be parsed by BEF2MLIR.  These are referenced by
index into the section, not offset.

***Rationale***

Type names are not needed by the BEF executor, and are decoded in a single dense
pass up-front in BEF2MLIR.  As such, they are referenced by index (not offset).

### MLIRTYPE_MD Section

**Grammar**

```none
  MLIRTYPE_MD_SECTION ::= INDEX<"MLIRTYPE_STRINGS_MD">*
```

The MLIRTYPE_MD section is an optional section which is not needed for BEF
execution, but allows BEF2MLIR conversion to regenerate the correct types for
BEF function arguments and the result types of primitives.  This section
contains a list of VBR encoded indexes into the 
[MLIRTYPE_STRINGS_MD section](#TYPE_STRINGS_MD-section), and is encoded in
the same order as the function table.  Each function has a list of argument
types and a type for each primitive result.
