# [Mojo] Checking constrained methods at overload resolution time

**March 25, 2025**
Status: Proposed, agreement to explore but not committed, not implemented.

**Sept 17, 2025**
Status: Updated to include param decl constraints, and renamed to "where".
Implementation has been scoped out and prioritized.

**Oct 1, 2025**
Status: Updated to remove the error message field. Implementation in progress.

This document explores adding “where” clauses to Mojo, a major missing
feature that will allow more safety, expressivity, and APIs that work better
for our users.

## Introduction

Generic programming has always been a core part of the Mojo library design
ecosystem, and it consists of three phases: 1) parse time, and 2) elaboration
time (which evaluates “comptime parameter expressions”) and 3) runtime. We aim
to move error detection earlier in this list by improving the type system, but
we eschew complexity because we don’t want to turn into other systems that
require (e.g.) a full constraint solver or interpreter built into the parser.

All this said, Mojo has a weakness, which is the lack of `where` clauses
that allow constraining method availability (**at parser time**) based on
static-ly known **parser-time** information. For example, we have methods like
this on SIMD (and thus on core aliases like `Int8`):

```mojo
fn _simd_construction_checks[type: DType, size: Int]():
    constrained[
        type is not DType.invalid, "simd type cannot be DType.invalid"
    ]()
    constrained[size.is_power_of_two(), "simd width must be power of 2"]()
    ...

struct SIMD[dtype: DType, size: Int]:
    @implicit
    fn __init__(out self, value: FloatLiteral):
        ...
        _simd_construction_checks[dtype, size]()
        constrained[
            dtype.is_floating_point(), "the SIMD type must be floating point"
        ]()

        <actual code>
```

This method allows one to construct a SIMD type with a float literal, like
`1.0` but checks at *elaboration time* that type `dtype` is a floating point
type, that `size` is a power of two, etc. This is deeply unfortunate for a few
reasons:

1. This wastes comptime by having the interpreter have to eval things like
   `_simd_construction_checks`.
2. This makes error messages on misuse much worse, because you get a stack trace
   out of the elaborator instead of a comptime error message.
3. This prevents defining ambiguous overload sets that are pruned at parse time
   based on conditional information.

The reason the last one matters is that there can be many ways to solve a
problem if you’re a library developer working with a closed set of overloads,
particularly if generic. Why does this matter? This allows **algorithm
selection** at **overload-resolution time**. You can choose implementation
based on type capabilities, or remove a candidate based on lack of capabilities
(to resolve an ambiguous candidate set):

```mojo
fn thing[size: Int](value: YourType[size])
  where size.is_power_of_2(): ...
fn thing[size: Int](value: YourType[size])
  where not size.is_power_of_2(): ...
```

Furthermore, we want to enable more generic and powerful libraries, which
require boolean conditions to be successful. It would be wonderful to be able
to express things like this (evaluated at parse time) - but note, this syntax
is not holy, it is just made up:

**Parameter relationships**: Express constraints between multiple type
parameters:

```mojo
fn convert[From:.., To:..](value: From) -> To where can_convert_from[From, To]:
```

**Property-based constraints**: Specify requirements beyond simple trait
conformance

```mojo
fn safe_divide[T](a: T, b: T) where (T instanceof Numeric and T.min_value() < 0):
```

There are many possibilities, this is a pretty important feature for us to have
for both expressiveness and user experience. This feature should be able to
remove many uses of `constrained` , directly improving QoI for each one.

## Concrete syntax design

We support `where` constraints in two places. Each serves different use cases.

### Inline parameter declaration constraints

Inline constraints are written next to parameter declarations. They are checked
as each parameter is bound, and immediately become part of the in-scope
assumptions that later parameter declarations can rely on them. This is
especially useful when constraints on earlier parameters enable later
parameters to take advantage of those assumptions.

Example:

```mojo
struct Matrix[
  m: Int where m > 0,
  n: Int where n > 0
]:
  ...

fn solve_linear_system[
  n: Int where n > 0,
  a: Matrix[n, n],       # can assume n > 0 here
  b: Vector[n],
]() -> Vector[n]:
  ...
```

Another example that references earlier parameters from the argument list:

```mojo
fn matmul[
  m: Int where m > 0,
  n: Int where n > 0,
  k: Int where k > 0,
](a: Matrix[m, k], b: Matrix[k, n]) -> Matrix[m, n]:
    ...
```

Inline constraints keep requirements next to their parameters, improving
readability and enabling left-to-right dependencies among parameters. You can
also use inline constraints on type parameters for structs:

```mojo
struct SIMD[
  dtype: DType where dtype is not DType.invalid,
  size: Int where size.is_power_of_two(),
]:
  ...
```

### Function-level trailing constraints

A trailing `where` clause after the result type and before the colon attaches
to the function type and is checked at overload resolution time. For the scope
of this design, we limit support to statically evaluatable constraints. That
means the trailing constraint is expressed in terms of parameter declarations,
but may also reference runtime arguments as long as only static components of
those arguments are accessed (e.g., member aliases and parameters). This is
useful when the constraint is only needed by the function itself and no other
parameter in the parameter signature depends on it.

Example:

```mojo
struct SIMD[dtype: DType, size: Int]:
  @implicit
  fn __init__(out self, value: FloatLiteral)
    requires dtype.is_floating_point():
      <actual code>
```

In both forms, a constraint takes a boolean expression.

Notice how this puts the constraints where they belong - put the constraints
for the SIMD type as a whole on the struct, and put the constraints for the
method on the method itself.

This capability is an “obviously good” thing, but all the questions revolve
around the implementation - how invasive is this, what are the limitations, and
does it require building an interpreter back into the parser just after we
excised it?

## Implementation approach

The major thing required (given our quality goals) is that we are able to
implement a conformance check - is this member a valid candidate of an overload
set - **at parse time**. We need to decide that (and report a good error
message if the only candidate fails the boolean predicate) at parser time
**without interpreting the code** (because the Mojo parser has no interpreter).
For the sake of this discussion, we only consider simple boolean expressions.

This has four main parts: 1) enable inline parameter constraints, 2) collect
function/method requirements, 3) propagate assumptions across declarations
(contextual invariants), and 4) perform symbolic checking during overload
resolution.

### Part #1: Inline Parameter Constraints

We extend parameter parsing and binding to accept and record `where`
constraints on parameters. These constraints are evaluated at parameter binding
time and are immediately added to the local invariant set so that subsequent
parameters can assume them. We store these constraints alongside the parameter
declaration (e.g., as a list of `TypedAttr`), and thread them into the current
context used by later parameters and nested regions.

### Part #2: Function/Method Constraints

Method constraints are pretty simple - the constraints for a function are the
union (with an ‘and’) of the function constraints and the enclosing struct
constraints:

```mojo
struct SomeThing[
  size: Int
    where size.is_odd()
    where size != 233,
]:

  fn thing(self) -> Int
     where size.is_prime():
```

In this example, calling `SomeThing.thing` requires `Self.size` to satisfy the
condition `size.is_odd() and size != 233 and size.is_prime()`. Note that this
builds on `@always_inline("builtin")` and the simplifications that
`ParamOperatorAttr` does, but does not utilize an interpreter. This means that
we’ll be able to inline and simplify very simple integer expressions (e.g. we
could simplify things like `size != 1 and size > 0` into just `size > 1` ) but
we can’t do symbolic manipulation of `size.is_prime() and size.is_even()` into
`size == 2`.

This behavior will be core to this feature - like our dependent type support,
we can have rough mastery over simple affine operations on basic types like
integers, float and bool, but just treat more complicated operations
symbolically: `ParamOperatorAttr` does know that `a.is_prime() and
a.is_prime()` canonicalizes to `a.is_prime()` because the trivially redundant
subexpressions.

How do we store this? Method and struct requirements should be stored as a new
list of `TypedAttr` on both function and struct declarations. This ensures
they’re serialized to modules etc. This is parser time only behavior, so these
do not need to be lowered to KGEN or later.

### Part #3: Contextual Invariants

Contextual invariants - something known true at the point in some code - is a
question asked by overload set resolution at some point in the program.
Consider an overly complicated example like:

```mojo
struct S[
  a: Int where pred1(a),
  b: Int,
  c: Int,
  d: Int,
]:

    fn some_method(self)
       where pred2(b):

       @parameter
       if pred3(c):

           fn nested()
             where pred4(d):
                # Checking at this point.
                some_callee(self)
```

We need to determine what are the invariants we know at the point of
`some_callee(self)`. This is determined by walking the MLIR region tree and
unioning the set of contextual invariants together with an “and” operation. In
this case, we know that the contextual invariant is `pred1(a) and pred2(b) and
pred3(c) and pred4(d)` because of the invariants on the struct, functions, and
parameter if.

### Part #4: *Symbolic* constraint checking at overload resolution time

Finally, given we have these two bits of information, we can use it at overload
resolution time. Overload resolution has to do a bunch of stuff (parameter
inference, implicit conversion checking etc) to determine if a candidate is
valid. At the end of checking, it then looks to see what the “contextual
invariant” boolean expression is, and the “function requirement” boolean
expression is. The candidate is considered valid if, the following is true:

```mojo
(contextual_invariant and function_requirement) == contextual_invariant
```

In English, this is saying that `function_requirement` doesn’t impose any novel
requirements that `contextual_invariant` doesn’t already encode.

But what is “truth” here and how do we determine this? The expressions may
themselves be conjunctions of nested subexpressions, may have unresolved
operands, and we don’t have an interpreter in the parser. To address this, we
just allow `ParamOperatorAttr` to canonicalize and simplify the expressions,
and use pointer equality of the resultant `TypedAttr`’s. If they are identical,
then they are known to be safe, if not, it should be rejected. I implemented
the requisite symbolic manipulation at the KGEN level ([in June
2022](https://github.com/modularml/modular/commit/9fcf5c859adb9e282378fbd37344a0c49cf2c895))
and we can make other new specific cases fancier as needed.

In the case of a rejection, we can do a bit more digging for better error
message quality: we can figure out which clause is failing and report the
failed constraint. For example, if we have something like `SIMD[f32, 17]` and
the following definition:

```mojo
struct SIMD[
    dtype: DType where dtype is not DType.invalid,
    size: Int where size.is_power_of_two(),
]:
```

The logical way for the compiler to check this is to build up a big conjunction
`(size.is_power_of_two() and dtype is not DType.invalid)` (and the
`is_power_of_two()` function will be inlined into subexpressions so it was be
much lower level) and then fold it and fail the whole expression - we want
overload checking to be efficient, because it is normal for some overload set
candidates to fail without the expression type checker failing overall.
However, if the whole set fails, we want to report the first failing condition.
This can be done by adding a new failure kind to `OverloadFitness` which error
emission uses.

## Logical extensions (not in scope for this proposal)

Once we have the core language feature in place, we can address various
extensions. Things that come to mind, but which aren’t fully scoped out:

1. We should implement a `T instanceof Trait` and `T == Type` boolean
   predicates, this would allow subsuming trait testing.

2. We should look at type rebinding so we can fully subsume the existing
   “conditional conformance” logic, without requiring a rebind in the function
   body:

```mojo
struct A[T: AnyType]:
    var elt : T

    # existing
    fn constrained[T2: Copyable](self: A[T2]):
         var elt_copy = self.elt # invoke copyinit

    # desired:
    fn constrained(self)
      where T instanceof Copyable:
         # need to know T is copyable even though declared AnyType
         var elt_copy = self.elt
```

1. Depending on how we implement #2, maybe we can make `@parameter if T
  instanceof SomeTrait` refine the value of T within the body of the `if`.
  Theoretically if we did this, we could eliminate the need for `rebind` in a
  lot of kernel code. I’m not sure if this is possible though.

2. Implement other special case attributes where we want to. For example, I’d
   love metatypes to be comparable and `Variant.__getitem__` ’s type list to be
   something like `where T in Ts` , which would be straightforward to
   implement once metatypes are comparable. We could also support things like
   `where T in [Float32, Float64]` when we fix up list literals.

3. Remove the existing CTAD conditional conformance stuff. This would be
wonderful, simplifying some fairly weird code.

4. ✅ I’d like to add implicit conversions to types like `UnsafePointer` to
   another `UnsafePointer` which are only valid if the OriginSet+mutability are
   a superset of the source ones. These things would now be simple enough to
   implement. NOTE: We have since figured out how to do this with the existing
   hack.

## Primary Limitation

The primary limitation of this proposal is that it doesn’t have a parser time
interpreter, and I think it is important that we evaluate these things at
parser time. This should be fine for most symbolic cases, but will have one
annoying limitation. Consider the following:

```mojo
struct X[A: Int]:
   fn example(self) where A.is_prime(): ...


fn test(value: X[2]):
    # Error, cannot symbolically evaluate '2.is_prime()' to a constant.
    value.example()

    @parameter
    if 2.is_prime(): # tell dumb mojo that 2 is prime.
       # This is ok.
       value.example()
```

Because we don’t have a parser time interpreter, there will be some “obvious”
cases that cannot be evaluated.

**When does this occur?**: This will occur with non-trivial functions like
`is_prime` and other general logic (e.g. you could have a comptime function
that computes a hash table, that won’t be a thing in this phase). On the other
hand, this won’t happen for a lot of the trivial cases - the
`@always_inline("builtin")` ones will always be foldable, so basic math and
simple predicates like `is_power_of_two()` are all fine. Additionally,
parametric aliases can/will be used to expression generic and guaranteed
foldable expressions - but you can’t do a loop or complicated logic needed for
“is_prime” in an alias.

**What can we do about it?**

This could be annoying for people pushing the limits of the dependent types and
conditional conformance feature, there will always be limitations to symbolic
evaluation. If this becomes important in the future, we can consider bringing a
(different than we had) interpreter back into the parser, potentially building
on the work to make it support parameterized functions correctly.

Specifically, there is a proposal to make the comptime interpreter support
direct-interpreting parametric IR instead of relying on the elaborator to
specialize it before interpretation. If we had that (and a couple other minor
things) we could bring it back and make it “actually work and be predictable”
into the parser, and solve this in full generality.

We could also look to expand the support of `@always_inline("builtin")` for
specific narrow cases that need to be supported, but I think it is important
that we keep this pretty narrow for predictability reasons.

## Alternatives considered

### Syntax

There are multiple ways to spell this keyword, for example:

- `require` is not plural.
- Swift and Rust use `where`
- C++ uses `requires`

### Check at elaboration time

The one significant constraint in this proposal comes from checking
requirements at parser overload resolution time. One might ask “why not check
later, e.g. at elaboration time?

The problem is that it really is the parser that needs to determine which
concrete method is called, because this affects type checking. For example:

```mojo
fn your_function(a: SIMD[F32, _]) -> Int where a.size.is_prime(): ...
fn your_function(a: SIMD[F32, _]) -> F32: ...
```

We really do need to resolve (at parser time) which candidate gets picked,
because otherwise we don’t know the result type of a function call. You could
“support elaboration time checking” if the result types are consistent, but
that reduces the quality of the error messages (to include a stack trace on
failures) and is just sugar for a parameter if in the body.
