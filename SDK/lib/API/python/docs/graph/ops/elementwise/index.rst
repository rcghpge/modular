:description: Elementwise operations for the MAX Graph Python API
:title: elementwise
:type: module
:lang: python

Elementwise operations
======================

Operations that perform element-wise computations/comparisons.

Operations in this module are split into either unary or binary operations.

Unary Operations
----------------

* Elementwise-unary-operations all have the following properties:

  * They operate on a single symbolic tensor value of any shape
  * Their output is a single symbolic tensor value with the same shape as their
    input
  * The computation they represent will be itemwise-independent, in other words
    the output value in any position of the output tensor at computation time
    will depend only on the input value at that same position, and no others.

Binary Operations
-----------------

* Elementwise-binary-operations all have the following properties:

  * They operate on two symbolic tensor values, a ``left`` value and a ``right``
    value.
  * The input tensor types must be compatible according to the
    :ref:`broadcasting-rules-target`.
  * If the input tensor types have different element types, they will each
    be *promoted* to the same dtype according to the
    :ref:`dtype-promotion-rules-target` *before* executing
    the operation. This may involve a cast that changes the representation
    (including precision) of the data values.
  * Their output is a single symbolic tensor value with dtype depending on the
    op and the *promoted* dtype, such as ``promote(lhs, rhs)``.shape equal to
    the result of ``broadcast(lhs, rhs)``
  * The computation they represent will be itemwise-independent, in other words
    *after broadcasting* the input values to the same shape, the output value in
    any position of the output tensor at computation time will depend only on
    the input position at the two broadcast input values at that same
    position, and no others.

.. _dtype-promotion-rules-target:

DType Promotion Rules
---------------------

The Graph API splits dtype promotion into two pieces: bit width and category.
Bit width is simply the number of bits that are needed to represent a dtype.
Category is an order hierarchy: ``bool < unsigned int < signed int < float``.

A promotion candidate is calculated between two dtypes (``a`` and ``b``) as:
``(max(category(a), category(b)), max(bitwidth(a), bitwidth(b)))``.

An exception will be raised if a either input dtype might contain a value that is
unrepresentable by the promotion candidate (e.g ``u32 -> i32`` or ``i32 -> f32``).

An exception will be raised if the input has the same bit width but a different format
than the promotion candidate (e.g. ``f16 -> bf16`` or ``f32 -> tf32``).

If no exception is raised, the promotion candidate is accepted.
All inputs will be cast to the promotion candidate before the underlying operation is run.

.. _broadcasting-rules-target:

Broadcasting Rules
------------------

Given two input tensor shapes, broadcasting works as following:

1. Prepend static 1 dimensions onto the tensor with lower rank to make it so that both tensors have the same rank.
2. If a dimension is a static 1 dimension, it will broadcast to the size of the dimension in the other tensor.
3. All other dimensions will be asserted to be equivalent. If they are not, an exception will be raised.

.. toctree::
   :maxdepth: 1

   abs
   add
   cos
   div
   equal
   erf
   exp
   floor
   gelu
   greater
   greater_equal
   is_inf
   is_nan
   log
   log1p
   logical_not
   logsoftmax
   max
   min
   mod
   mul
   negate
   not_equal
   pow
   relu
   round
   roundeven
   rsqrt
   sigmoid
   silu
   sin
   softmax
   sqrt
   sub
   tanh
   trunc