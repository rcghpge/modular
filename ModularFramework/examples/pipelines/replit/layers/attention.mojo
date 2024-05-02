# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from max.graph import ops, Dim, TensorType, Symbol
from max.graph.error import error
from math.limit import min_finite
from tensor import Tensor, TensorShape
from weights.hyperparams import HyperParams
from layers.linear import Linear
from collections import Optional


fn permute(input: Symbol, perm: Symbol, out_dims: List[Dim]) raises -> Symbol:
    """Permutes the dimensions of an input Tensor."""
    var g = input.graph()
    return g.op(
        "mo.transpose",
        List[Symbol](input, perm),
        TensorType(input.tensor_type().dtype, out_dims),
    )


@always_inline
fn tril(input: Symbol, k: Int = 0) raises -> Symbol:
    """Gets the bottom triangle of the input.

    The upper triangle above the kth diagnoal is zero'd out.

    Args:
        input: The input tensor.
        k: The diagonal at which values at and below are True, and values
          above are False.

    Returns:
        A Dtype.bool matrix.

    Raises:
        If the input has rank < 2.
    """
    var g = input.graph()
    if input.tensor_type().rank() < 2:
        raise "Can't get tril of Tensor with rank < 2"
    var input_shape = ops.shape_of(input)
    var N = input_shape[-2]  # Number of rows
    var M = input_shape[-1]  # number of columns
    var mask = tri(N, M, g.scalar(Int64(k)))
    return input * mask


fn tri(rows: Symbol, cols: Symbol, k: Symbol) raises -> Symbol:
    """Returns a triangular mask matrix.

    Args:
        rows: Number of rows in the returned matrix.
        cols: Number of columns in the returned matrix.
        k: The diagonal at which values at and below are True, and values
          above are False.

    Returns:
        A Dtype.bool matrix.
    """
    var g = rows.graph()

    var N = rows  # Number of rows
    var M = cols  # number of columns
    var int_dtype = N.tensor_type().dtype

    var row = g.op(
        "mo.range",
        List[Symbol](
            g.scalar(0, int_dtype),
            N,
            g.scalar(1, int_dtype),
        ),
        TensorType(int_dtype, Dim.dynamic()),
    ).reshape(-1, 1)
    var col = g.op(
        "mo.range",
        List[Symbol](
            -k,
            M - k,
            g.scalar(1, int_dtype),
        ),
        TensorType(int_dtype, Dim.dynamic()),
    ).reshape(1, -1)
    return ops.greater_equal(row, col)


@value
struct GroupedQueryAttention:
    """An attention layer that uses an intermediate number of key-value heads.
    """

    var hyperparams: HyperParams
    var wkqv: Linear
    var out_proj: Linear

    fn __call__(
        self,
        input: Symbol,
        attn_bias: Optional[Symbol] = None,
        k_cache: Optional[Symbol] = None,
        v_cache: Optional[Symbol] = None,
        is_causal: Bool = True,
    ) raises -> (Symbol, Symbol, Symbol):
        """Builds the GQA layer.

        Args:
            input: Encoded inputs.
            attn_bias: An additive bias to apply to the attention weights.
            k_cache: Cached computed keys for previous tokens.
            v_cache: Cached computed values for previous tokens. If
              `k_cache` is defined, `v_cache` must be defined as well.
            is_causal: Whether to apply a mask to the attention layer to ensure
                that the output tokens are only based on past positions.
        Returns:
            Attention outputs, new k_cache, and new v_cache.
        Raises:
            Error when `v_cache` is not defined when `k_cache` is defined.
        """
        var g = input.graph()
        var n_heads = self.hyperparams.n_heads
        var kv_n_heads = self.hyperparams.kv_n_heads
        var d_model = self.hyperparams.d_model
        var batch_size = self.hyperparams.batch_size
        var head_dim = d_model // n_heads

        var qkv = self.wkqv(input)
        var split = ops.split[3](
            qkv,
            sizes=(d_model, kv_n_heads * head_dim, kv_n_heads * head_dim),
            axis=2,
        )
        var query = split[0]
        var key = split[1]
        var value = split[2]

        # Apply scaled dot product attention on the query, key and value.
        var q = query.reshape(batch_size, -1, n_heads, head_dim)
        var perm = g.constant(Tensor[DType.int32](TensorShape(4), 0, 2, 1, 3))
        var q_out_dims = List[Dim](1, n_heads, Dim.dynamic(), head_dim)
        q = permute(q, perm, q_out_dims)

        var k = key.reshape(batch_size, -1, kv_n_heads, head_dim)
        var k_perm = g.constant(Tensor[DType.int32](TensorShape(4), 0, 2, 3, 1))
        var k_out_dims = List[Dim](
            batch_size, kv_n_heads, head_dim, Dim.dynamic()
        )
        k = permute(k, k_perm, k_out_dims)

        var v_shape = g.constant(
            Tensor[DType.int32](TensorShape(4), 1, kv_n_heads, -1, 128)
        )
        var v = value.reshape(batch_size, -1, kv_n_heads, head_dim)
        var v_out_dims = List[Dim](
            batch_size, kv_n_heads, Dim.dynamic(), head_dim
        )
        v = permute(v, perm, v_out_dims)

        if k_cache:
            var k_cache_value = k_cache.value()[]
            k = ops.concat(List[Symbol](k_cache_value, k), axis=3)
            if not v_cache:
                raise error("v_cache cannot be None if k_cache is defined.")
            var v_cache_value = v_cache.value()[]
            v = ops.concat(List[Symbol](v_cache_value, v), axis=2)

        # Record the k and v into the cache. An extra dimension is added
        # so that all cached keys/values can be concatenated on that dimension.
        var k_cache_update = k.reshape(1, batch_size, kv_n_heads, head_dim, -1)
        var v_cache_update = v.reshape(1, batch_size, kv_n_heads, -1, head_dim)

        if kv_n_heads > 1 and kv_n_heads < n_heads:
            # Repeat interleave k and v to match the number of heads in the
            # query.
            var n_repeats = n_heads // kv_n_heads
            var repeats = g.constant(
                Tensor[DType.int32](
                    TensorShape(5), batch_size, 1, n_repeats, 1, 1
                )
            )
            k = k.reshape(batch_size, kv_n_heads, 1, head_dim, -1)
            k = g.op(
                "mo.tile",
                List[Symbol](k, repeats),
                TensorType(
                    k.tensor_type().dtype,
                    batch_size,
                    kv_n_heads,
                    n_repeats,
                    head_dim,
                    Dim.dynamic(),
                ),
            )
            k = k.reshape(batch_size, kv_n_heads * n_repeats, head_dim, -1)

            v = v.reshape(batch_size, kv_n_heads, 1, -1, head_dim)
            v = g.op(
                "mo.tile",
                List[Symbol](v, repeats),
                TensorType(
                    k.tensor_type().dtype,
                    batch_size,
                    kv_n_heads,
                    n_repeats,
                    Dim.dynamic(),
                    head_dim,
                ),
            )
            v = v.reshape(1, kv_n_heads * n_repeats, -1, head_dim)

        var softmax_scale = 1 / math.sqrt(d_model / n_heads)
        var attn_weight = (q @ k) * softmax_scale
        var s_q = ops.shape_of(q)[2]
        var s_k = ops.shape_of(k)[-1]

        if attn_bias:
            var bias = attn_bias.value()[]
            var attn_bias_shape = ops.shape_of(bias)
            var _s_q = ops.max(g.scalar(Int32(0)), attn_bias_shape[2] - s_q)
            var _s_k = ops.max(g.scalar(Int32(0)), attn_bias_shape[3] - s_k)
            bias = bias[:, :, _s_q:, _s_k:].rebind(
                Dim.dynamic(), Dim.dynamic(), 1, Dim.dynamic()
            )
            attn_weight = attn_weight + bias

        if is_causal and (not q.tensor_type().dims[2] == 1):
            # Apply a triangular mask to the attention weight so that in the
            # later matmul, each token in the ouput doesn't involve
            # information from future positions.
            var s = ops.max(s_q, s_k)
            var causal_mask = g.full[DType.bool](1, List[Symbol](s, s))
            causal_mask = tril(causal_mask)
            causal_mask = causal_mask[-s_q:, -s_k:].reshape(1, 1, s_q, s_k)

            var attn_weight_shape = ops.shape_of(attn_weight)

            var min_val = g.op(
                "mo.broadcast_to",
                List[Symbol](
                    g.scalar(min_finite[DType.float32]()), attn_weight_shape
                ),
                TensorType(
                    DType.float32,
                    Dim.dynamic(),
                    Dim.dynamic(),
                    Dim.dynamic(),
                    Dim.dynamic(),
                ),
            )
            causal_mask = g.op(
                "mo.broadcast_to",
                List[Symbol](causal_mask, attn_weight_shape),
                TensorType(
                    causal_mask.tensor_type().dtype,
                    Dim.dynamic(),
                    Dim.dynamic(),
                    Dim.dynamic(),
                    Dim.dynamic(),
                ),
            )
            attn_weight = g.op(
                "mo.select",
                List[Symbol](causal_mask, attn_weight, min_val),
                TensorType(
                    DType.float32,
                    Dim.dynamic(),
                    Dim.dynamic(),
                    Dim.dynamic(),
                    Dim.dynamic(),
                ),
            )

        attn_weight = ops.softmax(attn_weight)
        var out = attn_weight @ v
        var out_perm = g.constant(
            Tensor[DType.int32](TensorShape(4), 0, 2, 1, 3)
        )
        var out_dims = List[Dim](batch_size, Dim.dynamic(), n_heads, head_dim)
        out = permute(out, out_perm, out_dims)
        out = out.reshape(batch_size, -1, self.hyperparams.d_model)
        return self.out_proj(out), k_cache_update, v_cache_update
