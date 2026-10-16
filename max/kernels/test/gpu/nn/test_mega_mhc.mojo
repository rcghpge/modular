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
"""Checks `mega_mhc` against a float64 host reference on every dispatch path.

The token counts walk all three kernel cutovers: H=5120 switches from decode
to mid-T at 17 tokens, shifted H=4096 at 65, and normal H=4096 stays on
decode through 128; every variant switches to the large-T kernel at 129. The
counts also reach every split count of the large-T kernel, including the
9-split shifted H=5120 wave. Each test reuses one counter buffer across all
its token counts, as a model would. H=2048 runs the fallback kernels.

A separate canary covers the boundary the float64 reference cannot afford:
the fused path's token limit, where its arrival counters run out of m-blocks,
and the first fallback count past it, checked on sampled tokens only.
"""

from std.math import exp, sqrt
from std.memory import bitcast
from layout import TileTensor, row_major
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from max.gpu.host.info import _is_sm10x_gpu
from nn.mega_mhc import (
    MEGA_MHC_COUNTERS,
    _MAX_T,
    mega_mhc,
    mega_mhc_scratch_size,
)
from nn.mega_mhc.common import _NUM_SMS
from std.testing import TestSuite, assert_true

comptime BF = DType.bfloat16
comptime F32 = DType.float32
comptime HC = 4
comptime NMIX = 24

# DeepGEMM's mega_mhc test values.
comptime HC_NORM_EPS = Float32(2e-5)
comptime HC_PRE_EPS = Float32(3e-4)
comptime HC_POST_SCALE = Float32(1.25)
comptime SINKHORN_EPS = Float32(2e-6)
comptime SINKHORN_ITERS = 10
comptime RMS_EPS = Float32(7e-6)
comptime RMS_SCALE = Float32(1.25)


def _mix64(x0: UInt64) -> UInt64:
    var x = x0
    x ^= x >> 30
    x *= 0xBF58476D1CE4E5B9
    x ^= x >> 27
    x *= 0x94D049BB133111EB
    x ^= x >> 31
    return x


def _uniform(stream: UInt64, i: Int) -> Float64:
    var h = _mix64(stream * 0xD1B54A32D192ED03 + UInt64(i))
    return Float64(h >> 11) * (1.0 / 9007199254740992.0)


def _normal(stream: UInt64, i: Int) -> Float64:
    """Irwin-Hall(4) rescaled to unit variance."""
    var acc = Float64(0)
    for j in range(4):
        acc += _uniform(stream, 4 * i + j)
    return (acc - 2.0) * 1.7320508075688772


def _sigmoid(x: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-x))


def _bf16(v: Float64) -> Float64:
    return Float64(Float32(v).cast[BF]().cast[F32]())


def _bf16_ulp(v: Float64) -> Float64:
    var bits = bitcast[DType.uint32, 1](Float32(abs(v)))
    var e = Int((bits >> 23) & 0xFF)
    if e == 0:
        return 1e-38
    return Float64(bitcast[F32, 1](UInt32(max(e - 7, 1)) << 23))


def _upload[
    dtype: DType
](ctx: DeviceContext, values: List[Float64]) raises -> DeviceBuffer[dtype]:
    var host = ctx.enqueue_create_host_buffer[dtype](len(values))
    for i in range(len(values)):
        host[i] = Float32(values[i]).cast[dtype]()
    var device = ctx.enqueue_create_buffer[dtype](len(values))
    ctx.enqueue_copy(device, host)
    return device^


def _download[
    dtype: DType
](ctx: DeviceContext, device: DeviceBuffer[dtype]) raises -> List[Float64]:
    var host = ctx.enqueue_create_host_buffer[dtype](len(device))
    ctx.enqueue_copy(host, device)
    ctx.synchronize()
    var values = List[Float64](capacity=len(device))
    for i in range(len(device)):
        values.append(Float64(host[i].cast[F32]()))
    return values^


def _accumulate(
    mut dot: List[Float64],
    mut norms: List[Float64],
    o: Int,
    a: Float64,
    b: Float64,
):
    dot[o] += a * b
    norms[o] += a * a + b * b


def _calc_diff(dot: Float64, norms: Float64) -> Float64:
    """DeepGEMM's `calc_diff`: 1 - 2<a, b> / (|a|^2 + |b|^2)."""
    # Only an exactly-zero norm is the empty case. A NaN norm must fall
    # through so an all-NaN kernel output fails the caller's tolerance.
    if norms == 0:
        return 0.0
    return 1.0 - 2.0 * dot / norms


def _check[
    hidden: Int, shifted: Bool
](
    ctx: DeviceContext, mut counters: DeviceBuffer[DType.uint64], count: Int
) raises:
    var x = List[Float64](capacity=count * hidden)
    for i in range(count * hidden):
        x.append(_bf16(_normal(1, i)))
    var residual = List[Float64](capacity=count * HC * hidden)
    for i in range(count * HC * hidden):
        residual.append(_bf16(_normal(2, i)))
    var post = List[Float64](capacity=count * HC)
    var previous_pre = List[Float64](capacity=count * HC)
    for i in range(count * HC):
        post.append(Float64(Float32(_sigmoid(_normal(3, i)))))
        previous_pre.append(Float64(Float32(_sigmoid(_normal(5, i)))))
    var comb = List[Float64](length=count * 16, fill=0)
    for t in range(count):
        for k in range(16):
            comb[t * 16 + k] = _uniform(4, t * 16 + k)
        for _ in range(3):
            for i in range(HC):
                var s = Float64(0)
                for j in range(HC):
                    s += comb[t * 16 + i * HC + j]
                for j in range(HC):
                    comb[t * 16 + i * HC + j] /= s
            for j in range(HC):
                var s = Float64(0)
                for i in range(HC):
                    s += comb[t * 16 + i * HC + j]
                for i in range(HC):
                    comb[t * 16 + i * HC + j] /= s
        for k in range(16):
            comb[t * 16 + k] = Float64(Float32(comb[t * 16 + k]))
    var projection = List[Float64](capacity=NMIX * HC * hidden)
    for i in range(NMIX * HC * hidden):
        projection.append(Float64(Float32(_normal(6, i) * 1e-2)))
    var scales = List[Float64](capacity=3)
    for i in range(3):
        scales.append(Float64(Float32(_normal(7, i) * 0.1)))
    var bias = List[Float64](capacity=NMIX)
    for i in range(NMIX):
        bias.append(Float64(Float32(_normal(8, i) * 0.1)))
    var weight = List[Float64](capacity=hidden)
    for i in range(hidden):
        weight.append(_bf16(1.0 + 0.1 * _normal(9, i)))

    var d_x = _upload[BF](ctx, x)
    var d_residual = _upload[BF](ctx, residual)
    var d_post = _upload[F32](ctx, post)
    var d_comb = _upload[F32](ctx, comb)
    var d_previous_pre = _upload[F32](ctx, previous_pre)
    var d_projection = _upload[F32](ctx, projection)
    var d_scales = _upload[F32](ctx, scales)
    var d_bias = _upload[F32](ctx, bias)
    var d_weight = _upload[BF](ctx, weight)
    var d_new_residual = ctx.enqueue_create_buffer[BF](count * HC * hidden)
    var d_new_pre = ctx.enqueue_create_buffer[F32](count * HC)
    var d_new_post = ctx.enqueue_create_buffer[F32](count * HC)
    var d_new_comb = ctx.enqueue_create_buffer[F32](count * 16)
    var d_output = ctx.enqueue_create_buffer[BF](count * hidden)
    var scratch_size = mega_mhc_scratch_size[hidden, shifted](count)
    var d_scratch = ctx.enqueue_create_buffer[F32](scratch_size)

    mega_mhc[hidden, shifted](
        TileTensor(d_new_residual, row_major(count * HC * hidden)),
        TileTensor(d_new_pre, row_major(count * HC)),
        TileTensor(d_new_post, row_major(count * HC)),
        TileTensor(d_new_comb, row_major(count * 16)),
        TileTensor(d_output, row_major(count * hidden)),
        TileTensor(d_x, row_major(count * hidden)),
        TileTensor(d_residual, row_major(count * HC * hidden)),
        TileTensor(d_post, row_major(count * HC)),
        TileTensor(d_comb, row_major(count * 16)),
        TileTensor(d_previous_pre, row_major(count * HC)),
        TileTensor(d_projection, row_major(NMIX * HC * hidden)),
        TileTensor(d_scales, row_major(3)),
        TileTensor(d_bias, row_major(NMIX)),
        TileTensor(d_weight, row_major(hidden)),
        TileTensor(counters, row_major(MEGA_MHC_COUNTERS)),
        TileTensor(d_scratch, row_major(scratch_size)),
        HC_NORM_EPS,
        HC_PRE_EPS,
        HC_POST_SCALE,
        SINKHORN_EPS,
        SINKHORN_ITERS,
        RMS_EPS,
        RMS_SCALE,
        ctx,
    )
    var got_residual = _download(ctx, d_new_residual)
    var got_pre = _download(ctx, d_new_pre)
    var got_post = _download(ctx, d_new_post)
    var got_comb = _download(ctx, d_new_comb)
    var got_output = _download(ctx, d_output)

    # The B200 large-T kernel projects in TF32. It is held to DeepGEMM's
    # bar for that path: calc_diff on every output, new_residual elementwise.
    var tf32 = count >= 129 and (hidden == 4096 or hidden == 5120)
    # new_residual, output, new_post, new_comb, new_pre
    var dot = List[Float64](length=5, fill=0)
    var norms = List[Float64](length=5, fill=0)
    var residual_err = Float64(0)
    var output_err = Float64(0)
    var mixed = List[Float64](length=HC * hidden, fill=0)
    var collapsed = List[Float64](length=hidden, fill=0)
    var exact = List[Float64](length=hidden, fill=0)
    var slack = List[Float64](length=hidden, fill=0)
    for t in range(count):
        var squares = Float64(0)
        for j in range(HC):
            for h in range(hidden):
                var v = x[t * hidden + h] * post[t * HC + j]
                var mag = abs(v)
                for i in range(HC):
                    var term = (
                        comb[t * 16 + i * HC + j]
                        * residual[(t * HC + i) * hidden + h]
                    )
                    v += term
                    mag += abs(term)
                var rounded = _bf16(v)
                mixed[j * hidden + h] = rounded
                squares += rounded * rounded
                var g = got_residual[(t * HC + j) * hidden + h]
                _accumulate(dot, norms, 0, g, rounded)
                # One BF16 rounding of the exact value, plus FP32 error.
                residual_err = max(
                    residual_err, abs(g - v) / (_bf16_ulp(v) + mag * 4e-7)
                )
        var rms = 1.0 / sqrt(
            squares / Float64(HC * hidden) + Float64(HC_NORM_EPS)
        )
        var m = List[Float64](length=NMIX, fill=0)
        # Normal mode collapses with the kernel's own Pre, which carries the
        # FP32 projection's rounding: about u * sqrt(K) of the summed term
        # magnitudes, times the sigmoid's slope of at most 1/4, with 8x
        # headroom.
        var pre_err = List[Float64](length=HC, fill=0)
        for k in range(NMIX):
            var acc = Float64(0)
            var size = Float64(0)
            for q in range(HC * hidden):
                var term = mixed[q] * projection[k * HC * hidden + q]
                acc += term
                size += abs(term)
            var scale = scales[0 if k < HC else (1 if k < 2 * HC else 2)]
            m[k] = acc * rms * scale + bias[k]
            if k < HC:
                pre_err[k] = (
                    0.25
                    * 8.0
                    * 5.96e-8
                    * sqrt(Float64(HC * hidden))
                    * (size * rms * abs(scale))
                )
        var pre = List[Float64](length=HC, fill=0)
        for j in range(HC):
            pre[j] = _sigmoid(m[j]) + Float64(HC_PRE_EPS)
            _accumulate(
                dot,
                norms,
                2,
                got_post[t * HC + j],
                _sigmoid(m[HC + j]) * Float64(HC_POST_SCALE),
            )
            if shifted:
                _accumulate(dot, norms, 4, got_pre[t * HC + j], pre[j])
        var c = List[Float64](length=16, fill=0)
        var eps = Float64(SINKHORN_EPS)
        for i in range(HC):
            var top = m[2 * HC + i * HC]
            for j in range(1, HC):
                top = max(top, m[2 * HC + i * HC + j])
            var s = Float64(0)
            for j in range(HC):
                c[i * HC + j] = exp(m[2 * HC + i * HC + j] - top)
                s += c[i * HC + j]
            for j in range(HC):
                c[i * HC + j] = c[i * HC + j] / s + eps
        for it in range(SINKHORN_ITERS):
            if it > 0:
                for i in range(HC):
                    var s = Float64(0)
                    for j in range(HC):
                        s += c[i * HC + j]
                    for j in range(HC):
                        c[i * HC + j] /= s + eps
            for j in range(HC):
                var s = Float64(0)
                for i in range(HC):
                    s += c[i * HC + j]
                for i in range(HC):
                    c[i * HC + j] /= s + eps
        for k in range(16):
            _accumulate(dot, norms, 3, got_comb[t * 16 + k], c[k])
        # Against the reference new_residual for calc_diff; against the
        # kernel's own for the elementwise bound, so a one-ulp new_residual
        # rounding flip does not count against the collapse and RMSNorm.
        for which in range(2):
            var norm_squares = Float64(0)
            for h in range(hidden):
                var v = Float64(0)
                var mag = Float64(0)
                var gate_err = Float64(0)
                for j in range(HC):
                    var coeff = previous_pre[t * HC + j] if shifted else pre[j]
                    var stream = mixed[j * hidden + h] if which == 0 else (
                        got_residual[(t * HC + j) * hidden + h]
                    )
                    v += coeff * stream
                    mag += abs(coeff * stream)
                    if not shifted:
                        gate_err += pre_err[j] * abs(stream)
                collapsed[h] = _bf16(v)
                exact[h] = v
                slack[h] = 2e-6 * mag + gate_err
                norm_squares += collapsed[h] * collapsed[h]
            var rstd = 1.0 / sqrt(
                norm_squares / Float64(hidden) + Float64(RMS_EPS)
            )
            for h in range(hidden):
                var g = got_output[t * hidden + h]
                var w = weight[h] * Float64(RMS_SCALE)
                if which == 0:
                    _accumulate(
                        dot, norms, 1, g, _bf16(collapsed[h] * rstd * w)
                    )
                else:
                    # One BF16 rounding of the collapse and of y, plus FP32
                    # error.
                    var y = exact[h] * rstd * w
                    var tol = _bf16_ulp(y) + abs(rstd * w) * (
                        _bf16_ulp(exact[h]) + slack[h]
                    )
                    output_err = max(output_err, abs(g - y) / tol)

    # DeepGEMM's test_mega_mhc bar.
    for o in range(5 if shifted else 4):
        assert_true(
            _calc_diff(dot[o], norms[o]) < 5e-5,
            String(
                "hidden ",
                hidden,
                " shifted ",
                shifted,
                " T ",
                count,
                ": output ",
                o,
                " calc_diff ",
                _calc_diff(dot[o], norms[o]),
            ),
        )
    assert_true(
        residual_err <= 1.0,
        String("T ", count, ": new_residual error/tolerance ", residual_err),
    )
    if not tf32:
        assert_true(
            output_err <= 1.0,
            String("T ", count, ": output error/tolerance ", output_err),
        )


def _check_all[hidden: Int, shifted: Bool](counts: List[Int]) raises:
    with DeviceContext() as ctx:
        var counters = ctx.enqueue_create_buffer[DType.uint64](
            MEGA_MHC_COUNTERS
        )
        counters.enqueue_fill(0)
        for count in counts:
            _check[hidden, shifted](ctx, counters, count)


def test_h4096_normal() raises:
    _check_all[4096, False]([1, 16, 17, 64, 65, 128, 129, 256, 320, 1024])


def test_h4096_shifted() raises:
    _check_all[4096, True]([1, 16, 17, 64, 65, 128, 129, 256, 320, 1024])


def test_h5120_normal() raises:
    _check_all[5120, False]([1, 16, 17, 33, 64, 128, 129, 256, 320, 384, 1024])


def test_h5120_shifted() raises:
    _check_all[5120, True]([1, 16, 17, 33, 64, 128, 129, 256, 320, 384, 1024])


def test_fallback() raises:
    _check_all[2048, False]([1, 77, 300])
    _check_all[2048, True]([1, 77, 300])


def _canary_value(seed: Int, i: Int, scale: Float32) -> Float64:
    """Returns the exact `Float32` value uploaded at element `i`.

    Recomputable per element, so the boundary canary never materializes a
    16384-token input on the host.
    """
    var cell = Float64((i * 17 + seed * 13) % 97 - 48) / 64
    return Float64(Float32(cell) * scale)


def _upload_canary[
    dtype: DType
](
    ctx: DeviceContext, count: Int, seed: Int, scale: Float32
) raises -> DeviceBuffer[dtype]:
    var host = ctx.enqueue_create_host_buffer[dtype](count)
    for i in range(count):
        host[i] = Float32(_canary_value(seed, i, scale)).cast[dtype]()
    var device = ctx.enqueue_create_buffer[dtype](count)
    ctx.enqueue_copy(device, host)
    return device^


def _check_boundary[
    hidden: Int, shifted: Bool
](ctx: DeviceContext, count: Int, expect_fused: Bool) raises:
    """Checks `mega_mhc` at `count` without the full float64 reference cost.

    Samples the first, middle and last token — the last sits in the arrival
    counters' final m-block — and, unlike `_check`, also proves which kernel
    ran: only the fused kernels leave the counters nonzero.
    """
    var x = _upload_canary[BF](ctx, count * hidden, 11, 1)
    var residual = _upload_canary[BF](ctx, count * HC * hidden, 12, 1)
    var post = _upload_canary[F32](ctx, count * HC, 13, 1)
    var comb = _upload_canary[F32](ctx, count * 16, 14, 1)
    var previous_pre = _upload_canary[F32](ctx, count * HC, 15, 1)
    var projection = _upload_canary[F32](ctx, NMIX * HC * hidden, 16, 0.01)
    var scales = _upload_canary[F32](ctx, 3, 17, 0.1)
    var bias = _upload_canary[F32](ctx, NMIX, 18, 0.1)
    var weight = _upload_canary[BF](ctx, hidden, 19, 1)
    var new_residual = ctx.enqueue_create_buffer[BF](count * HC * hidden)
    var new_pre = ctx.enqueue_create_buffer[F32](count * HC)
    var new_post = ctx.enqueue_create_buffer[F32](count * HC)
    var new_comb = ctx.enqueue_create_buffer[F32](count * 16)
    var output = ctx.enqueue_create_buffer[BF](count * hidden)
    var counters = ctx.enqueue_create_buffer[DType.uint64](MEGA_MHC_COUNTERS)
    counters.enqueue_fill(0)
    var scratch_size = mega_mhc_scratch_size[hidden, shifted](count)
    var scratch = ctx.enqueue_create_buffer[F32](scratch_size)

    mega_mhc[hidden, shifted](
        TileTensor(new_residual, row_major(count * HC * hidden)),
        TileTensor(new_pre, row_major(count * HC)),
        TileTensor(new_post, row_major(count * HC)),
        TileTensor(new_comb, row_major(count * 16)),
        TileTensor(output, row_major(count * hidden)),
        TileTensor(x, row_major(count * hidden)),
        TileTensor(residual, row_major(count * HC * hidden)),
        TileTensor(post, row_major(count * HC)),
        TileTensor(comb, row_major(count * 16)),
        TileTensor(previous_pre, row_major(count * HC)),
        TileTensor(projection, row_major(NMIX * HC * hidden)),
        TileTensor(scales, row_major(3)),
        TileTensor(bias, row_major(NMIX)),
        TileTensor(weight, row_major(hidden)),
        TileTensor(counters, row_major(MEGA_MHC_COUNTERS)),
        TileTensor(scratch, row_major(scratch_size)),
        HC_NORM_EPS,
        HC_PRE_EPS,
        HC_POST_SCALE,
        SINKHORN_EPS,
        SINKHORN_ITERS,
        RMS_EPS,
        RMS_SCALE,
        ctx,
    )
    var got_pre = _download[F32](ctx, new_pre)
    var got_post = _download[F32](ctx, new_post)
    var got_comb = _download[F32](ctx, new_comb)
    var out_host = ctx.enqueue_create_host_buffer[BF](count * hidden)
    ctx.enqueue_copy(out_host, output)
    var ctr_host = ctx.enqueue_create_host_buffer[DType.uint64](
        MEGA_MHC_COUNTERS
    )
    ctx.enqueue_copy(ctr_host, counters)
    ctx.synchronize()

    var touched = False
    for i in range(MEGA_MHC_COUNTERS):
        if ctr_host[i] != UInt64(0):
            touched = True
    assert_true(
        touched == expect_fused,
        String(
            "T ",
            count,
            ": expected the ",
            "fused" if expect_fused else "fallback",
            " kernels, but the counters are ",
            "untouched" if not touched else "nonzero",
        ),
    )

    var ref_projection = List[Float64](capacity=NMIX * HC * hidden)
    for i in range(NMIX * HC * hidden):
        ref_projection.append(_canary_value(16, i, 0.01))
    var ref_scales = List[Float64](capacity=3)
    for i in range(3):
        ref_scales.append(_canary_value(17, i, 0.1))
    var ref_bias = List[Float64](capacity=NMIX)
    for i in range(NMIX):
        ref_bias.append(_canary_value(18, i, 0.1))

    # new_residual(0, unused), output(1), new_post(2), new_comb(3), new_pre(4)
    var dot = List[Float64](length=5, fill=0)
    var norms = List[Float64](length=5, fill=0)
    var mixed = List[Float64](length=HC * hidden, fill=0)
    var collapsed = List[Float64](length=hidden, fill=0)
    var m = List[Float64](length=NMIX, fill=0)
    var c = List[Float64](length=16, fill=0)
    var eps = Float64(SINKHORN_EPS)
    var samples = List[Int]()
    samples.append(0)
    samples.append(count // 2)
    samples.append(count - 1)
    for t in samples:
        var squares = Float64(0)
        for j in range(HC):
            for h in range(hidden):
                var v = _bf16(_canary_value(11, t * hidden + h, 1)) * (
                    _canary_value(13, t * HC + j, 1)
                )
                for i in range(HC):
                    v += _canary_value(14, t * 16 + i * HC + j, 1) * _bf16(
                        _canary_value(12, (t * HC + i) * hidden + h, 1)
                    )
                var rounded = _bf16(v)
                mixed[j * hidden + h] = rounded
                squares += rounded * rounded
        var rms = 1.0 / sqrt(
            squares / Float64(HC * hidden) + Float64(HC_NORM_EPS)
        )
        for k in range(NMIX):
            var acc = Float64(0)
            for q in range(HC * hidden):
                acc += mixed[q] * ref_projection[k * HC * hidden + q]
            var scale = ref_scales[0 if k < HC else (1 if k < 2 * HC else 2)]
            m[k] = acc * rms * scale + ref_bias[k]
        for j in range(HC):
            _accumulate(
                dot,
                norms,
                2,
                got_post[t * HC + j],
                _sigmoid(m[HC + j]) * Float64(HC_POST_SCALE),
            )
            if shifted:
                _accumulate(
                    dot,
                    norms,
                    4,
                    got_pre[t * HC + j],
                    _sigmoid(m[j]) + Float64(HC_PRE_EPS),
                )
        for i in range(HC):
            var top = m[2 * HC + i * HC]
            for j in range(1, HC):
                top = max(top, m[2 * HC + i * HC + j])
            var s = Float64(0)
            for j in range(HC):
                c[i * HC + j] = exp(m[2 * HC + i * HC + j] - top)
                s += c[i * HC + j]
            for j in range(HC):
                c[i * HC + j] = c[i * HC + j] / s + eps
        for it in range(SINKHORN_ITERS):
            if it > 0:
                for i in range(HC):
                    var s = Float64(0)
                    for j in range(HC):
                        s += c[i * HC + j]
                    for j in range(HC):
                        c[i * HC + j] /= s + eps
            for j in range(HC):
                var s = Float64(0)
                for i in range(HC):
                    s += c[i * HC + j]
                for i in range(HC):
                    c[i * HC + j] /= s + eps
        for k in range(16):
            _accumulate(dot, norms, 3, got_comb[t * 16 + k], c[k])
        var norm_squares = Float64(0)
        for h in range(hidden):
            var v = Float64(0)
            for j in range(HC):
                var coeff = _canary_value(
                    15, t * HC + j, 1
                ) if shifted else _sigmoid(m[j]) + Float64(HC_PRE_EPS)
                v += coeff * mixed[j * hidden + h]
            collapsed[h] = _bf16(v)
            norm_squares += collapsed[h] * collapsed[h]
        var rstd = 1.0 / sqrt(norm_squares / Float64(hidden) + Float64(RMS_EPS))
        for h in range(hidden):
            var w = _bf16(_canary_value(19, h, 1)) * Float64(RMS_SCALE)
            _accumulate(
                dot,
                norms,
                1,
                Float64(out_host[t * hidden + h].cast[F32]()),
                _bf16(collapsed[h] * rstd * w),
            )
    for o in range(1, 5 if shifted else 4):
        assert_true(
            _calc_diff(dot[o], norms[o]) < 5e-5,
            String(
                "hidden ",
                hidden,
                " shifted ",
                shifted,
                " T ",
                count,
                ": output ",
                o,
                " calc_diff ",
                _calc_diff(dot[o], norms[o]),
            ),
        )


def test_token_limit() raises:
    with DeviceContext() as ctx:
        # Mirrors `mega_mhc`'s dispatch: the fused kernels run only on an
        # sm10x GPU with the 148 SMs their arrival-counter schedule assumes.
        # Past `_MAX_T` every GPU takes the fallback.
        var fused = False
        comptime if _is_sm10x_gpu(ctx.default_device_info):
            fused = (
                ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
                == _NUM_SMS
            )
        _check_boundary[4096, False](ctx, _MAX_T, fused)
        _check_boundary[4096, False](ctx, _MAX_T + 1, False)
        _check_boundary[4096, True](ctx, _MAX_T, fused)
        _check_boundary[4096, True](ctx, _MAX_T + 1, False)


def main() raises:
    var suite = TestSuite()
    suite.test[test_h4096_normal]()
    suite.test[test_h4096_shifted]()
    suite.test[test_h5120_normal]()
    suite.test[test_h5120_shifted]()
    suite.test[test_fallback]()
    suite.test[test_token_limit]()
    suite^.run()
