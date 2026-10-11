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

from std.math import inf, isinf, isnan, lgamma, nan
from std.testing import assert_equal, assert_true, TestSuite
from std.sys import CompilationTarget
from std.utils.numerics import FlushDenormals, FPUtils


# References are correctly rounded MPFR values. Near a zero of lgamma on the
# negative axis the result is tiny while the reflection terms are not; the
# zero tables check the floats nearest each zero, where any absolute error
# is largest relative to the result.
def _within_ulp[dtype: DType](got: Scalar[dtype], want: Scalar[dtype]) -> Bool:
    if isnan(got) or isinf(got) or isnan(want) or isinf(want):
        return False
    var got_bits = got._to_bits_signed()
    var want_bits = want._to_bits_signed()
    if (got_bits < 0) != (want_bits < 0):
        return False
    # Same-sign finite bit patterns span less than Int.MAX, so neither the
    # subtraction nor its absolute value can overflow.
    return abs(Int(got_bits) - Int(want_bits)) <= 2


# Independently rounded 600-digit mpmath values for bit-created signed
# subnormals and neighbors of the positive zeros at 1 and 2.
def _bit_refs[dtype: DType]() -> List[Tuple[UInt64, Float64]]:
    comptime if dtype == .float32:
        return [
            (UInt64(0x00000001), 103.2789306640625),
            (UInt64(0x00000200), 97.04060363769531),
            (UInt64(0x007FFFFF), 87.3365478515625),
            (UInt64(0x80000001), 103.2789306640625),
            (UInt64(0x80000200), 97.04060363769531),
            (UInt64(0x807FFFFF), 87.3365478515625),
            (UInt64(0x3F7FFFFF), 3.4404738613602603e-08),
            (UInt64(0x3F800001), -6.880945591092313e-08),
            (UInt64(0x3FFFFFFF), -5.039981587628972e-08),
            (UInt64(0x40000001), 1.0079966017428887e-07),
            (UInt64(0x3F7FF000), 0.0001409708202118054),
            (UInt64(0x3F800800), -0.00014087276940699667),
            (UInt64(0x3FFFF800), -0.00010319960711058229),
            (UInt64(0x40000400), 0.00010323805327061564),
        ]
    else:
        comptime assert dtype == .float64, "expected float32 or float64"
        return [
            (UInt64(0x0000000000000001), 744.4400719213812),
            (UInt64(0x0000000000004000), 734.7360113935421),
            (UInt64(0x000FFFFFFFFFFFFF), 708.3964185322641),
            (UInt64(0x8000000000000001), 744.4400719213812),
            (UInt64(0x8000000000004000), 734.7360113935421),
            (UInt64(0x800FFFFFFFFFFFFF), 708.3964185322641),
            (UInt64(0x3FEFFFFFFFFFFFFF), 6.408381213480008e-17),
            (UInt64(0x3FF0000000000001), -1.2816762426960008e-16),
            (UInt64(0x3FFFFFFFFFFFFFFF), -9.387698065543117e-17),
            (UInt64(0x4000000000000001), 1.8775396131086244e-16),
            (UInt64(0x3FEFFE0000000000), 0.00014097082187592238),
            (UInt64(0x3FF0010000000000), -0.00014087277616326634),
            (UInt64(0x3FFFFF0000000000), -0.00010319961029799209),
            (UInt64(0x4000008000000000), 0.00010323805136409636),
        ]


# (x, lgamma(x)) at the float32 nearest each negative zero of lgamma.
def _zeros_f32() -> List[Tuple[Float64, Float64]]:
    return [
        (-2.747682571411133, -1.4418168348129257e-07),
        (-2.4570248126983643, -1.1287842482943233e-07),
        (-3.955294370651245, 1.7780596408556448e-06),
        (-3.143580913543701, -1.9605461432092852e-07),
        (-4.991544723510742, 9.666771802585572e-06),
        (-4.039361953735352, -3.053971795452526e-06),
        (-5.998607635498047, 0.0001113230682676658),
        (-5.008218288421631, -1.4815573194937315e-05),
        (-6.9998016357421875, 0.0006440637516789138),
        (-6.00138521194458, 5.9716250689234585e-05),
        (-7.999975204467773, 0.00029724976047873497),
        (-7.0001983642578125, -0.0001555986818857491),
        (-8.000024795532227, 0.00019109305867459625),
    ]


def _zeros_f64() -> List[Tuple[Float64, Float64]]:
    return [
        (-2.7476826467274127, 1.733509244024501e-16),
        (-2.4570247382208006, 5.619192358950097e-17),
        (-3.955294284858598, -4.14382750757705e-16),
        (-3.14358088834998, 1.6978655906121085e-15),
        (-4.991544640560048, 1.7683619350849613e-14),
        (-4.039361839740537, -5.664578074060335e-15),
        (-5.998607480080875, -2.3721063667118474e-13),
        (-5.0082181683225935, 5.4188509265538106e-15),
        (-6.999801507890638, 5.313011065735902e-14),
        (-6.001385294453155, -4.6433836942838004e-14),
        (-7.999975197095821, -2.1213071311827735e-12),
        (-7.000198333407325, -1.263203743493977e-12),
        (-8.999997244250977, -8.050618056741812e-11),
        (-8.000024800270682, 1.75595561986039e-11),
        (-9.99999972442663, 1.7719543958825935e-09),
        (-9.000002755714823, 3.444263328391509e-11),
        (-10.99999997494789, 7.92108817813105e-09),
        (-10.000000275573013, 1.2668051387565237e-09),
        (-11.999999997912324, -4.799928255018621e-08),
        (-11.000000025052106, 2.734640389948086e-08),
        (-12.99999999983941, 4.201542925270726e-06),
        (-12.000000002087676, -5.854619992113373e-08),
        (-13.99999999998853, 7.057210284625434e-05),
        (-13.00000000016059, 4.200706921268597e-06),
        (-14.999999999999236, 0.0011552549109141656),
        (-14.00000000001147, 7.057204149708364e-05),
        (-15.999999999999952, -0.0034851246475472554),
        (-15.000000000000764, 0.0011552549067268103),
    ]


# (x, lgamma(x)) next to negative integers and at half-integers.
def _near_f32() -> List[Tuple[Float64, Float64]]:
    return [
        (-0.5, 1.2655121088027954),
        (-1.5, 0.8600470423698425),
        (-2.5, -0.056243717670440674),
        (-100.5, -364.9009704589844),
        (-1e-08, 18.42068099975586),
        (-7.000202178955078, -0.019211480394005775),
        (-8.99999713897705, -0.037489715963602066),
        (-9.00000286102295, -0.03750259801745415),
        (-0.9990000128746033, 6.908192157745361),
        (-1.0010000467300415, 6.907287120819092),
        (-0.9999989867210388, 13.802319526672363),
        (-1.0000009536743164, 13.862943649291992),
        (-1.9989999532699585, 6.215485572814941),
        (-2.000999927520752, 6.213759422302246),
        (-1.9999990463256836, 13.16979694366455),
        (-2.0000009536743164, 13.169795989990234),
        (-2.999000072479248, 5.117325782775879),
        (-3.000999927520752, 5.114813804626465),
        (-2.9999990463256836, 12.071185111999512),
        (-3.0000009536743164, 12.071183204650879),
        (-4.999000072479248, 2.1220436096191406),
        (-5.000999927520752, 2.118631601333618),
        (-4.999999046325684, 9.075453758239746),
        (-5.000000953674316, 9.07544994354248),
        (-6.999000072479248, -1.6153165102005005),
        (-7.000999927520752, -1.6193475723266602),
        (-6.999999046325684, 5.337784290313721),
        (-7.000000953674316, 5.337780475616455),
        (-9.99899959564209, -8.194706916809082),
        (-10.00100040435791, -8.19941234588623),
        (-9.999999046325684, -1.241466760635376),
        (-10.000000953674316, -1.2414711713790894),
        (-13.99899959564209, -18.281192779541016),
        (-14.00100040435791, -18.286544799804688),
        (-13.999999046325684, -11.328274726867676),
        (-14.000000953674316, -11.328280448913574),
    ]


def _near_f64() -> List[Tuple[Float64, Float64]]:
    return [
        (-0.5, 1.2655121234846454),
        (-1.5, 0.860047015376481),
        (-2.5, -0.056243716497674054),
        (-100.5, -364.90096830942736),
        (-1e-08, 18.42068074972452),
        (-7.0002022, -0.019315607864957115),
        (-13.999999999988846, 0.028024653614358217),
        (-16.999999999999996, -0.23400878325950386),
        (-16.000000000000046, 0.03425520333503583),
        (-17.000000000000004, -0.2340087832595242),
        (-0.999, 6.908179385717436),
        (-1.001, 6.907333817182055),
        (-0.999999, 13.815510980721177),
        (-1.000001, 13.815510135263528),
        (-0.999999999999, 27.63104323789378),
        (-1.000000000001, 27.630932219297208),
        (-1.999, 6.215532330199284),
        (-2.001, 6.213686761580459),
        (-1.999999, 13.122364300272379),
        (-2.000001, 13.122362454481664),
        (-1.999999999999, 26.937785038738607),
        (-2.000000000001, 26.937785038736763),
        (-2.999, 5.117253430432412),
        (-3.001, 5.114741195122229),
        (-2.999999, 12.023752344715612),
        (-3.000001, 12.023749832480275),
        (-2.999999999999, 25.83917275007083),
        (-3.000000000001, 25.839172750068318),
        (-4.999, 2.121971208135854),
        (-5.001, 2.1185589728099194),
        (-4.999999, 9.028020521161674),
        (-5.000001, 9.028017108926337),
        (-4.999999999999, 22.84344047651729),
        (-5.000000000001, 22.843440476513877),
        (-6.999, -1.6153888622425043),
        (-7.001, -1.619420145192517),
        (-6.999999, 5.290351212402139),
        (-7.000001, 5.290347181119182),
        (-6.999999999999, 19.10577085823423),
        (-7.000000000001, 19.1057708582302),
        (-9.999, -8.194303944153827),
        (-10.001, -8.19900744932894),
        (-9.999999, -1.2888996626086562),
        (-10.000001, -1.2889043661138306),
        (-9.999999999999, 12.526519646224466),
        (-10.000000000001, 12.526519646219763),
        (-13.999, -18.28078994662963),
        (-14.001, -18.286138639951364),
        (-13.999999, -11.375707949677738),
        (-14.000001, -11.375713298371057),
        (-13.999999999999, 2.439711036561623),
        (-14.000000000001, 2.4397110365562735),
    ]


def _check_zeros[
    dtype: DType
](refs: List[Tuple[Float64, Float64]]) raises where dtype.is_floating_point():
    for r in refs:
        var x = Scalar[dtype](r[0])
        var want = Scalar[dtype](r[1])
        var got = lgamma(x)
        assert_true(
            _within_ulp(got, want),
            String(t"lgamma({x}) = {got}, exact = {want}"),
        )
        var mixed = lgamma(SIMD[dtype, 4](0.5, x, -1.5, 40.0))
        assert_true(_within_ulp(mixed[1], want))
        assert_equal(mixed[1], got)
        assert_equal(mixed[0], lgamma(Scalar[dtype](0.5)))
        assert_equal(mixed[2], lgamma(Scalar[dtype](-1.5)))
        assert_equal(mixed[3], lgamma(Scalar[dtype](40.0)))


def _check_near[
    dtype: DType
](refs: List[Tuple[Float64, Float64]]) raises where dtype.is_floating_point():
    for r in refs:
        var x = Scalar[dtype](r[0])
        var want = Scalar[dtype](r[1])
        var got = lgamma(x)
        assert_true(
            _within_ulp(got, want),
            String(t"lgamma({x}) = {got}, exact = {want}"),
        )


def _check_specials[dtype: DType]() raises where dtype.is_floating_point():
    comptime S = Scalar[dtype]
    for x in [
        S(0.0),
        S(-0.0),
        S(-1.0),
        S(-2.0),
        S(-7.0),
        S(-1e6),
        S(-8388608.0),
        inf[dtype](),
        -inf[dtype](),
    ]:
        assert_equal(lgamma(x), inf[dtype](), String(t"lgamma({x})"))
    assert_true(isnan(lgamma(nan[dtype]())))
    assert_equal(lgamma(S(1.0)), 0.0)
    assert_equal(lgamma(S(2.0)), 0.0)
    # Lanes of either sign, a pole and a NaN in one vector match the scalar
    # results.
    var v = SIMD[dtype, 8](-2.5, 3.5, -3.0, -0.25, 0.75, -17.25, 40.0, -1e-3)
    v[3] = nan[dtype]()
    var got = lgamma(v)
    comptime for k in range(8):
        var want = lgamma(v[k])
        if isnan(want):
            assert_true(isnan(got[k]))
        else:
            assert_equal(got[k], want, String(t"lane {k}"))


def test_lgamma_negative_zeros_float32() raises:
    _check_zeros[.float32](_zeros_f32())


def test_lgamma_negative_zeros_float64() raises:
    _check_zeros[.float64](_zeros_f64())


def test_lgamma_near_negative_integers_float32() raises:
    _check_near[.float32](_near_f32())


def test_lgamma_near_negative_integers_float64() raises:
    _check_near[.float64](_near_f64())


def _check_bits[dtype: DType]() raises where dtype.is_floating_point():
    comptime bits_dtype = FPUtils[dtype].uint_type
    for r in _bit_refs[dtype]():
        var x = Scalar[dtype](from_bits=r[0].cast[bits_dtype]())
        var want = Scalar[dtype](r[1])
        var got = lgamma(x)
        assert_true(
            _within_ulp(got, want),
            String(t"lgamma(bits={r[0]}) = {got}, exact = {want}"),
        )
        var mixed = lgamma(SIMD[dtype, 4](0.5, x, -1.5, 40.0))
        assert_true(_within_ulp(mixed[1], want))
        assert_equal(mixed[1], got)
        assert_equal(mixed[0], lgamma(Scalar[dtype](0.5)))
        assert_equal(mixed[2], lgamma(Scalar[dtype](-1.5)))
        assert_equal(mixed[3], lgamma(Scalar[dtype](40.0)))


def test_lgamma_bit_references() raises:
    _check_bits[.float32]()
    _check_bits[.float64]()


@inline(.never)
def _lgamma_runtime[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    return lgamma(x)


@inline(.never)
def _double_runtime[dtype: DType](x: Scalar[dtype]) -> type_of(x):
    return x + x


def _check_flush[dtype: DType]() raises where dtype.is_floating_point():
    comptime bits_dtype = FPUtils[dtype].uint_type
    var refs = _bit_refs[dtype]()
    for i in range(6):
        var x = Scalar[dtype](from_bits=refs[i][0].cast[bits_dtype]())
        var want = Scalar[dtype](refs[i][1])
        var doubled: Scalar[dtype]
        var got: Scalar[dtype]
        var mixed: SIMD[dtype, 4]
        with FlushDenormals():
            doubled = _double_runtime(x)
            got = _lgamma_runtime(x)
            mixed = _lgamma_runtime(SIMD[dtype, 4](0.5, x, -1.5, 40.0))
        var doubled_bits = doubled._to_bits_signed()
        assert_equal(
            doubled_bits & ~type_of(doubled_bits)(FPUtils[dtype].sign_mask()), 0
        )
        assert_true(
            _within_ulp(got, want),
            String(
                t"flushed lgamma(bits={refs[i][0]}) = {got}, exact = {want}"
            ),
        )
        assert_true(_within_ulp(mixed[1], want))
        assert_equal(mixed[1], got)
        assert_equal(mixed[0], lgamma(Scalar[dtype](0.5)))
        assert_equal(mixed[2], lgamma(Scalar[dtype](-1.5)))
        assert_equal(mixed[3], lgamma(Scalar[dtype](40.0)))


def test_lgamma_flushed_subnormals() raises:
    comptime if CompilationTarget.has_sse4() or CompilationTarget.has_neon():
        _check_flush[.float32]()
        _check_flush[.float64]()


def test_lgamma_ulp_guard() raises:
    # This wrong-sign value previously wrapped the shared signed distance
    # to Int.MIN and passed the negative-zero reference assertion.
    assert_true(
        not _within_ulp(
            Float64(-7.1615747515792536e16), Float64(5.619192358950097e-17)
        )
    )
    assert_true(
        not _within_ulp(
            Float32(-0.00010305408795829862), Float32(-0.00010319960711058229)
        )
    )
    assert_true(not _within_ulp(inf[.float64](), Float64.MAX_FINITE))
    assert_true(not _within_ulp(-inf[.float64](), -Float64.MAX_FINITE))
    assert_true(not _within_ulp(nan[.float64](), Float64(1)))
    assert_true(not _within_ulp(Float64(-0.0), Float64(0.0)))
    assert_true(_within_ulp(Float64(1), Float64(1.0000000000000002)))


def test_lgamma_specials() raises:
    _check_specials[.float32]()
    _check_specials[.float64]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
