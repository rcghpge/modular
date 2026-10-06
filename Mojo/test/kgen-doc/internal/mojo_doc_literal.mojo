# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# RUN: kgen-doc %s | FileCheck %s


from max.gpu.host.info import GPUInfo, AMDRDNAFamily

# CHECK: "name": "Radeon7600",
# CHECK: "value": "GPUInfo.from_family(AMDRDNAFamily, StringSpan(\"Radeon 7600\"), StringSpan(\"hip\"), StringSpan(\"gfx1102\"), SIMD(11), StringSpan(\"RDNA3\"), Int(32))"
comptime Radeon7600 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 7600",
    api="hip",
    arch_name="gfx1102",
    compute=11.0,
    version="RDNA3",
    sm_count=32,
)
