target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @vec_add(ptr addrspace(1) %out,
                                    ptr addrspace(1) %in0,
                                    ptr addrspace(1) %in1,
                                    i64 %len) {
entry:
  ret void
}
