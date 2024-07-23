# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
@value
struct HyperParams:
    var batch_size: Int
    var seq_len: Int
    var n_heads: Int
    var causal: Bool
    var alibi: Bool
    var alibi_bias_max: Int
    var num_blocks: Int
    var vocab_size: Int
    var d_model: Int
    var kv_n_heads: Int


def get_default() -> HyperParams:
    return HyperParams(
        batch_size=1,
        seq_len=4096,
        n_heads=24,
        causal=True,
        alibi=True,
        alibi_bias_max=8,
        num_blocks=32,
        vocab_size=32768,
        d_model=3072,
        kv_n_heads=8,
    )
