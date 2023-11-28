# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo -I %tokenizer_dir %s | FileCheck "%s"
from sentence_piece import BPETokenizer
from utils.vector import DynamicVector2


fn create_dummy_model() raises -> BPETokenizer:
    var model = BPETokenizer()
    model.add_unknown("<unk>")
    model.add_control("<s>")
    model.add_control("</s>")

    model.add_token("ab", 0.0)
    model.add_token("cd", -0.1)
    model.add_token("abc", -0.2)
    model.add_token("a", -0.3)
    model.add_token("b", -0.4)
    model.add_token("c", -0.5)
    model.add_token("qr", -0.5)

    return model ^


fn test_tokenization() raises:
    let model = create_dummy_model()

    let encoded = model.encode("xabcabaabcdd")
    let tokens = encoded.get_tokens()

    var expected = DynamicVector[StringRef]()
    expected.push_back("x")
    expected.push_back("abc")
    expected.push_back("ab")
    expected.push_back("a")
    expected.push_back("ab")
    expected.push_back("cd")
    expected.push_back("d")

    # CHECK: True
    print(len(tokens) == len(expected))

    for i in range(len(tokens)):
        # CHECK-NOT: False
        print(tokens[i] == expected[i])

    _ = model ^


fn test_decode() raises:
    let model = create_dummy_model()

    let encoded = model.encode("abcab")
    let token_ids = encoded.get_token_ids()

    # CHECK: abcab
    print(model.decode(token_ids))

    _ = model ^


fn main() raises:
    test_tokenization()
    test_decode()
