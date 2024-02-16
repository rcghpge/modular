# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo -I %gentext_dir %s | FileCheck "%s"

from tokenizer.bpe import BPETokenizer
from collections.vector import DynamicVector


fn create_dummy_model() raises -> BPETokenizer:
    var model = BPETokenizer()

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

    let tokens = model.encode("xabcabaabcddxabcabaabcdd")

    var expected = DynamicVector[String]()
    expected.push_back("x")
    expected.push_back("abc")
    expected.push_back("ab")
    expected.push_back("a")
    expected.push_back("ab")
    expected.push_back("cd")
    expected.push_back("d")
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
        print(tokens[i].token == expected[i])


fn test_decode() raises:
    let model = create_dummy_model()

    var encoded = model.encode("qrcbaabccdabqrcbaabccdab")
    var decoded = String()
    for token in encoded:
        decoded += token[].token

    # CHECK: qrcbaabccdabqrcbaabccdab
    print(decoded)


fn main() raises:
    test_tokenization()
    test_decode()
