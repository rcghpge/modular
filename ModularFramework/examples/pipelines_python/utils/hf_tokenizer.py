# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Utilities for creating a HuggingFace tokenizer for a pipeline model."""

from typing import List, Optional

import gguf
import numpy as np
from gguf import GGUFReader, GGUFValueType, Keys
from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    pre_tokenizers,
    processors,
)
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast


def tokenizer_from_gguf(gguf_or_path) -> Tokenizer:
    reader = gguf_or_path
    if not isinstance(gguf_or_path, GGUFReader):
        reader = GGUFReader(gguf_or_path)

    architecture = _read_string(reader, gguf.KEY_GENERAL_ARCHITECTURE)
    if architecture != "llama":
        raise NotImplementedError(f"Unsupported GGUF architecture: {model}")

    model = _read_string(reader, Keys.Tokenizer.MODEL)
    if model != "gpt2":
        raise NotImplementedError(f"Unsupported tokenizer model: {model}")

    vocab_list = _read_string_array(reader, Keys.Tokenizer.LIST)
    if vocab_list is None:
        raise ValueError("Unable to find vocab list in GGUF weight file.")
    bpe_vocab = {token: n for n, token in enumerate(vocab_list)}

    merges = _read_string_array(reader, Keys.Tokenizer.MERGES)
    if merges:
        merges = [tuple(s.split(" ")) for s in merges]

    tokenizer = Tokenizer(
        BPE(
            bpe_vocab,
            merges,
            fuse_unk=False,
            byte_fallback=False,
            ignore_merges=True,
        )
    )

    # Mark special tokens.
    special_tokens = []
    bos_token = None
    bos_token_id = _read_int(reader, Keys.Tokenizer.BOS_ID)
    if bos_token_id is not None:
        bos_token = vocab_list[bos_token_id]
        special_tokens.append(
            AddedToken(bos_token, normalized=False, special=True)
        )

    eos_token = None
    eos_token_id = _read_int(reader, Keys.Tokenizer.EOS_ID)
    if eos_token_id is not None:
        eos_token = vocab_list[eos_token_id]
        special_tokens.append(
            AddedToken(vocab_list[eos_token_id], normalized=False, special=True)
        )
    token_type = _read_int_array(reader, Keys.Tokenizer.TOKEN_TYPE)
    if token_type:
        for i, type in enumerate(token_type):
            if type == LlamaTokenType.LLAMA_TOKEN_TYPE_CONTROL:
                special_tokens.append(
                    AddedToken(vocab_list[i], normalized=False, special=True)
                )

    # Note: special tokens do not increase the size of the vocabulary, since
    # they are already in the vocab list
    tokenizer.add_special_tokens(special_tokens)
    pretrained_tokenizer_kwargs = {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "add_prefix_space": True,
        "legacy": True,
        "clean_up_tokenization_spaces": True,
    }

    # Set up different processors.
    # These settings are not defined in the GGUF file, and so these are manually
    # pulled from:
    # https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/raw/main/tokenizer.json
    tokenizer.normalizers = None
    pattern = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(pattern), behavior="isolated", invert=False
            ),
            pre_tokenizers.ByteLevel(
                add_prefix_space=False, trim_offsets=True, use_regex=False
            ),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel(
        add_prefix_space=True, trim_offsets=True, use_regex=True
    )
    bos = bos_token
    add_bos_token = True
    eos = eos_token
    add_eos_token = False
    single = (
        f"{(bos+':0 ') if add_bos_token else ''}$A:0{(' '+eos+':0') if add_eos_token else ''}"
    )
    pair = (
        f"{single}{(' '+bos+':1') if add_bos_token else ''} $B:1{(' '+eos+':1') if add_eos_token else ''}"
    )
    special_tokens = [(bos_token, bos_token_id)]
    tokenizer.post_processor = processors.TemplateProcessing(
        single=single, pair=pair, special_tokens=special_tokens
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, **pretrained_tokenizer_kwargs
    )


def _to_string(arr: np.ndarray) -> str:
    return arr.tobytes().decode()


def _read_string(reader, key) -> Optional[str]:
    field = reader.get_field(key)
    if field is None:
        return None
    assert field.types[0] == GGUFValueType.STRING
    return _to_string(field.parts[field.data[0]])


def _read_string_array(reader, key) -> Optional[List[str]]:
    field = reader.get_field(key)
    if field is None:
        return None
    assert field.types[0] == GGUFValueType.ARRAY
    assert field.types[1] == GGUFValueType.STRING
    return [_to_string(field.parts[x]) for x in field.data]


def _read_int(reader, key) -> Optional[int]:
    field = reader.get_field(key)
    if field is None:
        return None
    return field.parts[field.data[0]][0]


def _read_int_array(reader, key) -> Optional[List[int]]:
    field = reader.get_field(key)
    if field is None:
        return None
    return [field.parts[x][0] for x in field.data]


# Llama token types from
# https://github.com/ggerganov/llama.cpp/blob/master/include/llama.h
class LlamaTokenType:
    LLAMA_TOKEN_TYPE_UNDEFINED = 0
    LLAMA_TOKEN_TYPE_NORMAL = 1
    LLAMA_TOKEN_TYPE_UNKNOWN = 2
    LLAMA_TOKEN_TYPE_CONTROL = 3
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4
    LLAMA_TOKEN_TYPE_UNUSED = 5
    LLAMA_TOKEN_TYPE_BYTE = 6
