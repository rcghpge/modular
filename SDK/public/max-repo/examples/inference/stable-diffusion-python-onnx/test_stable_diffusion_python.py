#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from pathlib import Path

from diffusers import PNDMScheduler
from max.engine import InferenceSession
from python.runfiles import runfiles
from text_to_image import parse, run_stable_diffusion
from transformers import CLIPTokenizer


# Look up runfile's runtime path
def find_runtime_path(model_path):
    r = runfiles.Create()
    path = r.Rlocation(model_path)

    if path is None:
        raise Exception(f"Runtime path for {model_path} was not found.")
    else:
        print(f"Runtime path for {model_path} was located at {path}")
    return Path(path)


def test_stable_diffusion_python():
    dir = "stable-diffusion-onnx"
    # Use same args as in `run.sh`.
    nprompt = """bad anatomy, looking away, looking sideways, crooked stick,
    stick not going through jaw, orange tongue"""
    pprompt = "Cute puppy chewing on a stick"

    args = parse(
        [
            "--prompt",
            pprompt,
            "--seed",
            "7",
            "--num-steps",
            "20",
            "--negative-prompt",
            nprompt,
        ]
    )

    # Compile & load models - this may take a few minutes.
    print("Loading and compiling models...")
    session = InferenceSession()

    text_encoder_path = find_runtime_path(f"{dir}/text_encoder/model.onnx")
    txt_encoder = session.load(text_encoder_path)

    img_decoder_path = find_runtime_path(f"{dir}/vae_decoder/model.onnx")
    img_decoder = session.load(img_decoder_path)

    img_diffuser_path = find_runtime_path(f"{dir}/unet/model.onnx")
    img_diffuser = session.load(img_diffuser_path)
    print("Models compiled.\n")

    tokenizer_path = find_runtime_path(f"{dir}/tokenizer")
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

    scheduler_path = find_runtime_path(f"{dir}/scheduler")
    scheduler = PNDMScheduler.from_pretrained(scheduler_path)

    run_stable_diffusion(
        args, txt_encoder, img_decoder, img_diffuser, tokenizer, scheduler
    )
