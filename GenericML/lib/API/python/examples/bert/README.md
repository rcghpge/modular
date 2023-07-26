# AI Engine Python API example

This is a simple API demo using the ["bert-base-uncased"
model](https://huggingface.co/bert-base-uncased) that does not perform any
meaningful inference. It's designed simply to show how you can run an inference
with the Modular AI Engine Python API.

First, you need to download the BERT model, which we've provided in a
TensorFlow SavedModel format here (run this command inside the `examples/bert/`
directory):

```sh
curl -L https://modular-model-storage-public.s3.amazonaws.com/bert-base-uncased.tar.gz \
| tar -xz
```

Then you can run an inference:

```sh
python3 main.py
```

For a walkthrough of this code, see the guide to [Get started in
Python](https://docs.modular.com/engine/python/get-started.html).

Also check out the [Python API
reference](https://docs.modular.com/engine/python/api/engine.html).

## BERT model conversion script

The BERT model in the above TAR package is from the [Hugging Face
"bert-base-uncased" model](https://huggingface.co/bert-base-uncased), converted
into the TensorFlow SavedModel format.

If you'd like, you can download the Hugging Face model yourself and convert
it into a SavedModel with the included Python script (which requires a
few other Python packages):

```sh
python3 -m pip install -r requirements.txt

python3 download-model.py
```
