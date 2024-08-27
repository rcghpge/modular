:description: The MAX Python Graph API reference.
:title: MAX Python Graph API
:sidebar_label: overview

MAX Python Graph API
====================

APIs to quantize graph tensors.

This package includes a generic quantization encoding interface and some
quantization encodings that conform to it, such as bfloat16 and Q4_0 encodings.

The main interface for defining a new quantized type is
`QuantizationEncoding.quantize()`. This takes a full-precision tensor
represented as float32 and quantizes it according to the encoding. The
resulting quantized tensor is represented as a bytes tensor. For that reason,
the `QuantizationEncoding` must know how to translate between the tensor shape
and its corresponding quantized buffer shape.