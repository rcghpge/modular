# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from max.dtype import DType
from max.graph import Graph, TensorType, ops
from max.graph.weights import SafetensorWeights
from max.pipelines import PipelineConfig
from nn import Conv2D, Linear, RMSNorm
from transformers import AutoConfig

from .attention import Attention
from .attention_utils import causal_attention_mask_2d
from .rotary_embedding_2d import RotaryEmbedding2D, patch_position_ids
from .transformer import MLP, Transformer, TransformerBlock
from .vision_encoder import VisionEncoder


def _patch_conv2d(
    dtype: DType,
    in_channels: int,
    patch_size: int,
    out_channels: int,
    weights: SafetensorWeights,
) -> Conv2D:
    """Creates a 2D convolution layer with the following assumptions:
    - kernel size = (patch_size, patch_size)
    - stride = (patch_size, patch_size)
    - padding = (0, 0, 0, 0)

    This convolution splits the image into patches and then learns an embedding
    of each patch. The embedding dim is out_channels.
    """
    # Loaded torch weights shape = torch.Size([1024, 3, 16, 16]).
    # Conv2D expects (height, width, in_channels, out_channels) = [16, 16, 3, 1024].
    weights = ops.permute(  # type: ignore
        weights.weight.allocate(
            dtype, [out_channels, in_channels, patch_size, patch_size], None
        ),
        [2, 3, 1, 0],
    )
    return Conv2D(
        weights,  # type: ignore
        stride=(patch_size, patch_size),
    )


def _linear(
    dtype: DType,
    in_features: int,
    out_features: int,
    weights: SafetensorWeights,
) -> Linear:
    return Linear(
        weights.weight.allocate(dtype, [in_features, out_features], None)
    )


def _feed_forward(
    dtype: DType,
    hidden_dim: int,
    feed_forward_length: int,
    weights: SafetensorWeights,
):
    return MLP(
        _linear(  # gate_proj
            dtype,
            feed_forward_length,
            hidden_dim,
            weights.feed_forward.w1,
        ),
        _linear(  # down_proj
            dtype,
            hidden_dim,
            feed_forward_length,
            weights.feed_forward.w2,
        ),
        _linear(  # up_proj
            dtype,
            feed_forward_length,
            hidden_dim,
            weights.feed_forward.w3,
        ),
    )


def _rms_norm(dims: int, eps: float, weights: SafetensorWeights) -> RMSNorm:
    return RMSNorm(weights.weight.allocate(DType.bfloat16, [dims]), eps)


def _encoder_attention(
    pipeline_config: PipelineConfig,
    model_config: AutoConfig,
    weights: SafetensorWeights,
) -> Attention:
    # TODO: Do we need to transpose weights? Not obvious from shapes. Both dims are the same.
    hidden_dim = model_config.vision_config.hidden_size
    wq = _linear(
        pipeline_config.dtype,
        hidden_dim,
        hidden_dim,
        weights.attention.wq,
    )
    wk = _linear(
        pipeline_config.dtype,
        hidden_dim,
        hidden_dim,
        weights.attention.wk,
    )
    wv = _linear(
        pipeline_config.dtype,
        hidden_dim,
        hidden_dim,
        weights.attention.wv,
    )
    wo = _linear(
        pipeline_config.dtype,
        hidden_dim,
        hidden_dim,
        weights.attention.wo,
    )

    return Attention(
        n_heads=model_config.vision_config.num_attention_heads,
        dim=hidden_dim,
        head_dim=hidden_dim // model_config.vision_config.num_attention_heads,
        dropout=model_config.vision_config.attention_dropout,
        wq=wq,
        wk=wk,
        wv=wv,
        wo=wo,
    )


def _transformer(
    graph: Graph,
    pipeline_config: PipelineConfig,
    model_config: AutoConfig,
    weights: SafetensorWeights,
):
    with graph:
        layers = [
            TransformerBlock(
                attention=_encoder_attention(
                    pipeline_config, model_config, weights.layers[i]
                ),
                mlp=_feed_forward(
                    pipeline_config.dtype,
                    model_config.vision_config.hidden_size,
                    model_config.vision_config.intermediate_size,
                    weights.layers[i],
                ),
                attention_norm=_rms_norm(
                    model_config.vision_config.hidden_size,
                    1e-5,
                    weights.layers[i].attention_norm,
                ),
                mlp_norm=_rms_norm(
                    model_config.vision_config.hidden_size,
                    1e-5,
                    weights.layers[i].ffn_norm,
                ),
            )
            for i in range(model_config.vision_config.num_hidden_layers)
        ]

        return Transformer(
            n_heads=model_config.vision_config.num_attention_heads,
            layers=layers,
        )


def _vision_encoder(
    graph: Graph,
    pipeline_config: PipelineConfig,
    model_config: AutoConfig,
    weights: SafetensorWeights,
) -> VisionEncoder:
    patch_conv = _patch_conv2d(
        pipeline_config.dtype,
        model_config.vision_config.num_channels,
        model_config.vision_config.patch_size,
        model_config.vision_config.hidden_size,
        weights.vision_encoder.patch_conv,
    )
    ln_pre = _rms_norm(
        model_config.vision_config.hidden_size,
        1e-5,
        weights.vision_encoder.ln_pre,
    )
    patch_rope = RotaryEmbedding2D(
        dim=model_config.vision_config.hidden_size,
        n_heads=model_config.vision_config.num_attention_heads,
        theta=model_config.vision_config.rope_theta,
        max_patches_per_side=model_config.vision_config.image_size
        // model_config.vision_config.patch_size,
    )
    encoder_transformer = _transformer(
        graph, pipeline_config, model_config, weights.vision_encoder.transformer
    )

    return VisionEncoder(
        patch_conv=patch_conv,
        layer_norm=ln_pre,
        patch_positional_embedding=patch_rope,
        transformer=encoder_transformer,
        patch_size=model_config.vision_config.patch_size,
        max_image_size=model_config.vision_config.image_size,
    )


def _build_graph(
    pipeline_config: PipelineConfig,
    model_config: AutoConfig,
    weights: SafetensorWeights,
) -> Graph:
    # Graph input types.
    # TODO: What is the image type?
    image_type = TensorType(DType.bfloat16, shape=[300, 400, 3])

    # Initialize Graph.
    with Graph(
        "pixtral_vision_encoder",
        input_types=[
            image_type,
        ],
    ) as graph:
        model = _vision_encoder(graph, pipeline_config, model_config, weights)
        imgs = graph.inputs
        logits = model(imgs)  # type: ignore
        graph.output(logits)
        return graph
