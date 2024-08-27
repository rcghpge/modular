# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from llama3.llama3 import Llama3Context
from max.serve.api_server import fastapi_app, fastapi_config
from max.serve.config import APIType, Settings
from max.serve.pipelines.deps import token_pipeline
from max.serve.pipelines.llm import TokenGenerator, TokenGeneratorPipeline
from uvicorn import Server


async def serve_token_generator(model: TokenGenerator):
    pipeline = TokenGeneratorPipeline[Llama3Context](model)
    pipelines = [pipeline]

    settings = Settings(api_types=[APIType.OPENAI])
    app = fastapi_app(settings, pipelines)
    app.dependency_overrides[token_pipeline] = lambda: pipeline

    config = fastapi_config(app=app)
    server = Server(config)
    await server.serve()
