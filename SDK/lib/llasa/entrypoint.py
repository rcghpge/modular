# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import click

from llasa.config import LlasaConfig


@click.command()
@click.help_option("--help", "-h")
def main():
    """Llasa TTS - Text-to-Speech using Llama + XCodec2"""
    config = LlasaConfig()
    click.echo("Llasa TTS configured successfully!")
    click.echo(f"Llama model: {config.llama_model_path}")
    click.echo(f"XCodec2 model: {config.xcodec2_model_path}")


if __name__ == "__main__":
    main()
