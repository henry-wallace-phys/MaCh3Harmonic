from typing import Literal
import logging
from rich.logging import RichHandler

import click
from pathlib import Path
from .single_flow_comp import single_flow_comp_cmd

import yaml

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}


logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[click])]
)

@click.group(context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.option("--config", "-c", type=click.File("r"), help="YAML Config file", required=True)
@click.option(
    "--log-level", "-l",
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Log level",
)
@click.pass_context
def cli(ctx, config, log_level):
    logging.getLogger().setLevel(log_level.upper())
    ctx.ensure_object(dict)
    ctx.obj = yaml.safe_load(config) if config else {}

@cli.command(
    "single_flow_comp"
)
@click.pass_obj
def single_flow_comp(config):
    single_flow_comp_cmd(config)