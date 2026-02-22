import click


@click.group()
def cli():
    """Ravenna — tiled bioacoustic spectrogram pipeline."""


@cli.command()
@click.option("--source", required=True, help="Audio source URI (local path, s3://, or http://)")
@click.option("--output", required=True, help="Output archive path (.pmtiles or .mbtiles)")
@click.option("--stages", default=None, help="Comma-separated subset of stages to run")
@click.option("--config", default=None, help="Path to JSON config file")
def run(source, output, stages, config):
    """Run the spectrogram pipeline."""
    raise NotImplementedError("Pipeline not yet implemented")


@cli.command()
def status():
    """Show pipeline progress for the current working directory."""
    raise NotImplementedError("Status not yet implemented")
