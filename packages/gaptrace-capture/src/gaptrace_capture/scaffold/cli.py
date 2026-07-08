import click

from gaptrace_capture.scaffold.template import generate_scaffold


@click.group()
def main():
    """gaptrace-capture -- pipeline instrumentation SDK for gaptrace observability system."""


@main.command()
def init():
    """Generate a starter gaptrace_pipeline.py in the current directory."""
    try:
        path = generate_scaffold()
    except FileExistsError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1)
    click.echo(f"Created {path.name} — fill in your pipeline stages.")
