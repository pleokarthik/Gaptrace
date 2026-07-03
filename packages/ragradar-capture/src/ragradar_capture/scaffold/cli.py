import sys

from ragradar_capture.scaffold.template import generate_scaffold


def main():
    try:
        path = generate_scaffold()
        print(f"Created {path.name} — fill in your pipeline stages.")
    except FileExistsError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
