"""Post every example request to a running API and save each response.

Usage (from the repository root, with the server running):

    uv run python -m scripts.run_scenarios [--url http://127.0.0.1:8000]
        [--examples examples] [--out validation/responses]

The saved JSON files are the API responses submitted for the assignment scenarios.
"""

import argparse
import json
import sys
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--examples", default="examples", type=Path)
    parser.add_argument("--out", default="validation/responses", type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    failures = 0
    # The drawdown example takes about a second; allow generous time per request.
    with httpx.Client(base_url=args.url, timeout=120) as client:
        for request_path in sorted(args.examples.glob("*.json")):
            response = client.post(
                "/optimize",
                content=request_path.read_bytes(),
                headers={"Content-Type": "application/json"},
            )
            output = args.out / request_path.name
            output.write_text(json.dumps(response.json(), indent=2) + "\n")
            status = "ok" if response.status_code == 200 else "FAILED"
            failures += response.status_code != 200
            print(f"{status:6} HTTP {response.status_code} -> {output}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
