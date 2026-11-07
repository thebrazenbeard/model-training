from __future__ import annotations

import argparse
import json

from latent_proxy import build_default_cases, write_proxy_package


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the deterministic Vera latent proxy exercise package."
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="training/LATENT_PROXY_EXERCISES_V1.json",
    )
    args = parser.parse_args()
    payload = write_proxy_package(args.output, build_default_cases())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "case_count": payload["case_count"],
                "package_digest": payload["package_digest"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
