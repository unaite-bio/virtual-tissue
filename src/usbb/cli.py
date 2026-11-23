"""Command line: `usbb run` (the method step) and `usbb validate` (a SpatialData store)."""

from __future__ import annotations

import argparse
import json

from usbb.run import run_method


def main(argv=None) -> None:
    """Run one subcommand; a failure raises, so the process exits with code 1."""
    parser = argparse.ArgumentParser(prog="usbb", description="Unaite Spatial Biology Benchmark")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run", help="run one method on a test bundle (Open Problems method component)"
    )
    run.add_argument("--method", required=True, help="method_id of a card in src/usbb/methods")
    run.add_argument("--input_train", help="train bundle folder, used by fitted methods only")
    run.add_argument("--input_test", required=True, help="test bundle folder")
    run.add_argument(
        "--output", required=True, help="run folder; the result is published in <output>/output/"
    )
    run.add_argument("--model", help="folder holding the method's pinned weights")
    run.add_argument(
        "--params", type=json.loads, help="JSON object of parameters over the card's defaults"
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--env_lock", help="lock file of the method's environment, hashed into the run key"
    )

    validate = commands.add_parser(
        "validate", help="check a SpatialData store against the usbb schema"
    )
    validate.add_argument("store", help="path to a <store_id>.zarr store")

    args = parser.parse_args(argv)
    if args.command == "run":
        output = run_method(
            args.method,
            args.input_train,
            args.input_test,
            args.output,
            args.params,
            args.seed,
            args.model,
            args.env_lock,
        )
        print(output)
    else:
        from usbb.data.schema import validate_store  # spatialdata exists in the core env only

        validate_store(args.store)
        print(f"{args.store} is a valid usbb store")


if __name__ == "__main__":
    main()
