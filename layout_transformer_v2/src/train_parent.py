"""Train the V2 parent structural layout transformer."""

from __future__ import annotations

import argparse

from .models.parent_model import ParentLayoutTransformer
from .training import add_train_args, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_train_args(parser)
    train_model(args=parser.parse_args(), model_class=ParentLayoutTransformer, dataset_type="parent")


if __name__ == "__main__":
    main()

