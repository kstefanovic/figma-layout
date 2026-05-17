"""Train the V2 child relative layout transformer."""

from __future__ import annotations

import argparse

from .models.child_model import ChildLayoutTransformer
from .training import add_train_args, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_train_args(parser)
    train_model(args=parser.parse_args(), model_class=ChildLayoutTransformer, dataset_type="child")


if __name__ == "__main__":
    main()

