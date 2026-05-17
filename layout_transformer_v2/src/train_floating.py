"""Train the V2 floating canvas-relative layout transformer."""

from __future__ import annotations

import argparse

from .models.floating_model import FloatingLayoutTransformer
from .training import add_train_args, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_train_args(parser)
    train_model(args=parser.parse_args(), model_class=FloatingLayoutTransformer, dataset_type="floating")


if __name__ == "__main__":
    main()

