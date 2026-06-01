#!/usr/bin/env python3
"""Run only explainability / XAI step (requires prior training or retrains quickly)."""

from src.config import RANDOM_STATE
from src.explainability import run_full_explainability
from src.train import (
    prepare_splits,
    refit_on_train_val,
    select_best_model,
    train_all_models,
)


def main() -> None:
    print("Loading splits and training models...")
    splits = prepare_splits()
    trained = train_all_models(splits)
    best_name = select_best_model(trained)
    final_model = refit_on_train_val(trained, splits, best_name)
    print(f"Best model: {best_name} | Running XAI pipeline...")
    run_full_explainability(trained, splits, best_name, final_model)
    print("Done. See outputs/figures/ and outputs/results/")


if __name__ == "__main__":
    main()
