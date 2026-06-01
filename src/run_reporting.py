#!/usr/bin/env python3
"""Regenerate reporting artifacts only (requires re-training models)."""

from src.reporting import run_full_reporting
from src.train import (
    prepare_splits,
    refit_on_train_val,
    select_best_model,
    train_all_models,
)
from src.evaluate import evaluate_all_on_test, evaluate_final_model


def main() -> None:
    splits = prepare_splits()
    trained = train_all_models(splits)
    best_name = select_best_model(trained)
    test_df = evaluate_all_on_test(trained, splits)
    final_model = refit_on_train_val(trained, splits, best_name)
    final_eval = evaluate_final_model(final_model, splits, model_name=best_name)
    run_full_reporting(trained, splits, best_name, final_model, final_eval, test_df)
    print("Reporting complete.")


if __name__ == "__main__":
    main()
