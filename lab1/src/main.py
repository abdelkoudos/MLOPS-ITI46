import argparse

from eval import evaluate_model, save_metrics
from pred import load_test_data, predict, save_predictions
from train import AVAILABLE_MODELS, DEFAULT_MODEL, save_model, train_model

from data import load_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Titanic survival models."
    )
    parser.add_argument(
        "--model",
        choices=AVAILABLE_MODELS,
        default=DEFAULT_MODEL,
        help="Model to train.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all available model options.",
    )
    return parser.parse_args()


def run_pipeline(df, test_df, model_name):
    print(f"Training {model_name}...")
    model, X_val, y_val = train_model(df, model_name=model_name)
    save_model(model, model_name=model_name)

    print(f"Evaluating {model_name}...")
    metrics = evaluate_model(model, X_val, y_val)
    save_metrics(metrics, model_name=model_name)

    print(f"Generating {model_name} predictions...")
    predictions = predict(model, test_df)
    save_predictions(test_df, predictions, model_name=model_name)

    print(f"{model_name} complete.")
    if "accuracy" in metrics:
        print(f"Accuracy: {metrics['accuracy']:.4f}")
    if "auc" in metrics:
        print(f"AUC: {metrics['auc']:.4f}")
    return metrics


def main():
    args = parse_args()
    model_names = AVAILABLE_MODELS if args.all else (args.model,)

    print("Loading data...")
    df = load_data()
    test_df = load_test_data()

    results = {}
    for model_name in model_names:
        results[model_name] = run_pipeline(df, test_df, model_name)

    print("Pipeline complete.")
    if len(results) > 1:
        print("Model comparison:")
        for model_name, metrics in results.items():
            metrics_text = ", ".join(
                f"{name}={value:.4f}" for name, value in metrics.items()
            )
            print(f"{model_name}: {metrics_text}")


if __name__ == "__main__":
    main()
