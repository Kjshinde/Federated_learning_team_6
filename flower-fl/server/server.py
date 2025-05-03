#!/usr/bin/env python3

# Standard library imports
import os, sys
# Add the project root directory to the Python path so that 'common/' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Third-party and Flower (FL) imports
import argparse
import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg, FedAdagrad, FedAdam, FedYogi

# Map of strategy name to corresponding Flower strategy class
STRATEGIES = {
    "FedAvg":     FedAvg,
    "FedAdagrad": FedAdagrad,
    "FedAdam":    FedAdam,
    "FedYogi":    FedYogi,
}

def aggregate_metrics(metrics_list):
    """Aggregate client evaluation metrics.

    - Average accuracy across clients.
    - Sum total misclassifications.
    """
    accuracies = [m.get("accuracy", 0.0) for _, m in metrics_list]
    miscls     = [m.get("misclassified", 0) for _, m in metrics_list]
    return {
        "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "misclassified": sum(miscls),
    }

def make_savebest_strategy(base_cls):
    """Create a subclass of the given strategy class that saves the best model by loss."""
    class SaveBest(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.best_loss = float("inf")  # Track lowest evaluation loss seen so far
            self._last_ndarrays = None     # Store latest parameters to save best model

        def aggregate_fit(self, rnd, results, failures):
            # Standard aggregation of model updates
            params, agg_metrics = super().aggregate_fit(rnd, results, failures)
            # Save parameters from this round
            self._last_ndarrays = parameters_to_ndarrays(params)
            return params, agg_metrics

        def aggregate_evaluate(self, rnd, results, failures):
            # Evaluate aggregated model and check if this is the best model so far
            loss, agg_metrics = super().aggregate_evaluate(rnd, results, failures)
            if loss is not None and loss < self.best_loss:
                self.best_loss = loss
                # Save the best model as a .npz file
                np.savez(
                    "best_model.npz",
                    *self._last_ndarrays,
                    metadata={
                        "round": rnd,
                        "loss": float(loss),
                        **(agg_metrics or {}),
                    },
                )
                print(f"→ Round {rnd}: new best loss={loss:.4f}; saved best_model.npz")
            return loss, agg_metrics

    return SaveBest

if __name__ == "__main__":
    # Parse CLI arguments for server configuration
    parser = argparse.ArgumentParser(
        description="Flower server w/ selectable strategy & best-model saving"
    )
    parser.add_argument("-p", "--port", type=int, default=8080)
    parser.add_argument("-r", "--num-rounds", type=int, default=10)
    parser.add_argument("-e", "--local-epochs", type=int, default=1)
    parser.add_argument("-l", "--learning-rate", type=float, default=0.01)
    parser.add_argument("-m", "--strategy", type=str, default="FedAvg", choices=list(STRATEGIES.keys()))
    parser.add_argument("-nc", "--num-classes", type=int, default=5, help="Model output classes (for dummy init)")
    args = parser.parse_args()

    # Import model building function and Torch
    from common.models.cnn import build_model
    import torch

    # Build a dummy model to initialize global parameters
    model = build_model(num_classes=args.num_classes)
    initial_ndarrays = [t.cpu().numpy() for t in model.state_dict().values()]
    initial_parameters = ndarrays_to_parameters(initial_ndarrays)

    # Configuration function to send learning config to clients each round
    def fit_config(rnd: int):
        return {"local_epochs": args.local_epochs, "lr": args.learning_rate}

    # Instantiate strategy with custom SaveBest wrapper
    BaseStrat = STRATEGIES[args.strategy]
    SaveBest  = make_savebest_strategy(BaseStrat)
    strategy  = SaveBest(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        on_fit_config_fn=fit_config,
    )

    # Log configuration summary
    print(
        f"→ Starting server on 0.0.0.0:{args.port} | rounds={args.num_rounds} | "
        f"strategy={args.strategy} | epochs/client={args.local_epochs} | lr={args.learning_rate}"
    )

    # Launch the Flower server
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
