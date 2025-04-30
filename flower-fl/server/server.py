#!/usr/bin/env python3
import argparse
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedAvg

def aggregate_metrics(metrics_list):
    """Average accuracy; sum misclassifications."""
    accuracies = [m.get("accuracy", 0.0) for _, m in metrics_list]
    miscls     = [m.get("misclassified",    0) for _, m in metrics_list]
    return {
        "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "misclassified": sum(miscls),
    }

def make_savebest_strategy(base_cls):
    """Wrap a strategy to save best‐ever weights whenever eval loss improves."""
    class SaveBest(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.best_loss = float("inf")
            self._last_ndarrays = None

        def aggregate_fit(self, rnd, results, failures):
            parameters, agg_metrics = super().aggregate_fit(rnd, results, failures)
            # Convert to numpy arrays for saving later
            self._last_ndarrays = parameters_to_ndarrays(parameters)
            return parameters, agg_metrics

        def aggregate_evaluate(self, rnd, results, failures):
            loss, agg_metrics = super().aggregate_evaluate(rnd, results, failures)
            if loss is not None and loss < self.best_loss:
                self.best_loss = loss
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
    parser = argparse.ArgumentParser(
        description="Flower server with on_fit_config and best‐model saving"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=8080,
        help="Port to listen on"
    )
    parser.add_argument(
        "-r", "--num-rounds", type=int, default=10,
        help="Number of federated learning rounds"
    )
    parser.add_argument(
        "-e", "--local-epochs", type=int, default=1,
        help="Local epochs per client"
    )
    parser.add_argument(
        "-l", "--learning-rate", type=float, default=0.01,
        help="Learning rate for client optimizers"
    )
    args = parser.parse_args()

    # Fit‐config: sent to each client every round
    def fit_config(rnd: int):
        return {"local_epochs": args.local_epochs, "lr": args.learning_rate}

    # Wrap FedAvg to save best eval loss
    SaveBestFedAvg = make_savebest_strategy(FedAvg)
    strategy = SaveBestFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        on_fit_config_fn=fit_config,
    )

    print(
        f"→ Starting Flower server on 0.0.0.0:{args.port} "
        f"for {args.num_rounds} rounds, "
        f"{args.local_epochs} epochs/client, lr={args.learning_rate}"
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
