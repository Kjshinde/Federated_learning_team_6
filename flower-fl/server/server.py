#!/usr/bin/env python3
import argparse
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedAvg, FedAdagrad, FedAdam, FedYogi

def aggregate_metrics(metrics_list):
    """Average accuracy; sum misclassifications."""
    accuracies = [m.get("accuracy", 0.0) for _, m in metrics_list]
    miscls     = [m.get("misclassified", 0)   for _, m in metrics_list]
    return {
        "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "misclassified": sum(miscls),
    }

def make_savebest_strategy(base_cls):
    class SaveBest(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.best_loss = float("inf")
            self._last_ndarrays = None

        def aggregate_fit(self, rnd, results, failures):
            # Get the new global parameters from super
            parameters, agg_metrics = super().aggregate_fit(rnd, results, failures)
            # Convert Flower Parameters → list of numpy ndarrays
            self._last_ndarrays = parameters_to_ndarrays(parameters)
            return parameters, agg_metrics

        def aggregate_evaluate(self, rnd, results, failures):
            # Run the normal evaluate aggregation (loss, metrics)
            loss, agg_metrics = super().aggregate_evaluate(rnd, results, failures)

            # If improved, save out the last parameters as a .npz
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

# Strategy map
STRATEGIES = {
    "FedAvg":    FedAvg,
    "FedAdagrad":FedAdagrad,
    "FedAdam":   FedAdam,
    "FedYogi":   FedYogi,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flower FL server with strategy selection & best-model saving"
    )
    parser.add_argument("-p", "--port",       type=int, default=8080, help="Port to listen on")
    parser.add_argument("-r", "--num-rounds", type=int, default=10,  help="No. of federated rounds")
    parser.add_argument(
        "-s", "--strategy", type=str, default="FedAvg",
        choices=list(STRATEGIES),
        help="Aggregation strategy to use"
    )
    args = parser.parse_args()

    # Build and wrap the chosen strategy
    Base = STRATEGIES[args.strategy]
    SaveBestStrategy = make_savebest_strategy(Base)
    strategy = SaveBestStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
    )

    print(f"→ Starting server on 0.0.0.0:{args.port} "
          f"for {args.num_rounds} rounds using {args.strategy}")
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
