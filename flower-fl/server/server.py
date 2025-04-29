#!/usr/bin/env python3
import argparse
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg, FedAdagrad, FedAdam, FedYogi

def aggregate_metrics(metrics_list):
    """Average accuracy, sum misclassifications across clients."""
    accuracies = [m.get("accuracy", 0.0) for _, m in metrics_list]
    miscls     = [m.get("misclassified", 0)   for _, m in metrics_list]
    return {
        "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "misclassified": sum(miscls),
    }

def make_savebest_strategy(base_cls):
    """
    Dynamically subclass base_cls to save best‐ever parameters
    whenever eval loss improves.
    """
    class SaveBest(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.best_loss = float("inf")
            self._last_parameters = None

        def aggregate_fit(self, rnd, results, failures):
            # Collect new global weights
            params, num_examples = super().aggregate_fit(rnd, results, failures)
            self._last_parameters = params
            return params, num_examples

        def aggregate_evaluate(self, rnd, results, failures):
            loss, metrics = super().aggregate_evaluate(rnd, results, failures)
            # If you got a valid loss and it's better than before, save
            if loss is not None and loss < self.best_loss:
                self.best_loss = loss
                # Dump as NumPy archive
                np.savez(
                    "best_model.npz",
                    *self._last_parameters,
                    metadata={
                        "round": rnd,
                        "loss": float(loss),
                        **metrics,
                    },
                )
                print(f"→ Round {rnd}: new best loss={loss:.4f}, weights saved to best_model.npz")
            return loss, metrics

    return SaveBest

# Map flag to Flower strategy class
STRATEGIES = {
    "FedAvg":    FedAvg,
    "FedAdagrad":FedAdagrad,
    "FedAdam":   FedAdam,
    "FedYogi":   FedYogi,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower FL server with strategy selection & best-model saving")
    parser.add_argument(
        "-p", "--port", type=int, default=8080,
        help="Port to listen on"
    )
    parser.add_argument(
        "-r", "--num-rounds", type=int, default=10,
        help="Number of federated rounds"
    )
    parser.add_argument(
        "-s", "--strategy", type=str, default="FedAvg",
        choices=list(STRATEGIES),
        help="Which aggregation strategy to use"
    )
    args = parser.parse_args()

    # Instantiate & wrap chosen strategy
    Base = STRATEGIES[args.strategy]
    Strategy = make_savebest_strategy(Base)
    strategy = Strategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
    )

    print(f"→ Starting Flower server on 0.0.0.0:{args.port}, "
          f"rounds={args.num_rounds}, strategy={args.strategy}")

    # Start server and block until done
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
