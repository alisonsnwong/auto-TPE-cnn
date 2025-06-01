import time
import pathlib
import json
import numpy as np
from types import SimpleNamespace

from hyperopt import fmin, tpe, Trials, STATUS_OK, hp
from ..train import train_model, evaluate_model, train_model_kfold
from ..utils import save_plots

def run_tpe_auto(base_args,
                 search_space: dict,
                 max_trials: int,
                 results_root: str,
                 k_folds: int = 10,
                 min_trials: int = 5):
    """
    TPE optimization with automatic early stopping based on:
        (second_best_mean_loss - best_mean_loss) < sqrt(0.21 * var(best_val_losses))
    using k-fold cross-validation.
    """
    root = pathlib.Path(results_root)
    root.mkdir(parents=True, exist_ok=True)

    # Build Hyperopt search space (hp.choice for each categorical/discrete list)
    space = {k: hp.choice(k, v) for k, v in search_space.items()}

    trials = Trials()
    metrics_cbs = []
    results = []

    start_time = time.time()

    def objective(cfg):
        # Merge base_args and this trial's hyperparameters
        params = dict(vars(base_args))
        params.update(cfg)
        args = SimpleNamespace(**params)

        # Run k-fold CV (train_model_kfold should return: model, metrics_cb, val_losses, val_accuracies)
        model, metrics_cb, val_losses, val_accuracies = train_model_kfold(args, k_folds=k_folds)

        # Compute mean and variance of the k-fold validation losses
        mean_loss = float(np.mean(val_losses))
        var_loss = float(np.var(val_losses))
        mean_accuracy = float(np.mean(val_accuracies))

        # Record metrics for later
        metrics_cbs.append(metrics_cb)
        results.append({
            "args": params,
            "mean_loss": mean_loss,
            "var_loss": var_loss,
            "mean_accuracy": mean_accuracy,
        })

        return {"loss": mean_loss, "status": STATUS_OK}

    # Incremental fmin loop
    for i in range(max_trials):
        # Let hyperopt run exactly (i+1) evaluations total
        fmin(fn=objective,
             space=space,
             algo=tpe.suggest,
             max_evals=i+1,
             trials=trials,
             show_progressbar=False)

        # Early stopping once we have at least `min_trials` results
        if len(results) >= min_trials:
            # Sort all completed trials by ascending mean_loss
            sorted_results = sorted(results, key=lambda x: x["mean_loss"])
            best = sorted_results[0]
            second_best = sorted_results[1]

            # Regret = gap between best and second-best (mean validation loss)
            regret = second_best["mean_loss"] - best["mean_loss"]
            # Statistical noise = sqrt(0.21 * variance_of_best_loss)
            stat_error = np.sqrt(0.21 * best["var_loss"])

            if regret < stat_error:
              print(f"Early stopping at trial {i+1}: "
                    f"regret={regret:.5f} < noise={stat_error:.5f}")
              break
            else:
              print(f"Trial {i+1}: "
                    f"regret={regret:.5f} > noise={stat_error:.5f}")

    total_time = time.time() - start_time

    # Save results to JSON
    all_out = {
        "total_time": total_time,
        "results": results
    }
    (root / "all_results.json").write_text(json.dumps(all_out, indent=2))

    # Identify the best trial by mean_loss
    best_idx = int(np.argmin([r["mean_loss"] for r in results]))
    best = results[best_idx]
    best_metrics = metrics_cbs[best_idx]

    # Save plots from the best trial's metrics callback
    loss_png, acc_png = save_plots(best_metrics, root)
    best_out = {
        **best,
        "plots": {"loss": loss_png, "accuracy": acc_png}
    }
    (root / "best_result.json").write_text(json.dumps(best_out, indent=2))

    print(f"Done! Completed {len(results)} trials in {total_time:.1f}s. "
          f"Best mean loss = {best['mean_loss']:.4f}, "
          f"Best mean accuracy = {best['mean_accuracy']:.4f}")

    return best, best_metrics
