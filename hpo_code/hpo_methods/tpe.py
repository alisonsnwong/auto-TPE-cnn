import time
import pathlib
import json
from types import SimpleNamespace
import numpy as np

from hyperopt import fmin, tpe, Trials, STATUS_OK, hp

from ..train import train_model, evaluate_model
from ..utils import save_plots

def run_tpe(base_args,
            search_space: dict,
            n_trials: int,
            results_root: str):
    """
    Tree‐structured Parzen Estimator (TPE) search over `search_space`
    for `n_trials` evaluations.

    base_args     : SimpleNamespace with your default hyper-params
    search_space  : dict {param_name: [list_of_choices]}
    n_trials      : number of TPE evaluations
    results_root  : folder where all_results.txt, best_result.txt, and
                    the best‐trial plots will be saved
    """
    root = pathlib.Path(results_root)
    root.mkdir(parents=True, exist_ok=True)

    # build Hyperopt space: hp.choice for each list of choices
    space = {k: hp.choice(k, choices) for k, choices in search_space.items()}

    trials      = Trials()
    metrics_cbs = []
    results     = []

    start_time = time.time()

    def objective(cfg):
        # merge base_args with this trial's cfg
        params = dict(vars(base_args))
        params.update(cfg)
        args = SimpleNamespace(**params)

        # train & evaluate
        model, metrics_cb, val_ds = train_model(args)
        stats = evaluate_model(model, val_ds)

        # record for later
        metrics_cbs.append(metrics_cb)
        results.append({
            "args":   params,
            "stats":  stats
        })

        # Hyperopt minimizes loss, so negate accuracy
        return {"loss": -stats.get("accuracy", 0.0),
                "status": STATUS_OK}

    # run TPE
    fmin(fn=objective,
         space=space,
         algo=tpe.suggest,
         max_evals=n_trials,
         trials=trials)

    total_time = time.time() - start_time

    # write all results
    all_out = {
        "total_time": total_time,
        "results":    results
    }
    (root / "all_results.txt").write_text(
        json.dumps(all_out, indent=2)
    )

    # pick best trial by validation accuracy
    best_idx  = max(range(len(results)),
                    key=lambda i: results[i]["stats"]["accuracy"])
    best_trial   = results[best_idx]
    best_metrics = metrics_cbs[best_idx]

    # save only the best's plots
    loss_png, acc_png = save_plots(best_metrics, root)

    best_out = {
        **best_trial,
        "plots": {"loss": loss_png, "accuracy": acc_png}
    }
    (root / "best_result.txt").write_text(
        json.dumps(best_out, indent=2)
    )

    print(f"Done! TPE completed {len(results)} trials in "
          f"{total_time:.1f}s. Best accuracy={best_trial['stats']['accuracy']:.4f}")