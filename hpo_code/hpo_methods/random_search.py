import random
import time
import pathlib
from ..train import train_model, evaluate_model
from ..utils import save_plots
import json
from types import SimpleNamespace

def run_random_search(base_args,
                      search_space: dict,
                      n_trials: int,
                      results_root: str):
    """
    Random search: sample hyperparameter configurations from the
    search space n_trials times.

    base_args     : SimpleNamespace with your default hyper-params
    search_space  : dict {param_name: [list_of_choices]}
    n_trials      : how many random configs to try
    results_root  : folder where all_results.txt, best_result.txt, and
                    the best-trial plots will be saved
    """
    root = pathlib.Path(results_root)
    root.mkdir(parents=True, exist_ok=True)

    trial_records = []
    metrics_cbs = []
    start_time = time.time()

    for i in range(n_trials):
        # sample configuration
        cfg = dict(vars(base_args))
        for k, choices in search_space.items():
            cfg[k] = random.choice(choices)
        args = SimpleNamespace(**cfg)

        # train & evaluate
        model, metrics_cb, val_ds = train_model(args)
        stats = evaluate_model(model, val_ds)

        # record results
        trial_records.append({
            "trial": i + 1,
            "args":   cfg,
            "stats":  stats
        })
        metrics_cbs.append(metrics_cb)

        print(f"trial {i+1}/{n_trials} finished")

    total_time = time.time() - start_time

    # save all_results.txt
    all_results = {
        "total_time": total_time,
        "results":    trial_records
    }
    (root / "all_results.txt").write_text(
        json.dumps(all_results, indent=2)
    )

    # find the best trial (by validation accuracy)
    best_idx = max(
        range(n_trials),
        key=lambda idx: trial_records[idx]["stats"]["accuracy"]
    )
    best_trial = trial_records[best_idx]
    best_metrics = metrics_cbs[best_idx]

    # save plots for the best trial
    loss_png, acc_png = save_plots(best_metrics, root)

    # save best_result.txt
    best_trial_with_plots = {
        **best_trial,
        "plots": {
            "loss":     loss_png,
            "accuracy": acc_png
        }
    }
    (root / "best_result.txt").write_text(
        json.dumps(best_trial_with_plots, indent=2)
    )

    print(f"Done! Ran {n_trials} trials in {total_time:.1f} sec. "
          f"Best trial: #{best_trial['trial']} with validation accuracy "
          f"{best_trial['stats']['accuracy']:.4f}")
