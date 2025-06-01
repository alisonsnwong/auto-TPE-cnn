import random
import time
import math
import pathlib
import json
from types import SimpleNamespace
from ..train import train_model, evaluate_model
from ..utils import save_plots

def run_hyperband(base_args,
                  search_space: dict,
                  max_iter: int,
                  eta: int,
                  results_root: str):
    """
    Hyperband over `search_space`, using resource up to `max_iter` and reduction factor `eta`.
    
    base_args     : SimpleNamespace with default hyper-params
    search_space  : dict {param_name: [list_of_choices]}
    max_iter      : maximum budget (e.g. epochs)
    eta           : down-sampling rate (e.g. 3)
    results_root  : folder where all_results.txt, best_result.txt, and best plots are saved
    """
    root = pathlib.Path(results_root)
    root.mkdir(parents=True, exist_ok=True)

    all_results = []
    best_trial = None
    best_metrics = None
    best_acc = 0
    start_time = time.time()

    # Hyperband constants
    s_max = int(math.floor(math.log(max_iter, eta)))
    B = (s_max + 1) * max_iter

    # Loop over each “bracket” s
    for s in reversed(range(s_max + 1)):
        n = int(math.ceil(B / max_iter / (s + 1) * (eta ** s)))
        r = max_iter * (eta ** (-s))

        # initial n random configs for this bracket
        configs = []
        for _ in range(n):
            cfg = dict(vars(base_args))
            for k, choices in search_space.items():
                cfg[k] = random.choice(choices)
            configs.append(cfg)

        # run successive halving for this bracket
        for i in range(s + 1):
            n_i = n * (eta ** (-i))
            r_i = r * (eta ** i)

            # evaluate each of the n_i configs with budget r_i
            results = []
            for cfg in configs:
                # inject budget into the args (assumes train_model reads `args.epochs`)
                trial_args = SimpleNamespace(**cfg)
                setattr(trial_args, 'epochs', int(r_i))

                model, metrics_cb, val_ds = train_model(trial_args)
                stats = evaluate_model(model, val_ds)

                # record
                record = {
                    "args":  cfg,
                    "stats": {**stats, "resource": int(r_i)}
                }
                all_results.append(record)
                results.append((record, metrics_cb))

                # update global best
                acc = stats.get('accuracy', -float('inf'))
                if acc > best_acc:
                    best_acc = acc
                    best_trial = record
                    best_metrics = metrics_cb

            # select top configs to continue
            results.sort(key=lambda x: x[0]['stats']['accuracy'], reverse=True)
            k = int(n_i / eta)
            configs = [r[0]['args'] for r in results[:k]]

    total_time = time.time() - start_time

    # write all_results.txt
    with open(root / "all_results.txt", "w") as f:
        json.dump({
            "total_time": total_time,
            "results":    all_results
        }, f, indent=2)

    # save plots for the single best trial
    loss_png, acc_png = save_plots(best_metrics, root)

    # write best_result.txt
    best_result = {
        **best_trial,
        "plots": {"loss": loss_png, "accuracy": acc_png}
    }
    with open(root / "best_result.txt", "w") as f:
        json.dump(best_result, f, indent=2)

    print(f"Done! Hyperband tested {len(all_results)} runs in "
          f"{total_time:.1f}s. Best accuracy={best_acc:.4f}.")