import matplotlib.pyplot as plt

def save_plots(metrics_cb, outdir):
    """Generate loss & accuracy plots and save them into *outdir*.

    Returns the relative filenames of the created PNG files so they can be
    logged in metrics.
    """
    loss_path = outdir / "loss.png"
    acc_path  = outdir / "accuracy.png"

    # Loss per batch --------------------------------------------------------
    plt.figure()
    plt.plot(range(1, len(metrics_cb.batch_losses) + 1), metrics_cb.batch_losses)
    plt.title("Training Loss per Batch")
    plt.xlabel("Batch iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    # Accuracy --------------------------------------------------------------
    plt.figure()
    plt.plot(range(1, len(metrics_cb.batch_acc) + 1), metrics_cb.batch_acc, label="Train Acc")
    plt.plot(range(1, len(metrics_cb.epoch_val_acc) + 1), metrics_cb.epoch_val_acc,
             linestyle="--", marker="o", label="Val Acc")
    plt.title("Accuracy over Training")
    plt.xlabel("Epoch iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()

    return loss_path.name, acc_path.name