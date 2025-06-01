import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, losses, metrics
from ..utils.callbacks import IterationMetrics
from ..utils.io import load_datasets
from ..models.alexnet import build_model
from sklearn.model_selection import KFold

def train_model(args):
    """
    1) Load train & val datasets from args.dir
    2) build_model(...)
    3) fit with IterationMetrics callback
    Returns:
      - model (trained)
      - metrics_cb (so you can inspect batch_losses, batch_acc, epoch_val_acc)
      - val_dataset (for evaluation later)
    """
    # load datasets
    train_ds, val_ds = load_datasets(args)

    # build model
    input_shape = (args.imageSize, args.imageSize, 1)
    n_categories = len(train_ds.class_names)
    model = build_model(args, input_shape, n_categories)
    print("model built successfully!")
    
    # train model
    metrics_cb = IterationMetrics()
    model.fit(
        train_ds,
        epochs=args.epochs1,
        validation_data=val_ds,
        callbacks=[metrics_cb]
    )
    return model, metrics_cb, val_ds

def evaluate_model(model, val_dataset):
    """
    Run model.predict on val_dataset, compute accuracy & log-likelihood.
    Returns a dict {'accuracy': float, 'loglik': float}.
    """
    phat = model.predict(val_dataset)

    # true labels as integer indices
    y_true = np.concatenate([y for x, y in val_dataset], axis=0).argmax(axis=1)
    y_pred = phat.argmax(axis=1)

    accuracy = np.mean(y_true == y_pred)
    true_probs = phat[np.arange(len(y_true)), y_true]
    loglik     = np.sum(np.log(true_probs))

    return {'accuracy': accuracy}

########################################
# for TPE automatic stopping
def dataset_to_numpy(dataset):
    """
    Convert a tf.data.Dataset of (x, y_onehot) batches into
    NumPy arrays X (images) and y (integer labels).
    """
    X_list, y_list = [], []
    for x_batch, y_batch in dataset.unbatch().as_numpy_iterator():
        X_list.append(x_batch)
        # y_batch is one-hot; convert to integer labels
        y_list.append(np.argmax(y_batch))
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    return X, y

def make_tf_dataset(X, y, batch_size, shuffle=False):
    """
    Build a tf.data.Dataset from NumPy arrays of images X and integer labels y.
    Converts labels back to one-hot for model.fit consistency.
    """
    n_classes = int(np.max(y) + 1)
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(y), seed=1)
    # One-hot encode labels
    ds = ds.map(lambda x, y: (x, tf.one_hot(y, depth=n_classes)), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def train_model_kfold(args, k_folds: int = 5):
    """
    Perform k-fold cross-validation for a single hyperparameter setting
    using the existing train_model (via build_model & load_datasets).

    Returns:
      - last_model: the model trained on the last fold
      - last_metrics: the IterationMetrics callback from the last fold
      - val_losses: list of final validation losses (one per fold)
      - val_accuracies: list of final validation accuracies (one per fold)
    """
    # load full training dataset and ignore its val split
    full_train_ds, _ = load_datasets(args)
    # convert to NumPy arrays
    X, y = dataset_to_numpy(full_train_ds)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=1)
    val_losses = []
    val_accuracies = [] 
    last_model = None
    last_metrics = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        # Slice NumPy arrays for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Build tf.data.Dataset objects
        train_ds = make_tf_dataset(X_train, y_train, batch_size=args.batch1, shuffle=True)
        val_ds   = make_tf_dataset(X_val, y_val, batch_size=args.batch1, shuffle=False)

        # Build a fresh model for this fold
        input_shape = (args.imageSize, args.imageSize, 1)
        n_categories = int(np.max(y) + 1)
        model = build_model(args, input_shape, n_categories)

        # Train with IterationMetrics callback
        metrics_cb = IterationMetrics()
        history = model.fit(
            train_ds,
            epochs=args.epochs1,
            validation_data=val_ds,
            callbacks=[metrics_cb],
            verbose=0
        )

        # Record final val_loss and val_accuracy for this fold
        val_loss = history.history["val_loss"][-1]
        val_losses.append(val_loss)

        val_accuracy = history.history["val_categorical_accuracy"][-1]
        val_accuracies.append(val_accuracy)

        # Save for return
        last_model = model
        last_metrics = metrics_cb

    return last_model, last_metrics, val_losses, val_accuracies