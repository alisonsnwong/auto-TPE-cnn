import os, tensorflow as tf
from typing import Tuple

def load_datasets(args) -> Tuple[tf.data.Dataset,
                                 tf.data.Dataset]:
    """
    Returns
    --------
    train_ds   : first training dataset  (batch = args.batch1)
    val_ds     : validation dataset      (batch = 64, no shuffle)

    All datasets are grayscale and label‑mode 'categorical'.
    """

    img_size = args.imageSize
    data_dir = args.dir

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=args.batch1,
        image_size=(img_size, img_size),
        shuffle=True, # so it doesn't reshuffle every epoch 
        seed=1 # same data so uniform across HPO methods
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "validation"),
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=64,
        image_size=(img_size, img_size),
        shuffle=False 
    )

    return train_ds, val_ds
