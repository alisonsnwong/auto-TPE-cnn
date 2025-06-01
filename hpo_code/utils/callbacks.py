from tensorflow.keras.callbacks import Callback

class IterationMetrics(Callback):
    def on_train_begin(self, logs=None):
        self.batch_losses = []
        self.batch_acc = []
        self.epoch_val_acc = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('categorical_accuracy')
        if acc is not None:
            self.batch_acc.append(acc)
        val_acc = logs.get('val_categorical_accuracy')
        if val_acc is not None:
            self.epoch_val_acc.append(val_acc)