"""1D CNN for the preprocessed AI4I feature vector (binary: 1 = machine failure)."""
import keras
from keras import layers


def build_cnn(n_features, conv_filters=(32, 64), kernel_size=3, dense_units=64, dropout=0.3,
              learning_rate=1e-3):
    """[Conv1D(ReLU) -> MaxPool1D] per entry in conv_filters, then Flatten -> Dense -> Dropout -> sigmoid.

    padding="same" on conv and pool keeps the sequence length >= 1 for any n_features (8 -> 4 -> 2 by default).
    """
    model = keras.Sequential([keras.Input(shape=(n_features, 1))], name="ai4i_cnn")
    for filters in conv_filters:
        model.add(layers.Conv1D(filters, kernel_size, padding="same", activation="relu"))
        model.add(layers.MaxPooling1D(2, padding="same"))
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="sigmoid"))  # P(machine failure), usable for ROC directly
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="binary_crossentropy",
                  metrics=["accuracy",
                           keras.metrics.AUC(curve="PR", name="pr_auc"),
                           keras.metrics.AUC(curve="ROC", name="roc_auc"),
                           keras.metrics.Precision(name="precision"),
                           keras.metrics.Recall(name="recall")])
    return model
