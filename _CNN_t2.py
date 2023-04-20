# Model / data parameters
from keras.callbacks import TerminateOnNaN, EarlyStopping
from keras import layers
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import numpy as np

num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)


# Convert class vectors to binary class matrices (loss='categorical_crossentropy')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Build the unpruned model
def get_model():
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


baseline_model = get_model()
baseline_model.summary()

ton = TerminateOnNaN()
esl = EarlyStopping(
    monitor="val_loss", patience=4, mode="auto", restore_best_weights=True
)
esa = EarlyStopping(
    monitor="val_accuracy", patience=4, mode="auto", restore_best_weights=True
)


def tflite_converter_for_int_quant(model):
    def representative_data_gen():
        for input_value in (
            tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100)
        ):
            yield [input_value]

    model_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    model_converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    model_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    model_converter.inference_input_type = tf.uint8
    model_converter.inference_output_type = tf.uint8
    return model_converter


def save_tflite_model(model, model_converter, model_name):
    if model_converter == None:
        model_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model = model_converter.convert()
    with open(f"{model_name}.tflite", "wb") as f:
        f.write(model)


# Train the unpruned model

batch_size = 32
epochs = 30
validation_split = 0.1  # 10% of training set will be used for validation set.

baseline_model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

baseline_model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    callbacks=[ton, esl, esa],
)

# Save baseline h5 model
baseline_model.save("baseline.h5")

# Save baseline tflite model
save_tflite_model(baseline_model, None, "baseline")

# Pruning process
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after n epochs
num_train = x_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_train / batch_size).astype(np.int32) * epochs

# Define model for pruning
pruning_params = {
    "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.5, final_sparsity=0.8, begin_step=0, end_step=end_step
    )
}

model_for_pruning = prune_low_magnitude(get_model(), **pruning_params)

# `prune_low_magnitude` requires a recompile
model_for_pruning.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model_for_pruning.summary()

# retrain the model, and apply a new callback, epoch = 2 to prevent over-fitting
ups = tfmot.sparsity.keras.UpdatePruningStep()
model_for_pruning.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=2,
    validation_split=validation_split,
    callbacks=[ton, esl, esa, ups],
)

# Remove pruning wrappers
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_export.summary()

# Save pruned h5 model
model_for_export.save("pruned.h5")

# Save pruned tflite model
save_tflite_model(model_for_export, None, "pruned")

# Save pruned and integer-quantized model
save_tflite_model(
    model_for_export,
    tflite_converter_for_int_quant(model_for_export),
    "pruned_quantized",
)

# quantize_model = tfmot.quantization.keras.quantize_model

# # q_aware stands for for quantization aware.
# q_aware_model = quantize_model(model_for_export)

# # `quantize_model` requires a recompile.
# q_aware_model.compile(
#     loss="categorical_crossentropy",
#     optimizer="adam",
#     metrics=["accuracy"],
# )

# q_aware_model.summary()

# q_aware_model.fit(
#     x_train,
#     y_train,
#     batch_size=batch_size,
#     epochs=2,
#     validation_split=validation_split,
#     callbacks=[ton, esl, esa],
# )
