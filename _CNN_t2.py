# Model / data parameters
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.callbacks import TerminateOnNaN, EarlyStopping
import tensorflow_model_optimization as tfmot
import numpy as np

MODEL_SIZE = {}
ACCURACY = {}
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
    model = Sequential()
    model.add(Conv2D(32, input_shape=input_shape, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
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

converter = tf.lite.TFLiteConverter.from_keras_model(baseline_model)
tflite_model = converter.convert()
with open("baseline_model.tflite", "wb") as f:
    f.write(tflite_model)
# KD begin

# KD end
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

# Remove pruning wrappers and save
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_export.summary()

# Convert pruned h5 model to tflite directly w/o quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
tflite_model = converter.convert()
with open("pruned_not_quantized.tflite", "wb") as f:
    f.write(tflite_model)

# After pruning, use "dynamic range quantization" to quantize the pruned model (Post-Traning Quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("pruned_quantized.tflite", "wb") as f:
    f.write(tflite_model)

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model_for_export)

# `quantize_model` requires a recompile.
q_aware_model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

q_aware_model.summary()

q_aware_model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=2,
    validation_split=validation_split,
    callbacks=[ton, esl, esa],
)

# After this, you have an actually quantized model with int8 weights and uint8 activations
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()
with open("pruned_QATed.tflite", "wb") as f:
    f.write(tflite_model)
