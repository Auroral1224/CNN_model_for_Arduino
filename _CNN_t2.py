# Model / data parameters
import os
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, EarlyStopping
from keras.models import load_model
from keras import layers
from tensorflow import keras
import zipfile
import tempfile
import tensorflow_model_optimization as tfmot
import tensorflow as tf
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


# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Build the unpruned model

def get_model():
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


baseline_model = get_model()
baseline_model.summary()

mcp = ModelCheckpoint(filepath='non_pruned.h5', monitor='val_loss', verbose=0,
                      save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
ton = TerminateOnNaN()
esl = EarlyStopping(monitor='val_loss', patience=4,
                    mode='auto', restore_best_weights=True)
esa = EarlyStopping(monitor='val_accuracy', patience=4,
                    mode='auto', restore_best_weights=True)

# Train the unpruned model

batch_size = 32
epochs = 30
validation_split = 0.1  # 10% of training set will be used for validation set.


baseline_model.compile(loss="categorical_crossentropy",
                       optimizer="adam", metrics=["accuracy"])

baseline_model.fit(x_train, y_train, batch_size=batch_size,
                   epochs=epochs, validation_split=0.1, callbacks=[mcp, ton, esl, esa])

baseline_model = load_model('non_pruned.h5')

# Evaluate the accuracy of the unpruned model
_, ACCURACY['baseline Keras model'] = baseline_model.evaluate(
    x_test, y_test, verbose=0)

# Save the size of unpruned model
MODEL_SIZE['baseline h5'] = os.path.getsize('non_pruned.h5')

# Pruning process

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after n epochs
num_train = x_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_train / batch_size).astype(np.int32) * epochs

# Define model for pruning
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.5,
                                                             final_sparsity=0.80,
                                                             begin_step=0,
                                                             end_step=end_step)
}

model_for_pruning = prune_low_magnitude(get_model(), **pruning_params)

# `prune_low_magnitude` requires a recompile
model_for_pruning.compile(optimizer='adam',
                          loss="categorical_crossentropy",
                          metrics=['accuracy'])

model_for_pruning.summary()

# retrain the model, and apply a new callback
ups = tfmot.sparsity.keras.UpdatePruningStep()
model_for_pruning.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                      validation_split=validation_split, callbacks=[ton, esl, esa, ups])

# Remove pruning wrappers and save
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_export.summary()
model_for_export.save('pruned_model.h5', include_optimizer=False)


_, ACCURACY['pruned Keras model'] = model_for_pruning.evaluate(
    x_test, y_test, verbose=0)
MODEL_SIZE['pruned h5'] = os.path.getsize('pruned_model.h5')

model_for_pruning = get_model('pruned_model.h5')
# Zip the .h5 model file
_, zip3 = tempfile.mkstemp(".zip")
with zipfile.ZipFile(zip3, "w", compression=zipfile.ZIP_DEFLATED) as f:
    f.write(model_for_pruning)
# # 剪枝壓縮後再量化模型
# converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# tflite_model = converter.convert()

# with open('pruned_quantized.tflite', 'wb') as f:
#     f.write(tflite_model)


# # A helper function to evaluate the TF Lite model using "test" dataset.
# def evaluate_model(interpreter):
#     input_index = interpreter.get_input_details()[0]["index"]
#     output_index = interpreter.get_output_details()[0]["index"]

#     # Run predictions on every image in the "test" dataset.
#     prediction_digits = []
#     for test_image in x_test:
#         # Pre-processing: add batch dimension and convert to float32 to match with
#         # the model's input data format.
#         test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
#         interpreter.set_tensor(input_index, test_image)

#         # Run inference.
#         interpreter.invoke()

#         # Post-processing: remove batch dimension and find the digit with highest
#         # probability.
#         output = interpreter.tensor(output_index)
#         digit = np.argmax(output()[0])
#         prediction_digits.append(digit)

#     # Compare prediction results with ground truth labels to calculate accuracy.
#     accurate_count = 0
#     for index in range(len(prediction_digits)):
#         if prediction_digits[index] == y_test[index]:
#             accurate_count += 1
#     accuracy = accurate_count * 1.0 / len(prediction_digits)

#     return accuracy


# ACCURACY['pruned and quantized tflite'] = evaluate_model(
#     'pruned_quantized.tflite')
print(ACCURACY)
print(MODEL_SIZE)
