import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt  # pip install matplotlib
import timeit

class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
model_name = "pruned_quantized.tflite"
image_label = 4
image_name = f"ImgData\\{image_label}.jpg"

image = Image.open(image_name).convert("L")
image = image.resize((28, 28), Image.Resampling.LANCZOS)
img = np.array(image).astype("uint8")
img = 255 - img
im = np.expand_dims(img, 0)
im = np.expand_dims(im, -1)
print(im.shape)
print(im)


def plot_image(predictions_array, true_label, imgarg):
    plt.title(f"Model:{model_name}")
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(imgarg, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    plt.xlabel(
        "{} {:2.0f}% (label:{})".format(
            class_names[predicted_label],
            100 * (np.max(predictions_array)) / 256,
            class_names[true_label],
        ),
        color=color,
    )


def plot_value_array(predictions_array, true_label, latency):
    plt.title(f"Latency:{latency}s")
    plt.grid(False)
    plt.xticks(range(10), class_names, rotation=45)
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array / 256, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")


# Load TFLite model and allocate tensors.
t1 = timeit.default_timer()
interpreter = tf.lite.Interpreter(model_path=model_name)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]["index"], im)
for i in range(100):
    interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]["index"])
t2 = timeit.default_timer()
t = round((t2 - t1), 3)


plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(output_data, image_label, img)
plt.subplot(1, 2, 2)
plot_value_array(output_data[0], image_label, t)
plt.show()
print(output_data)
