import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from micromlgen.utils import port_array

image_name = "5.jpg"

image = Image.open(image_name).convert("L")
image = image.resize((28, 28), Image.Resampling.LANCZOS)
image = np.array(image)
image = 255 - image


def plot_image(img):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(img, cmap="binary")
    plt.show()


plot_image(image)
image = image.astype("float32") / 255.0
c_array = port_array(image.flatten())
print(c_array)
