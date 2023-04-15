import numpy as np
import matplotlib.pyplot as plt  # pip install matplotlib
from PIL import Image
from micromlgen.utils import port_array  # pip install micromlgen
image = Image.open('input.jpg').convert('L')
image = image.resize((28, 28), Image.Resampling.LANCZOS)
im = np.array(image)
im = 255 - im


def plot_image(img):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(img, cmap='binary')
    plt.show()


plot_image(im)
im = im.astype('float32') / 255.0
c_array = port_array(im.flatten())
print(c_array)
