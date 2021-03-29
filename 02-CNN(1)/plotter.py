import numpy as np

np.seterr(divide='ignore', invalid='ignore')
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt


def Plotter(test_images, model):
    img_tensor = test_images[0]
    img_tensor = np.expand_dims(img_tensor, axis=0)

    layer_outputs = [layer.output for layer in model.layers[:6]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(img_tensor)

    layer_names = []
    for layer in model.layers[:6]:
        layer_names.append(layer.name)

    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]

        size = layer_activation.shape[1]

        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]

                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255.).astype('uint8')

                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image

        scale = 1. / size
        print('레이어 이름: ', layer_name)
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.savefig("plot.png")

    plt.show()