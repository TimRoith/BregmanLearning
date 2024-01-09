import numpy as np
import matplotlib.pyplot as plt


def visualize_layer_weights(layer):
    # Extracting the layer weights
    layer_weights = layer.weight.data
    layer_weights = layer_weights.view(layer_weights.shape[0], -1)

    # Converting the tensor to numpy for visualization
    layer_weights_np = layer_weights.cpu().numpy()

    layer_weights_np = np.abs(layer_weights_np)
    height, width = layer_weights_np.shape
    plt.figure(figsize=(width * 20 / 100, height * 20 / 100))
    # Plotting the weights
    plt.imshow(layer_weights_np, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Weights of layer {layer}")
    plt.xlabel('Neurons')
    plt.ylabel('Inputs/Connections')
    plt.show()
