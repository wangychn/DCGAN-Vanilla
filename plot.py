import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import os

plt.ion()

def training_process_plot(G_losses, D_losses, plot_path, plot_name=""):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('DCGAN Training Process Live Graph')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.plot(G_losses)
    plt.plot(D_losses)
    plt.legend(['Generator Loss', 'Discriminator Loss'])
    plt.ylim(ymin=0)
    if G_losses:
        plt.text(len(G_losses)-1, G_losses[-1], str(round(G_losses[-1], 4)), color='orange')
    if D_losses:
        plt.text(len(D_losses)-1, D_losses[-1], str(round(D_losses[-1], 4)), color='green')

    plt.show(block=False)
    plt.pause(.1)

    # Save the plot every time it's updated
    if plot_name != "":
        os.makedirs(plot_path, exist_ok=True)
        print("SAVED GRAPH!")
        plt.savefig(os.path.join(plot_path, plot_name))


