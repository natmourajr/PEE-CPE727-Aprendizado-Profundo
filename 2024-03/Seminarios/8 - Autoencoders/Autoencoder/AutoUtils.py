import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch


def PlotLoss(results):
    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(results["train_loss"], label="train loss")
    ax.plot(results["validation_loss"], label="validation loss")
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.grid()
    ax.legend()
    plt.show()


def PlotNumbers(test_dataset, model, device):
    numbers_to_plot = 3
    fig, ax = plt.subplots(2,3, figsize=(12,4))

    for index, (image, _) in enumerate(test_dataset): 
        if (index >= numbers_to_plot):
            break
        item = image.reshape(-1, 28, 28)
        ax[0,index].imshow(item[0].detach().numpy())
        model.eval()
        item = model(image.reshape(-1, input_shape).to(device))
        item = item.reshape(-1, 28, 28).cpu()
        ax[1,index].imshow(item[0].detach().numpy())

    plt.show()


def PlotKKFoldLoss(results):
    copyResults = results.copy()
    copyResults.pop("epoch")
    pdResults = pd.DataFrame(copyResults)

    fig, ax = plt.subplots(figsize=(6,4))
    mean_train_loss = None
    mean_validation_loss = 0

    for nFold in pdResults["fold"].unique():
        pdFilter = pdResults[pdResults["fold"] == nFold]
        axis = np.arange(len(pdFilter["train_loss"]))
        if nFold == 0:
            mean_train_loss = pdFilter["train_loss"].to_numpy()
            mean_validation_loss = pdFilter["validation_loss"].to_numpy()
        else:
            mean_train_loss += pdFilter["train_loss"].to_numpy()
            mean_validation_loss += pdFilter["validation_loss"].to_numpy()
    divisor = len(pdResults["fold"].unique())
    ax.plot(axis, mean_train_loss / divisor, label="train loss")
    ax.plot(axis, mean_validation_loss / divisor, label="validation loss")
    ax.grid()
    # ax.set_xscale('log')
    ax.legend()
    plt.show()

def PlotIris(test_dataset, target, model, device):
    numbers_to_plot = 3
    fig, ax = plt.subplots(1,2, figsize=(12,4))


    ax[0].scatter(test_dataset[:, 0], test_dataset[:, 1], c=target)
    model.eval()
    item = model(torch.tensor(test_dataset).to(device).float())
    item = item.cpu().detach().numpy()
    ax[1].scatter(item[:, 0], item[:, 1], c=target)
    plt.show()

    