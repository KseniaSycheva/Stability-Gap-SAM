#!/usr/bin/env python3
import argparse

# Standard libraries
import os
import tqdm
import numpy as np

# Pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

# For visualization
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Load custom-written code
from src.data.manipulate import TransformedDataset
from src.metrics import test_acc
from src.models.classifier import Classifier
from src.optimizers.utils import initialize_optimizer
from src.utils import open_pdf, plot_lines, print_model_info


# Define a function to train a model, while also evaluating its performance after each iteration
def train_and_evaluate(
    model: nn.Module, optimizer_name: str, trainset: torch.utils.data.Dataset, iters, lr: float, batch_size: int, testset: torch.utils.data.Dataset, test_size: int = 512, performance: list[float] = None
):
    """Function to train a [model] on a given [dataset],
    while evaluating after each training iteration on [testset]."""

    if performance is None:
        performance = []

    criterion = torch.nn.CrossEntropyLoss()

    data_loader = iter(
        torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, drop_last=True
        )
    )
    iters_left = len(data_loader) + 1

    optimizer = initialize_optimizer(model, optimizer_name=optimizer_name, lr=lr)
    if optimizer_name == "Entropy-SGD":
        def helper():
            def feval():
                xx, yy = next(data_loader)
                if torch.cuda.is_available():
                    xx, yy = xx.cuda(), yy.cuda()

                xx, yy = Variable(xx), Variable(yy.squeeze())

                optimizer.zero_grad()
                yh = model(xx)
                f = criterion.forward(yh, yy)
                f.backward()

                prec1, = accuracy(yh.data, yy.data, topk=(1,))
                err = 100. - prec1[0]
                return f.data[0], err

            return feval

    model.train()
    progress_bar = tqdm.tqdm(range(1, iters + 1))

    for _ in range(1, iters + 1):
        optimizer.zero_grad()

        # Collect data from training set and compute the loss
        iters_left -= 1
        if iters_left == 0:
            data_loader = iter(torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                           shuffle=True, drop_last=True))
            iters_left = len(data_loader)

        x, y = next(data_loader)
        if model.cuda:
            x, y = x.cuda(), y.cuda()
        y_hat = model(x)
        loss = criterion(
            input=y_hat, target=y,
        )

        # Calculate test accuracy (in %)
        accuracy = 100 * test_acc(
            model, testset, test_size=test_size, verbose=False, batch_size=512
        )
        performance.append(accuracy)

        # Take gradient step
        loss.backward()

        if optimizer_name == "Entropy-SGD":
            optimizer.step(helper(), model, criterion)
        else:
            optimizer.step()
        progress_bar.set_description(
            "<CLASSIFIER> | training loss: {loss:.3} | test accuracy: {prec:.3}% |".format(
                loss=loss.item(), prec=accuracy
            )
        )
        progress_bar.update(1)
    progress_bar.close()


def main(args):
    ################## INITIAL SET-UP ##################

    # Specify directories, and if needed create them
    p_dir = "./store/plots"
    d_dir = "./store/data"
    if not os.path.isdir(p_dir):
        print("Creating directory: {}".format(p_dir))
        os.makedirs(p_dir)
    if not os.path.isdir(d_dir):
        os.makedirs(d_dir)
        print("Creating directory: {}".format(d_dir))

    # Open pdf for plotting
    plot_name = "stability_gap_example"
    full_plot_name = "{}/{}.pdf".format(p_dir, plot_name)
    pp = open_pdf(full_plot_name)
    figure_list = []

    print(args.optimizer)

    ################## CREATE TASK SEQUENCE ##################

    ## Download the MNIST dataset
    print("\n\n " + " LOAD DATA ".center(70, "*"))
    MNIST_trainset = datasets.MNIST(
        root="data/", train=True, download=True, transform=transforms.ToTensor()
    )
    MNIST_testset = datasets.MNIST(
        root="data/", train=False, download=True, transform=transforms.ToTensor()
    )
    config = {"size": 28, "channels": 1, "classes": 10}

    # Set for each task the amount of rotation to use
    rotations = [0, 80, 160]

    # Specify for each task the transformed train- and testset
    n_tasks = len(rotations)
    train_datasets = []
    test_datasets = []
    for rotation in rotations:
        train_datasets.append(
            TransformedDataset(
                MNIST_trainset,
                transform=transforms.RandomRotation(degrees=(rotation, rotation)),
            )
        )
        test_datasets.append(
            TransformedDataset(
                MNIST_testset,
                transform=transforms.RandomRotation(degrees=(rotation, rotation)),
            )
        )

    # Visualize the different tasks
    figure, axis = plt.subplots(1, n_tasks, figsize=(3 * n_tasks, 4))
    n_samples = 49
    for task_id in range(len(train_datasets)):
        # Show [n_samples] examples from the training set for each task
        data_loader = torch.utils.data.DataLoader(
            train_datasets[task_id], batch_size=n_samples, shuffle=True
        )
        image_tensor, _ = next(iter(data_loader))
        image_grid = make_grid(
            image_tensor, nrow=int(np.sqrt(n_samples)), pad_value=1
        )  # pad_value=0 would give black borders
        axis[task_id].imshow(np.transpose(image_grid.numpy(), (1, 2, 0)))
        axis[task_id].set_title("Task {}".format(task_id + 1))
        axis[task_id].axis("off")
    figure_list.append(figure)

    ################## SET UP THE MODEL ##################

    print("\n\n " + " DEFINE THE CLASSIFIER ".center(70, "*"))

    # Specify the architectural layout of the network to use
    fc_lay = 3  # --> number of fully-connected layers
    fc_units = 400  # --> number of units in each hidden layer

    # Define the model
    model = Classifier(
        image_size=config["size"],
        image_channels=config["channels"],
        classes=config["classes"],
        fc_layers=fc_lay,
        fc_units=fc_units,
        fc_bn=False,
    )
    model = model.to("cuda") if torch.cuda.is_available() else model

    # Print some model info to screen
    print_model_info(model)

    ################## TRAINING AND EVALUATION ##################

    print("\n\n" + " TRAINING + CONTINUAL EVALUATION ".center(70, "*"))

    # Specify the training parameters
    iters = args.num_iters
    lr = args.lr
    batch_size = args.batch_size
    test_size = args.test_size

    # Define a list to keep track of the performance on task 1 after each iteration
    performance_task1 = []

    # Iterate through the contexts
    for task_id in range(n_tasks):
        current_task = task_id + 1

        # Concatenate the training data of all tasks so far
        joint_dataset = torch.utils.data.ConcatDataset(train_datasets[:current_task])

        # Determine the batch size to use
        batch_size_to_use = current_task * batch_size

        # Train
        print("Training after arrival of Task {}:".format(current_task))
        train_and_evaluate(
            model,
            optimizer_name=args.optimizer,
            trainset=joint_dataset,
            iters=iters,
            lr=lr,
            batch_size=batch_size_to_use,
            testset=test_datasets[0],
            test_size=test_size,
            performance=performance_task1,
        )

    ################## PLOTTING ##################

    ## Plot per-iteration performance curve
    figure = plot_lines(
        [performance_task1],
        x_axes=list(range(n_tasks * iters)),
        line_names=["Incremental Joint"],
        title="Performance on Task 1 throughout 'Incremental Joint Training'",
        ylabel="Test Accuracy (%) on Task 1",
        xlabel="Total number of training iterations",
        figsize=(10, 5),
        v_line=[iters * (i + 1) for i in range(n_tasks - 1)],
        v_label="Task switch",
        ylim=(70, 100),
    )
    figure_list.append(figure)

    ## Finalize the pdf with the plots
    for figure in figure_list:
        pp.savefig(figure)

    pp.close()
    print("\nGenerated plot: {}\n".format(full_plot_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_iters", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=2000)

    args = parser.parse_args()
    main(args)
