#!/usr/bin/env python3
import argparse
import json

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
from src.metrics import accuracy, compute_all_metrics
from src.models.classifier import Classifier
from src.optimizers.second_order_optimizer import second_order_helper
from src.optimizers.utils import initialize_optimizer
from src.utils import open_pdf, plot_lines, print_model_info, set_seed


def helper(model, optimizer, data_loader, criterion):
    def closure():
        x, y = next(data_loader)
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()

        x, y = Variable(x), Variable(y.squeeze())

        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(
            input=y_hat,
            target=y,
        )
        loss.backward()

        return loss.data

    return closure


# Define a function to train a model, while also evaluating its performance after each iteration
def train_and_evaluate(
    model: nn.Module,
    optimizer_name: str,
    train_set: torch.utils.data.Dataset,
    iters,
    lr: float,
    batch_size: int,
    test_sets: [torch.utils.data.Dataset],
    test_size: int = 512,
    performance: dict[str, list[float]] = None,
    **kwargs,
):
    """Function to train a [model] on a given [dataset],
    while evaluating after each training iteration on [testset]."""

    if performance is None:
        performance = {f"task_{i}": [] for i in range(1, len(test_sets) + 1)}

    criterion = torch.nn.CrossEntropyLoss()

    data_loader = iter(
        torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=True
        )
    )
    iters_left = len(data_loader) + 1

    if optimizer_name == "Second-Order":
        kwargs["data_loader"] = data_loader
        kwargs["criterion"] = criterion

    optimizer = initialize_optimizer(
        model, optimizer_name=optimizer_name, lr=lr, **kwargs
    )

    model.train()
    progress_bar = tqdm.tqdm(range(1, iters + 1))

    if optimizer_name == "Entropy-SGD":
        n_iter_per_step = kwargs.get("L", 0) + 1
    elif optimizer_name == "C-Flat":
        n_iter_per_step = 4
    else:
        n_iter_per_step = 1

    for _ in range(1, iters + 1):
        # Collect data from training set and compute the loss
        iters_left -= n_iter_per_step
        if iters_left < n_iter_per_step:
            data_loader = iter(
                torch.utils.data.DataLoader(
                    train_set, batch_size=batch_size, shuffle=True, drop_last=True
                )
            )
            iters_left = len(data_loader)

        # Calculate test accuracy (in %)
        for i, test_set in enumerate(test_sets):
            acc = 100 * accuracy(model, test_set, test_size=test_size, batch_size=512)
            performance[f"task_{i + 1}"].append(acc)

        if optimizer_name == "Entropy-SGD":
            loss = optimizer.step(
                helper(model, optimizer, data_loader, criterion), model, criterion
            )
        elif optimizer_name == "Second-Order":
            loss = optimizer.step(
                closure=second_order_helper(model, optimizer, data_loader, criterion)
            )
        else:
            loss = optimizer.step(
                closure=helper(model, optimizer, data_loader, criterion)
            )

        progress_bar.set_description(
            "<CLASSIFIER> | training loss: {loss:.3} | test accuracy: {prec:.3}% |".format(
                loss=loss.item(), prec=performance[f"task_{1}"][-1]
            )
        )
        progress_bar.update(1)
    progress_bar.close()


def main(args):
    ################## INITIAL SET-UP ##################
    set_seed(args.random_seed)

    # Specify directories, and if needed create them
    p_dir = "./store/plots"
    d_dir = "./store/data"
    if not os.path.isdir(p_dir):
        print("Creating directory: {}".format(p_dir))
        os.makedirs(p_dir)
    if not os.path.isdir(d_dir):
        os.makedirs(d_dir)
        print("Creating directory: {}".format(d_dir))

    os.makedirs("./store/metrics", exist_ok=True)

    # Open pdf for plotting
    plot_name = f"stability_gap_example-{args.optimizer}-{args.lr}-{args.num_iters}-{args.random_seed}"
    if args.optimizer == "Entropy-SGD":
        plot_name += f"-L{args.L}-{args.scale}"
    elif args.optimizer == "C-Flat":
        plot_name += f"-rho{args.rho}-lamb{args.lamb}-baseopt{args.base_optimizer}-baseoptlr{args.base_optimizer_lr}"

    cache_name = f"{plot_name}.json"
    full_plot_name = "{}/{}.pdf".format(p_dir, plot_name)
    pp = open_pdf(full_plot_name)
    figure_list = []

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
    rotations = [0, 160, 80]

    # Specify for each task the transformed train- and test set
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

    kwargs = {}

    if args.optimizer == "Entropy-SGD":
        kwargs["L"] = args.L
        kwargs["scale"] = args.scale
    elif args.optimizer == "Second-Order" or args.optimizer == "C-Flat":
        kwargs["lamb"] = args.lamb
        kwargs["base_optimizer"] = initialize_optimizer(
            model, args.base_optimizer, lr=args.base_optimizer_lr
        )
        if args.optimizer == "C-Flat":
            kwargs["rho"] = args.rho

    # Define a list to keep track of the performance on task 1 after each iteration
    performance_tasks = {f"task_{i}": [] for i in range(1, len(test_datasets) + 1)}

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
            train_set=joint_dataset,
            iters=iters,
            lr=lr,
            batch_size=batch_size_to_use,
            test_sets=test_datasets[: task_id + 1],
            test_size=test_size,
            performance=performance_tasks,
            **kwargs,
        )
        for i in range(task_id + 1, n_tasks):
            performance_tasks[f"task_{i + 1}"].extend(
                [None for _ in range(args.num_iters)]
            )

    ################## PLOTTING ##################

    ## Plot per-iteration performance curve
    figure = plot_lines(
        list(performance_tasks.values()),
        x_axes=list(range(n_tasks * iters)),
        line_names=[f"Task {i}" for i in range(1, len(test_datasets) + 1)],
        title="Performance on Task 1 throughout 'Incremental Joint Training'",
        ylabel="Test Accuracy (%) on Task 1",
        xlabel="Total number of training iterations",
        figsize=(10, 5),
        v_line=[iters * (i + 1) for i in range(n_tasks - 1)],
        v_label="Task switch",
        ylim=(0, 100),
    )
    figure_list.append(figure)

    ## Finalize the pdf with the plots
    for figure in figure_list:
        pp.savefig(figure)

    pp.close()

    metrics = compute_all_metrics(performance_tasks)
    print(metrics)

    with open(cache_name, "w") as f:
        json.dump(performance_tasks, f)

    print("\nGenerated plot: {}\n".format(full_plot_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_iters", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=2000)
    parser.add_argument("--random_seed", type=int, default=42)

    # Entropy-SGD parameters
    parser.add_argument("--L", type=int, default=5)
    parser.add_argument("--scale", type=float, default=1e-2)

    # C-Flat parameters (and Second-Order)
    parser.add_argument("--base_optimizer", type=str, default="SGD")
    parser.add_argument("--base_optimizer_lr", type=float, default=0.1)
    parser.add_argument("--lamb", type=float, default=0.2)
    parser.add_argument("--rho", type=float, default=0.2)

    args = parser.parse_args()
    main(args)
