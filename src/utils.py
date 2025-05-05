import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from torchvision import transforms
from torch.utils.data import DataLoader

from src.data.available import AVAILABLE_TRANSFORMS


def count_parameters(model, verbose=True):
    """Count number of parameters, print to screen."""
    total_params = learnable_params = fixed_params = 0
    for param in model.parameters():
        n_params = index_dims = 0
        for dim in param.size():
            n_params = dim if index_dims == 0 else n_params * dim
            index_dims += 1
        total_params += n_params
        if param.requires_grad:
            learnable_params += n_params
        else:
            fixed_params += n_params
    if verbose:
        print(
            "--> this network has {} parameters (~{} million)".format(
                total_params, round(total_params / 1000000, 1)
            )
        )
        print(
            "       of which: - learnable: {} (~{} million)".format(
                learnable_params, round(learnable_params / 1000000, 1)
            )
        )
        print(
            "                 - fixed: {} (~{} million)".format(
                fixed_params, round(fixed_params / 1000000, 1)
            )
        )
    return total_params, learnable_params, fixed_params


def print_model_info(model, message=None):
    """Print information on [model] onto the screen."""
    print(55 * "-" if message is None else " {} ".format(message).center(55, "-"))
    print(model)
    print(55 * "-")
    _ = count_parameters(model)


def open_pdf(full_path):
    return PdfPages(full_path)


def plot_lines(
    list_with_lines,
    x_axes=None,
    line_names=None,
    colors=None,
    title=None,
    title_top=None,
    xlabel=None,
    ylabel=None,
    ylim=None,
    figsize=None,
    list_with_errors=None,
    errors="shaded",
    x_log=False,
    with_dots=False,
    linestyle="solid",
    h_line=None,
    h_label=None,
    h_error=None,
    h_lines=None,
    h_colors=None,
    h_labels=None,
    h_errors=None,
    v_line=None,
    v_label=None,
):
    """Generates a figure containing multiple lines in one plot.

    :param list_with_lines: <list> of all lines to plot (with each line being a <list> as well)
    :param x_axes:          <list> containing the values for the x-axis
    :param line_names:      <list> containing the names of each line
    :param colors:          <list> containing the colors of each line
    :param title:           <str> title of plot
    :param title_top:       <str> text to appear on top of the title
    :return: f:             <figure>
    """

    # if needed, generate default x-axis
    if x_axes == None:
        n_obs = len(list_with_lines[0])
        x_axes = list(range(n_obs))

    # if needed, generate default line-names
    if line_names == None:
        n_lines = len(list_with_lines)
        line_names = ["line " + str(line_id) for line_id in range(n_lines)]

    # make plot
    size = (12, 7) if figsize is None else figsize
    f, axarr = plt.subplots(1, 1, figsize=size)

    # add error-lines / shaded areas
    if list_with_errors is not None:
        for line_id, name in enumerate(line_names):
            if errors == "shaded":
                axarr.fill_between(
                    x_axes,
                    list(
                        np.array(list_with_lines[line_id])
                        + np.array(list_with_errors[line_id])
                    ),
                    list(
                        np.array(list_with_lines[line_id])
                        - np.array(list_with_errors[line_id])
                    ),
                    color=None if (colors is None) else colors[line_id],
                    alpha=0.25,
                )
            else:
                axarr.plot(
                    x_axes,
                    list(
                        np.array(list_with_lines[line_id])
                        + np.array(list_with_errors[line_id])
                    ),
                    label=None,
                    color=None if (colors is None) else colors[line_id],
                    linewidth=1,
                    linestyle="dashed",
                )
                axarr.plot(
                    x_axes,
                    list(
                        np.array(list_with_lines[line_id])
                        - np.array(list_with_errors[line_id])
                    ),
                    label=None,
                    color=None if (colors is None) else colors[line_id],
                    linewidth=1,
                    linestyle="dashed",
                )

    # mean lines
    for line_id, name in enumerate(line_names):
        axarr.plot(
            x_axes,
            list_with_lines[line_id],
            label=name,
            color=None if (colors is None) else colors[line_id],
            linewidth=4,
            marker="o" if with_dots else None,
            linestyle=linestyle if type(linestyle) == str else linestyle[line_id],
        )

    # add horizontal line
    if h_line is not None:
        axarr.axhline(y=h_line, label=h_label, color="grey")
        if h_error is not None:
            if errors == "shaded":
                axarr.fill_between(
                    [x_axes[0], x_axes[-1]],
                    [h_line + h_error, h_line + h_error],
                    [h_line - h_error, h_line - h_error],
                    color="grey",
                    alpha=0.25,
                )
            else:
                axarr.axhline(
                    y=h_line + h_error,
                    label=None,
                    color="grey",
                    linewidth=1,
                    linestyle="dashed",
                )
                axarr.axhline(
                    y=h_line - h_error,
                    label=None,
                    color="grey",
                    linewidth=1,
                    linestyle="dashed",
                )

    # add horizontal lines
    if h_lines is not None:
        h_colors = colors if h_colors is None else h_colors
        for line_id, new_h_line in enumerate(h_lines):
            axarr.axhline(
                y=new_h_line,
                label=None if h_labels is None else h_labels[line_id],
                color=None if (h_colors is None) else h_colors[line_id],
            )
            if h_errors is not None:
                if errors == "shaded":
                    axarr.fill_between(
                        [x_axes[0], x_axes[-1]],
                        [
                            new_h_line + h_errors[line_id],
                            new_h_line + h_errors[line_id],
                        ],
                        [
                            new_h_line - h_errors[line_id],
                            new_h_line - h_errors[line_id],
                        ],
                        color=None if (h_colors is None) else h_colors[line_id],
                        alpha=0.25,
                    )
                else:
                    axarr.axhline(
                        y=new_h_line + h_errors[line_id],
                        label=None,
                        color=None if (h_colors is None) else h_colors[line_id],
                        linewidth=1,
                        linestyle="dashed",
                    )
                    axarr.axhline(
                        y=new_h_line - h_errors[line_id],
                        label=None,
                        color=None if (h_colors is None) else h_colors[line_id],
                        linewidth=1,
                        linestyle="dashed",
                    )

    # add vertical line(s)
    if v_line is not None:
        if type(v_line) == list:
            for id, new_line in enumerate(v_line):
                axarr.axvline(
                    x=new_line, label=v_label if id == 0 else None, color="black"
                )
        else:
            axarr.axvline(x=v_line, label=v_label, color="black")

    # finish layout
    # -set y-axis
    if ylim is not None:
        axarr.set_ylim(ylim)
    # -add axis-labels
    if xlabel is not None:
        axarr.set_xlabel(xlabel)
    if ylabel is not None:
        axarr.set_ylabel(ylabel)
    # -add title(s)
    if title is not None:
        axarr.set_title(title)
    if title_top is not None:
        f.suptitle(title_top)
    # -add legend
    if line_names is not None:
        axarr.legend()
    # -set x-axis to log-scale
    if x_log:
        axarr.set_xscale("log")

    # return the figure
    return f


def get_data_loader(dataset, batch_size, cuda=False, drop_last=False, augment=False):
    """Return <DataLoader>-object for the provided <DataSet>-object [dataset]."""

    # If requested, make copy of original dataset to add augmenting transform (without altering original dataset)
    if augment:
        dataset_ = copy.deepcopy(dataset)
        dataset_.transform = transforms.Compose(
            [dataset.transform, *AVAILABLE_TRANSFORMS["augment"]]
        )
    else:
        dataset_ = dataset

    # Create and return the <DataLoader>-object
    return DataLoader(
        dataset_,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        **({"num_workers": 0, "pin_memory": True} if cuda else {}),
    )


def checkattr(args, attr):
    """Check whether attribute exists, whether it's a boolean and whether its value is True."""
    return (
        hasattr(args, attr)
        and type(getattr(args, attr)) == bool
        and getattr(args, attr)
    )
