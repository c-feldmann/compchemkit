
import numpy as np
import numpy.typing as npt
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns


def evaluate_classification(
    y_true: npt.NDArray[np.int_],
    y_predicted: npt.NDArray[np.int_],
    y_score: npt.NDArray[np.float_] | None = None,
    nan2zero: bool = False,
) -> dict[str, float]:
    if len(y_true) != len(y_predicted):
        raise IndexError("y_true and y_predicted are not of equal size!")
    if y_score is not None:
        if len(y_true) != len(y_score):
            raise IndexError("y_true and y_score are not of equal size!")

    fill = 0 if nan2zero else np.nan

    if sum(y_predicted) == 0:
        mcc = fill
        precision = fill
    else:
        mcc = metrics.matthews_corrcoef(y_true, y_predicted)
        precision = metrics.precision_score(y_true, y_predicted)

    result_dict = {
        "MCC": mcc,
        "F1": metrics.f1_score(y_true, y_predicted),
        "BA": metrics.balanced_accuracy_score(y_true, y_predicted),
        "Precision": precision,
        "Recall": metrics.recall_score(y_true, y_predicted),
        "Average Precision": metrics.average_precision_score(y_true, y_predicted),
        "set_size": y_true.shape[0],
        "pos_true": len([x for x in y_true if x == 1]),
        "neg_true": len([x for x in y_true if x == 0]),
        "pos_predicted": len([x for x in y_predicted if x == 1]),
        "neg_predicted": len([x for x in y_predicted if x == 0]),
    }

    if y_score is not None:
        result_dict["AUC"] = metrics.roc_auc_score(y_true, y_score)
    else:
        result_dict["AUC"] = np.nan

    return result_dict


def evaluate_regression(
    y_true: npt.NDArray[np.float_], y_predicted: npt.NDArray[np.float_]
) -> dict[str, float]:
    if len(y_true) != len(y_predicted):
        raise IndexError("y_true and y_predicted are not of equal size!")

    result_dict = {
        "explained_variance": metrics.explained_variance_score(y_true, y_predicted),
        "max_error": metrics.max_error(y_true, y_predicted),
        "mean_absolute_error": metrics.mean_absolute_error(y_true, y_predicted),
        "mean_squared_error": metrics.mean_squared_error(y_true, y_predicted),
        "root_mean_squared_error": metrics.mean_squared_error(
            y_true, y_predicted, squared=False
        ),
        "mean_squared_log_error": metrics.mean_squared_log_error(y_true, y_predicted),
        "median_absolute_error": metrics.median_absolute_error(y_true, y_predicted),
        "r2": metrics.r2_score(y_true, y_predicted),
        "mean_poisson_deviance": metrics.mean_poisson_deviance(y_true, y_predicted),
        "mean_gamma_deviance": metrics.mean_gamma_deviance(y_true, y_predicted),
        "mean_absolute_percentage_error": metrics.mean_absolute_percentage_error(
            y_true, y_predicted
        ),
        "d2_absolute_error_score": metrics.d2_absolute_error_score(y_true, y_predicted),
        "d2_pinball_score": metrics.d2_pinball_score(y_true, y_predicted),
        "d2_tweedie_score": metrics.d2_tweedie_score(y_true, y_predicted),
    }
    return result_dict


def visualize_metrics(
    dataframe: pd.DataFrame,
    save_path: str | None = None,
    metric_list: list[str] | None = None,
    figsize: tuple[int, int] = (8, 6),
    show: bool = True,
    hue: str = "algorithm",
    swarm: bool = False,
    hue_order: list[str] | None = None,
    dpi: int = 300,
) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    if not metric_list:
        metric_list = ["MCC", "F1", "BA", "AUC"]

    zero2one_scores = ["F1", "BA", "AUC", "Precision", "Recall", "Average Precision"]
    # not using the more convenient set intersection to keep order of metric list
    zero2one_scores = [metric for metric in metric_list if metric in zero2one_scores]

    minus_one2one_scores = ["MCC"]
    minus_one2one_scores = [
        metric for metric in metric_list if metric in minus_one2one_scores
    ]

    unknown_metrics = (
        set(metric_list) - set(zero2one_scores) - set(minus_one2one_scores)
    )
    if unknown_metrics:
        raise ValueError("Unknown metric(s): {}".format(", ".join(unknown_metrics)))

    n_grid_cols = round(12 * len(metric_list))
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(11, n_grid_cols, wspace=0, hspace=0)
    len_left = 10 * len(zero2one_scores)
    len_right = 10 * len(minus_one2one_scores)
    ax1 = fig.add_subplot(gs[0:9, :len_left])
    ax2 = fig.add_subplot(gs[0:9, -len_right:])
    ax3 = fig.add_subplot(gs[-1, :])

    if not hue_order:
        hue_order = sorted(dataframe[hue].unique())

    if swarm:
        vis = sns.stripplot
        kwargs = {"dodge": True}
    else:
        vis = sns.boxplot
        kwargs = {}

    _ = vis(
        data=dataframe.query("metric.isin(@zero2one_scores)"),
        x="metric",
        y="value",
        hue=hue,
        order=zero2one_scores,
        hue_order=hue_order,
        ax=ax1,
        **kwargs
    )
    _ = vis(
        data=dataframe.query("metric.isin(@minus_one2one_scores)"),
        x="metric",
        y="value",
        hue=hue,
        order=minus_one2one_scores,
        hue_order=hue_order,
        ax=ax2,
        **kwargs
    )

    ax2.get_legend().remove()
    ax1.legend(loc="lower left", ncol=3)

    ax1.set_ylim(-0.05, 1.05)
    ax2.set_ylim(-1.1, 1.1)

    axs_0_xlim = ax1.get_xlim()
    ax1.hlines(0.5, xmin=axs_0_xlim[0], xmax=axs_0_xlim[1], ls="--", color="gray")
    ax1.set_xlim(axs_0_xlim)

    axs_2_xlim = ax2.get_xlim()
    ax2.hlines(0, xmin=axs_2_xlim[0], xmax=axs_2_xlim[1], ls="--", color="gray")
    ax2.set_xlim(axs_2_xlim)

    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    handles, labels = ax1.get_legend_handles_labels()
    ax3.legend(handles, labels, ncol=len(handles), loc="center")
    ax1.get_legend().remove()
    ax3.axis("off")

    fig.subplots_adjust(bottom=0.0, top=0.95, right=0.95)
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    if not show:
        plt.close()
    return fig, (ax1, ax2, ax3)
