import matplotlib.pyplot as plt
import seaborn as sns


def visualize_metrics(dataframe, save_path=None, metric_list=None, figsize=(8, 6), show=True, hue="algorithm",
                      swarm=False):
    if not metric_list:
        metric_list = ['MCC', 'F1', 'BA', 'AUC']

    zero2one_scores = {'F1', 'BA', 'AUC', 'Precision', 'Recall'}
    # not using the more convenient set intersection to keep order of metric list
    zero2one_scores = [metric for metric in metric_list if metric in zero2one_scores]
    minus_one2one_scores = {'MCC', }
    minus_one2one_scores = [metric for metric in metric_list if metric in minus_one2one_scores]

    unknown_metrics = set(metric_list) - set(zero2one_scores) - set(minus_one2one_scores)

    if unknown_metrics:
        raise ValueError("Unknown metric: {}".format(", ".join(unknown_metrics)))

    n_grid_cols = round(12 * len(metric_list))
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(11, n_grid_cols, wspace=0, hspace=0)
    len_left = 10 * len(zero2one_scores)
    len_right = 10 * len(minus_one2one_scores)
    ax1 = fig.add_subplot(gs[0:9, :len_left])
    ax2 = fig.add_subplot(gs[0:9, -len_right:])
    ax3 = fig.add_subplot(gs[-1, :])

    if swarm:
        vis = sns.stripplot
        kwargs = {"dodge": True}
    else:
        vis = sns.boxplot
        kwargs = {}

    left_plot = vis(data=dataframe.query("metric.isin(@zero2one_scores)"),
                    x="metric",
                    y="value",
                    hue=hue,
                    order=zero2one_scores,
                    hue_order=sorted(dataframe[hue].unique()),
                    ax=ax1,
                    **kwargs)
    right_plot = vis(data=dataframe.query("metric.isin(@minus_one2one_scores)"),
                     x="metric",
                     y="value",
                     hue=hue,
                     order=minus_one2one_scores,
                     hue_order=sorted(dataframe[hue].unique()),
                     ax=ax2,
                     **kwargs)

    ax2.get_legend().remove()
    ax1.legend(loc='lower left', ncol=3)

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
    ax3.axis('off')

    fig.subplots_adjust(bottom=0.0, top=0.95, right=0.95)
    if save_path:
        plt.savefig(save_path, dpi=300)
    if not show:
        plt.close()
    return fig, (ax1, ax2, ax3)
