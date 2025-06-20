def plot_hist(
    data,
    num_bins=20,
    *,
    ax=None,
    color='g',
    xlabel='Variable',
    ylabel='Count',
    title_prefix='Continuous Uniform Distribution',
    **hist_kw,
):
    ax = ax or plt.gca()
    ax.hist(data, bins=num_bins, color=color, alpha=0.9, **hist_kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title_prefix} with {num_bins} bins')
    return ax
