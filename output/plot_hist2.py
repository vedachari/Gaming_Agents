# SYSTEM IMPORTS
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# PYTHON PROJECT IMPORTS


def main() -> None:
    parser = ap.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the path to where MinimaxRuntimes.csv is")
    args = parser.parse_args()


    if not os.path.exists(args.filepath):
        raise Exception("ERROR: filepath [%s] does not exist!" % args.filepath)


    data_df: pd.DataFrame = pd.read_csv(args.filepath)
    # print(data_df.head())
    data_df.columns = [x.strip().rstrip() for x in data_df.columns] # remove extra whitespace around column names if any

    # data will be much easier to work with if we have it segmented into a nested dictionary wit keys [depth][numChildren]
    data_dict: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    # get some values for the histogram
    min_depth: int = np.inf
    max_depth: int = -np.inf

    min_num_children: int = np.inf
    max_num_children: int = -np.inf

    for idx, row in tqdm(data_df.iterrows(), desc="parsing data", total=len(data_df)):
        # print(row)
        depth: int = row["depth"]
        num_children: int = row["num_children"]

        min_depth = min(min_depth, depth)
        max_depth = max(max_depth, depth)

        min_num_children = min(min_num_children, num_children)
        max_num_children = max(max_num_children, num_children)

        data_dict[depth][num_children].append(row["runtime_ms"])


    # get min, avg, and max vals for each entry in the dict (i.e. squash the list of observations into avg + errbars)
    data_statistics: Dict[int, Dict[int, Tuple[float, float, float]]] = defaultdict(dict)
    for depth, depth_dict in data_dict.items():
        for num_children, runtimes in depth_dict.items():
            data_statistics[depth][num_children] = (np.min(runtimes), np.mean(runtimes), np.max(runtimes))


    # when we plot things, keep colors consistent
    depth_2_plot_colors: Dict[int, str] = dict()


    # make the figure
    fig = plt.figure(layout="constrained")
    gs = GridSpec(2, max_depth+1, figure=fig)


    main_ax = fig.add_subplot(gs[0, :])
    # plt.subplot(2, max_depth+1, int((max_depth + 1)/2))

    # make a series of histograms on the same plot starting from the largest depth
    # (which should have the smallest values and go in the back of the plot)
    # to the smallest depth (which should have the smallest values and go in the front of the plot)
    for depth in sorted(data_statistics.keys(), key=lambda x: -x): # sort in reverse order
        num_children: List[int] = list()

        runtime_mins: List[float] = list()
        runtime_avgs: List[float] = list()
        runtime_maxs: List[float] = list()

        for n, (min_val, avg_val, max_val) in data_statistics[depth].items():
            num_children.append(n)
            runtime_mins.append(min_val)
            runtime_avgs.append(avg_val)
            runtime_maxs.append(max_val)

        errs = np.vstack([np.array(runtime_avgs) - np.array(runtime_mins),
                          np.array(runtime_maxs) - np.array(runtime_avgs)])

        # now that the data is all separated, draw the bar
        y = main_ax.bar(num_children, runtime_avgs, label="depth=%s" % depth, yerr=errs, capsize=2)
        depth_2_plot_colors[depth] = y[-1].get_facecolor()

    main_ax.set_title("runtime (ms) vs # of children")
    main_ax.set_ylabel("runtime (ms)")
    main_ax.set_xlabel("# of children")
    main_ax.legend()



    # now plot each depth as its own graph so we can get a better view at smaller depths
    for plot_idx, depth in enumerate(sorted(data_statistics.keys(), key=lambda x: -x)): # sort in reverse order
        # plt.subplot(2, max_depth+1, max_depth + plot_idx + 2)
        ax = fig.add_subplot(gs[1, depth])
        num_children: List[int] = list()

        runtime_mins: List[float] = list()
        runtime_avgs: List[float] = list()
        runtime_maxs: List[float] = list()

        for n, (min_val, avg_val, max_val) in data_statistics[depth].items():
            num_children.append(n)
            runtime_mins.append(min_val)
            runtime_avgs.append(avg_val)
            runtime_maxs.append(max_val)

        errs = np.vstack([np.array(runtime_avgs) - np.array(runtime_mins),
                          np.array(runtime_maxs) - np.array(runtime_avgs)])


        # now that the data is all separated, draw the bar
        ax.bar(num_children, runtime_avgs, label="depth=%s" % depth, color=depth_2_plot_colors[depth], yerr=errs, capsize=2)
        ax.set_title("depth=%s runtime (ms) vs # of children" % depth)
        ax.set_ylabel("runtime (ms)")
        ax.set_xlabel("# of children")
    plt.show()


if __name__ == "__main__":
    main()

