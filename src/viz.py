import pandas as pd
import matplotlib.pyplot as plt

def bar(ax, x, y, title, xlabel, ylabel, grid: bool = True):
    ax.bar(x,y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid, axis="y")
    plt.suptitle("")
    return ax

def hist(ax, x, bins, title, xlabel, ylabel, grid: bool = True):
    ax.hist(x,bins,edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid, axis="y")
    plt.suptitle("")
    return ax

def box(df, ax, column, by, title, xlabel, ylabel, grid: bool = True):
    df.boxplot(column=column,by=by, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid, axis="y")
    plt.suptitle("")
    return ax