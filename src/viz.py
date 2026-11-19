import pandas as pd
import matplotlib.pyplot as plt

def bar(ax, x, y, title, xlabel, ylabel, grid: bool = True):
    """
    Ritar ett stapeldiagram
    """
    ax.bar(x,y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid, axis="y")
    plt.suptitle("")
    return ax

def hist(ax, x, bins, title, xlabel, ylabel, grid: bool = True):
    """
    Ritar ett histogram
    """
    ax.hist(x,bins,edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid, axis="y")
    plt.suptitle("")
    return ax

def box(df, ax, column, by, title, xlabel, ylabel, grid: bool = True):
    """
    Ritar en boxplot
    """
    df.boxplot(column=column,by=by, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid, axis="y")
    plt.suptitle("")
    return ax

def scatter(ax, x, y, alpha, title, xlabel, ylabel, grid: bool = True, label=None,):
    """
    Ritar ett spridningsdiagram
    """
    ax.scatter(x=x, y=y, alpha=alpha, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid)
    plt.suptitle("")
    return ax

def plot_line(ax, x, y, color, label):
    """
    Ritar en linje
    """
    ax.plot(x, y, color=color, label=label)
    