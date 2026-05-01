"""Top-level wrapper for plotting benchmark results."""

from src.plot_results import plot_all


if __name__ == "__main__":
    for output in plot_all():
        print(output)
