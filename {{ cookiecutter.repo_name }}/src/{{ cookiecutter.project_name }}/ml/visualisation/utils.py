import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt


def initi(size: int = 20):
    matplotlib.rcParams.update({"font.size": size})
    plt.rcParams["text.usetex"] = True
    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
    rc("font", **{"family": "serif", "serif": ["Palatino"]})
    rc("text", usetex=True)
