import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Opt_HW2.test.examples import *


def plot_path(x_history, func):
    if func == 1:
        fig = plt.figure()
        ax = Axes3D(fig)
        x = [1, 0, 0]
        y = [0, 1, 0]
        z = [0, 0, 1]
        verts = [list(zip(x, y, z))]
        poly_3d_collection = Poly3DCollection(verts, alpha=0.25)
        ax.add_collection3d(poly_3d_collection)
        scatter_min_path(ax, x_history, 1)
        plt.show()

    elif func == 2:
        x = np.arange(0,2)
        y = np.arange(0,1.1)
        X,Y = np.meshgrid(x,y)
        h1 = X + Y - 1
        h2 = Y - 1
        h3 = X - 2
        h4 = Y
        fig, ax = plt.subplots()
        ax.contour(X, Y, h1, levels=[0])
        ax.contour(X, Y, h2, levels=[0])
        ax.axhline(y=0, color='k')
        ax.axhline(y=1, color='k')
        ax.axvline(x=0, color='k')
        ax.axvline(x=2, color='k')

        x_fill = [0, 2, 2, 1]
        y_fill = [1, 1, 0, 0]
        plt.fill(x_fill, y_fill, facecolor='lightsalmon')
        scatter_min_path(ax, x_history, 2)
        plt.show()


def scatter_min_path(ax,x_history,f):
    x_history = np.asarray(x_history)
    x = x_history[:, 0]
    y = x_history[:, 1]
    if f == 1:
        z = x_history[:, 2]
        ax.plot(x,y,z,color="k",marker="*",linestyle="--")
        ax.set_zlabel('z')
    elif f == 2:
        ax.scatter(x, y, marker='x')
        annotate_path(ax, x_history)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Minimization path')


def annotate_path(ax,x_history):
    for i in range(1, x_history.shape[0]):
        ax.annotate('', xy=x_history[i], xytext=x_history[i - 1],arrowprops={'arrowstyle': '->', 'lw': 1})
