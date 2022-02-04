from enum import Enum

import numpy as np
from matplotlib import pyplot as plt

from sklearn.utils import check_random_state


class Color(Enum):
    BLUE = "#1e90ff"
    ORANGE = "#ffa500"
    GREEN = "#32CD32"
    MAGENTA = "#ff1e8a"
    GREYISH = "#8DABD3"


def get_reg_X_y(n_points=50, noise=.1, seed=1234):
    rng = check_random_state(seed)
    xs = rng.uniform(-10, 10, n_points)
    noise = rng.normal(scale=noise, size=n_points)
    y = np.sin(xs) + np.linspace(0, 1, n_points) + noise
    X = xs.reshape((-1, 1))
    return X, y


def generate_disk(n_points, radius=1., noise=.1, random_state=None):
    """Generate a noisy disk of points

    Parameters
    ----------
    n_points : int >0
        The number of points to generate
    radius : float, optional (default=1.)
        The expected radius of the circle
    noise : float, optional (default=.1)
        The variance of the gaussian radius perturbation
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    xs : array of shape [n_points]
        The x coordinate of the points. The nth points has coordinates
        (xs[n], ys[n])
    ys : array of shape [n_points]
        The y coordinate of the points. The nth points has coordinates
        (xs[n], ys[n])
    """
    # Build the random generator
    drawer = check_random_state(random_state)
    # Draw the angles
    thetas = drawer.uniform(0, 2*np.pi, n_points)
    # Draw the radius variations
    rhos = drawer.normal(0, noise, n_points)+radius
    # Transform to cartesian coordinates
    xs = rhos*np.cos(thetas)
    ys = rhos*np.sin(thetas)
    return xs, ys


def make_ellipse(n_points, radius=1., noise=.1, flattening=2., rotation=0.,
                 random_state=None):
    # A few preliminary computations
    drawer = check_random_state(random_state)
    sin = np.sin(rotation)
    cos = np.cos(rotation)
    dr = (flattening - 1.) / (flattening + 1.)

    xs1, ys1 = generate_disk(n_points, radius=radius, noise=noise,
                             random_state=drawer)
    scale_matrix = np.array([[1 + dr, 0], [0, 1 - dr]])
    rotation_matrix = np.array([[cos, -sin], [sin, cos]])
    ellipse_map = rotation_matrix.dot(scale_matrix)
    X = ellipse_map.dot(np.array([xs1, ys1]))

    return X.T


def get_cls_X_y(n_points1=200, n_points2=200, seed=12345):
    rng = check_random_state(seed)
    n1 = int(n_points1*0.9)
    X1 = make_ellipse(n1, flattening=2.3, rotation=0,
                      noise=.2, random_state=rng)
    y1 = np.zeros(n1, dtype=int)

    X3 = make_ellipse(n_points1 - n1, radius=0.1, flattening=1,
                      noise=.2, random_state=rng)
    y3 = np.zeros(n_points1 - n1, dtype=int)

    n2 = int(n_points2*.99)
    X2 = make_ellipse(n2, flattening=1.4, rotation=np.pi/2.1,
                      random_state=rng)
    y2 = np.ones(n2, dtype=int)
    X4 = make_ellipse(n_points2 - n2, radius=0.2, flattening=2,
                      rotation=0, random_state=rng)
    y4 = np.ones(n_points2 - n2, dtype=int)


    # Combine the ellipses
    X = np.vstack([X1, X2, X3, X4])
    y = np.concatenate([y1, y2, y3, y4])
    permutation = np.arange(len(y))
    rng.shuffle(permutation)

    return X[permutation], y[permutation]


def get_orange_blue_cmap():
    import numpy as np
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap

    top = mpl.cm.get_cmap('Oranges_r')
    bottom = mpl.cm.get_cmap('Blues')
    newcolors = np.vstack((top(np.linspace(.25, 1., 128)),
                           bottom(np.linspace(0., .75, 128))))
    bg_map = ListedColormap(newcolors, name='OrangeBlue')
    return bg_map


def plot_boundary(fitted_estimator, X, y, mesh_step_size=0.1, cmap=None):
    from matplotlib import pyplot as plt
    import numpy as np

    """Plot estimator decision boundary and scatter points

    Parameters
    ----------
    fname : str
        File name where the figures is saved.

    fitted_estimator : a fitted estimator

    X : array, shape (n_samples, 2)
        Input matrix

    y : array, shape (n_samples, )
        Binary classification target

    mesh_step_size : float, optional (default=0.2)
        Mesh size of the decision boundary

    title : str, optional (default="")
        Title of the graph

    """
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))

    if hasattr(fitted_estimator, "decision_function"):
        Z = fitted_estimator.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = fitted_estimator.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    # Put the result into a color plot
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=.8)


def plt_decorate(title=None, xlabel="Input", ylabel="Output"):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return plt.grid(linestyle="--", linewidth=.5)

