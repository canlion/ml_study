import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=.02, title='', x_label='', y_label='', test_idx=None):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier(np.array([xx1.flatten(), xx2.flatten()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        cl_sample = X[y==cl]
        plt.scatter(x=cl_sample[..., 0], y=cl_sample[..., 1], alpha=.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')
        
    if isinstance(test_idx, (np.ndarray, list, tuple, range)):
        X_test, y_test = X[test_idx], y[test_idx]
        plt.scatter(x=X_test[..., 0], y=X_test[..., 1], alpha=1., c='', marker='o', label=y_test, edgecolor='black', s=100)
    
    if title:
        plt.title(title)
    
    if x_label and y_label:
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='upper left')
    plt.show()