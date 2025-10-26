import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

n_clusters = 6  # 添加这行，定义聚类数量

n_samples = 1500
np.random.seed(0)
t = 1.5 * np.pi * (1 + 3 * np.random.rand(1, n_samples))
x = t * np.cos(t)
y = t * np.sin(t)

X = np.concatenate((x, y))
X += 0.7 * np.random.randn(2, n_samples)
X = X.T

knn_graph = kneighbors_graph(X, 30, include_self=False)

for connectivity in (None, knn_graph):
    plt.figure(figsize=(10, 10))
    for index, linkage in enumerate(("average", "complete", "ward", "single")):
        plt.subplot(1, 4, index + 1)
        model = AgglomerativeClustering(
            linkage=linkage, connectivity=connectivity, n_clusters=n_clusters  # 修正：使用 n_clusters
        )
        t0 = time.time()
        model.fit(X)
        elapsed_time = time.time() - t0
        plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap=plt.cm.nipy_spectral)
        plt.title(
            "linkage=%s\n(time %.2fs)" % (linkage, elapsed_time),
            fontdict=dict(verticalalignment="top"),
        )
        plt.axis("equal")
        plt.axis("off")

    # 修正：调整缩进，这些应该在外层循环内
        plt.subplots_adjust(bottom=0, top=0.83, wspace=0, left=0, right=1)
        plt.suptitle(
        "n_cluster=%i, connectivity=%r" % (n_clusters, connectivity is not None),
        size=17,
    )

plt.show()  # 添加显示图形的命令