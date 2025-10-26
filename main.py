from sklearn import metrics

score_funcs = [
    ("V-measure", metrics.v_measure_score),
    ("Rand index", metrics.rand_score),
    ("ARI", metrics.adjusted_rand_score),
    ("MI", metrics.mutual_info_score),
    ("NMI", metrics.normalized_mutual_info_score),
    ("AMI", metrics.adjusted_mutual_info_score),
]

#First experiment:fixed ground truth labels and growing number of clusters

import numpy as np
rng = np.random.RandomState(0)

def random_labels(n_samples, n_classes):
    return rng.randint(low=0, high=n_classes, size=n_samples)

def fixed_classes_uniform_labelings_scores(

        score_func, n_samples,n_clusters_range, n_classes,n_runs=5
):
    scores = np.zeros((len(n_clusters_range),n_runs))
    labels_a = random_labels(n_samples=n_samples, n_classes=n_classes)

    for i,n_clusters in enumerate(n_clusters_range):
        for j in range(n_runs):
            labels_b = random_labels(n_samples=n_samples, n_classes=n_clusters)
            scores[i,j] = score_func(labels_a, labels_b)
    return scores

import matplotlib.pyplot as plt
import seaborn as sns

n_samples = 1000
n_classes = 10
n_clusters_range = np.linspace(2,100,10).astype(int)
plots = []
names = []

sns.color_palette("colorblind")
plt.figure(1)

for marker,(score_name, score_func) in zip("d^vx.,", score_funcs):
    scores = fixed_classes_uniform_labelings_scores(
        score_func, n_samples, n_clusters_range, n_classes=n_classes
    )
    plots.append(
        plt.errorbar(
            n_clusters_range,
            scores.mean(axis=1),
            scores.std(axis=1),
            alpha=0.8,
            linewidth=1,
            marker=marker,
        )[0]
    )
    names.append(score_name)
plt.title(
    "Clustering measures for random uniform labeling\n"
    f"against reference assignment with {n_classes} classes"
)
plt.xlabel(f"Number of clusters (Number of samples is fixed to {n_samples})")
plt.ylabel("Score value")
plt.ylim(bottom=-0.05, top=1.05)
plt.legend(plots, names, bbox_to_anchor=(0.5,0.5))
plt.show()

def uniform_labelings_scores(score_func, n_samples, n_clusters_range, n_runs=5):
    scores = np.zeros((len(n_clusters_range), n_runs))

    for i,n_clusters in enumerate(n_clusters_range):
        for j in range(n_runs):
            labels_a = random_labels(n_samples = n_samples, n_classes=n_clusters)
            labels_b = random_labels(n_samples = n_samples, n_classes = n_clusters)
            scores[i, j] = score_func(labels_a, labels_b)
    return scores

n_samples = 100
n_clusters_range = np.linspace(2, n_samples, 10).astype(int)

plt.figure(2)

plots = []
names = []

for marker, (score_name, score_func) in zip("d^vx.,", score_funcs):
    scores = uniform_labelings_scores(score_func, n_samples, n_clusters_range)
    plots.append(
        plt.errorbar(
            n_clusters_range,
            np.median(scores, axis=1),
            scores.std(axis=1),
            alpha=0.8,
            linewidth=2,
            marker=marker,
        )[0]
    )
    names.append(score_name)

    plt.title(
        "Clustering measures for 2 random uniform labelings\nwith equal number of clusters"
    )
    plt.xlabel(f"Number of clusters (Number of samples is fixed to {n_samples})")
    plt.ylabel("Score value")
    plt.legend(plots, names)
    plt.ylim(bottom=-0.05, top=1.05)
    plt.show()