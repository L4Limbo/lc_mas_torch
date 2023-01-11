import torch

# static list of metrics
metricList = ['r1', 'r5', 'r10', 'mean', 'mrr']
# +1 - greater the better
# -1 - lower the better
trends = [1, 1, 1, -1, 1]


def evaluateMetric(ranks, metric):
    ranks = ranks.data.numpy()
    if metric == 'r1':
        ranks = ranks.reshape(-1)
        return 100 * (ranks == 1).sum() / float(ranks.shape[0])
    if metric == 'r5':
        ranks = ranks.reshape(-1)
        return 100 * (ranks <= 5).sum() / float(ranks.shape[0])
    if metric == 'r10':
        ranks = ranks.reshape(-1)
        return 100 * (ranks <= 10).sum() / float(ranks.shape[0])
    if metric == 'mean':
        ranks = ranks.reshape(-1).astype(float)
        return ranks.mean()
    if metric == 'mrr':
        ranks = ranks.reshape(-1).astype(float)
        return (1 / ranks).mean()


def computeMetrics(ranks):
    results = {metric: evaluateMetric(ranks, metric) for metric in metricList}
    return 