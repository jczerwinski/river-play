import json

from river import datasets
from river import metrics
from river import stats
from river import meta
from river import optim
from river import reco

from river.evaluate import progressive_val_score

for x, y in datasets.MovieLens100K():
    print(f'x = {json.dumps(x, indent=4)}\ny = {y}')
    break

def evaluate(model):
    X_y = datasets.MovieLens100K()
    metric = metrics.MAE() + metrics.RMSE()
    _ = progressive_val_score(X_y, model, metric, print_every=25_000, show_time=True, show_memory=True)

mean = stats.Mean()
metric = metrics.MAE() + metrics.RMSE()

for i, x_y in enumerate(datasets.MovieLens100K(), start=1):
    _, y = x_y
    metric.update(y, mean.get())
    mean.update(y)

    if not i % 25_000:
        print(f'[{i:,d}] {metric}')

baseline_params = {
    'optimizer': optim.SGD(0.025),
    'l2': 0.,
    'initializer': optim.initializers.Zeros()
}

model = meta.PredClipper(
    regressor=reco.Baseline(**baseline_params),
    y_min=1,
    y_max=5
)

evaluate(model)

funk_mf_params = {
    'n_factors': 10,
    'optimizer': optim.SGD(0.05),
    'l2': 0.1,
    'initializer': optim.initializers.Normal(mu=0., sigma=0.1, seed=73)
}

model = meta.PredClipper(
    regressor=reco.FunkMF(**funk_mf_params),
    y_min=1,
    y_max=5
)

evaluate(model)

biased_mf_params = {
    'n_factors': 10,
    'bias_optimizer': optim.SGD(0.025),
    'latent_optimizer': optim.SGD(0.05),
    'weight_initializer': optim.initializers.Zeros(),
    'latent_initializer': optim.initializers.Normal(mu=0., sigma=0.1, seed=73),
    'l2_bias': 0.,
    'l2_latent': 0.
}

model = meta.PredClipper(
    regressor=reco.BiasedMF(**biased_mf_params),
    y_min=1,
    y_max=5
)

evaluate(model)