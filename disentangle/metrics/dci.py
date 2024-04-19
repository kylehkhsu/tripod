import ipdb
import numpy as np
import jax
import jax.numpy as jnp
import scipy
from sklearn import ensemble, preprocessing, model_selection


def process_latents(latents):
    if scipy.sparse.issparse(latents):
        pass
    elif latents.dtype in [np.int64, np.int32, jnp.int64, jnp.int32]:
        one_hot_encoder = preprocessing.OneHotEncoder(sparse=False)
        latents = one_hot_encoder.fit_transform(latents)
    elif latents.dtype in [np.float32, np.float64, jnp.float32, jnp.float64]:
        standardizer = preprocessing.StandardScaler()
        latents = standardizer.fit_transform(latents)
    else:
        raise ValueError(f'latents.dtype {latents.dtype} not supported')
    return latents


def compute_relative_importance_matrix(latents, sources, max_depth=10):
    ret = np.zeros((latents.shape[1], sources.shape[1]))
    label_encoder = preprocessing.LabelEncoder()
    latents_processed = process_latents(latents)
    test_accs = []
    for i in range(sources.shape[1]):
        model = ensemble.RandomForestClassifier(
            n_estimators=100,
            criterion='entropy',
            max_depth=max_depth,
            min_samples_split=2,
            n_jobs=-1,
            verbose=0
        )
        source = label_encoder.fit_transform(sources[:, i])

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            latents_processed, source, test_size=0.1, random_state=42
            )

        model.fit(X_train, y_train)
        ret[:, i] = model.feature_importances_

        test_accs.append(np.mean(model.predict(X_test) == y_test))
    return ret, np.mean(test_accs)


def disentanglement_per_code(relative_importance_matrix):
    """Compute disentanglement score of each code."""
    # relative_importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(
        relative_importance_matrix.T + 1e-11,
        base=relative_importance_matrix.shape[1]
        )


def disentanglement(relative_importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(relative_importance_matrix)
    if relative_importance_matrix.sum() == 0.:
        relative_importance_matrix = np.ones_like(relative_importance_matrix)
    code_importance = relative_importance_matrix.sum(axis=1) / relative_importance_matrix.sum()

    return np.sum(per_code * code_importance)


def completeness_per_factor(relative_importance_matrix):
    """Compute completeness of each factor."""
    # relative_importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(
        relative_importance_matrix + 1e-11,
        base=relative_importance_matrix.shape[0]
        )


def completeness(relative_importance_matrix):
    """"Compute completeness of the representation."""
    per_factor = completeness_per_factor(relative_importance_matrix)
    if relative_importance_matrix.sum() == 0.:
        relative_importance_matrix = np.ones_like(relative_importance_matrix)
    factor_importance = relative_importance_matrix.sum(axis=0) / relative_importance_matrix.sum()
    return np.sum(per_factor * factor_importance)


def compute_dci(sources, latents):
    rim, I = compute_relative_importance_matrix(latents, sources)

    return {
        'D': disentanglement(rim),
        'C':    completeness(rim),
        'I': I
    }
