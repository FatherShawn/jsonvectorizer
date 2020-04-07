import numpy as np
import scipy.sparse as sp
from sklearn import utils

from .basevectorizer import BaseVectorizer
from ..utils import _validation


class NumberVectorizer(BaseVectorizer):
    """Vectorizer for numbers

    Bins data into the specfied number of equiprobable bins, or using
    the provded bin edges, and uses one-hot encoding to create a binary
    feature matrix. After binning, the resulting bins are processed from
    left to right, and are merged into their right neighbor until all
    bins contain at least the specified number of items. If necessary,
    the right-most bin is then merged into its left neighbor.

    Parameters
    ----------
    bins : int or list
        Number of bins to generate, or a list of bin edges excluding
        -inf and inf.
    min_f : int or float, optional (default=1)
        Minimum number of samples in each generated bin. An integer is
        taken as an absolute count, and a float indicates the proportion
        of `n_total` passed to the :meth:`fit` method.

    Raises
    ------
    ValueError
        If `min_f` is not a positive number.

    Attributes
    ----------
    feature_names_ : list of str

    """

    def __init__(self, bins, min_f=1):
        _validation.check_positive(min_f, alias='min_f')

        self.bins = bins
        self.min_f = min_f

    def fit(self, values, n_total=None, **kwargs):
        """Fit vectorizer to the provided data

        Parameters
        ----------
        values : array-like, [n_samples]
        n_total : int or None, optional (default=None)
            Total Number of documents that values are extracted from. If
            None, defaults to ``len(values)``.
        **kwargs
            Ignored keyword arguments.

        Returns
        -------
        self or None
            Returns `self` if at least two bins are generated, otherwise
            returns None.

        """
        values = np.asarray(values)
        if n_total is None:
            n_total = values.shape[0]

        if isinstance(self.min_f, float):
            min_f = max(int(self.min_f * n_total), 1)
        else:
            min_f = self.min_f

        if isinstance(self.bins, int):
            bin_edges = np.percentile(
                values, np.arange(100.0 / self.bins, 100.0, 100.0 / self.bins),
                interpolation='higher'
            ).tolist()
        else:
            bin_edges = self.bins

        hist, _ = np.histogram(
            values, bins=np.concatenate(([-np.inf], bin_edges, [np.inf]))
        )
        hist = hist.tolist()

        # Prune bins that hold less than min_f values
        i = 0
        while i != len(bin_edges):
            if hist[i] < min_f:
                hist[i + 1] += hist[i]
                del bin_edges[i]
                del hist[i]
            else:
                i += 1
        while bin_edges and hist[-1] < min_f:
            hist[-2] += hist[-1]
            del bin_edges[-1]
            del hist[-1]
        if not bin_edges:
            return None

        # (TODO) better feature names (e.g., for integers)
        feature_names = ['in (-inf,{:.3e})'.format(bin_edges[0])]
        feature_names.extend([
            'in [{:.3e},{:.3e})'.format(bin_edges[i], bin_edges[i+1])
            for i in range(len(bin_edges) - 1)
        ])
        feature_names.append('in [{:.3e},inf)'.format(bin_edges[-1]))

        self._bin_edges = bin_edges
        self.feature_names_ = feature_names
        return self

    def transform(self, values):
        """Transform values and return the resulting feature matrix

        Parameters
        ----------
        values : array-like, [n_samples]

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]

        Raises
        ------
        NotFittedError
            If the vectorizer has not yet been fitted.

        """
        if not hasattr(self, 'feature_names_'):
            raise utils.NotFittedError('Vectorizer has not yet been fitted')

        values = np.asarray(values)
        indices = np.digitize(values, self._bin_edges)
        ones = np.ones(indices.shape[0], dtype=bool)
        return sp.coo_matrix(
            (ones, (np.arange(indices.shape[0]), indices)),
            shape=(indices.shape[0], len(self.feature_names_)), dtype=bool
        )
