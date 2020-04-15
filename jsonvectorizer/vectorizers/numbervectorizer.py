import numpy as np
from sklearn import utils

from .basevectorizer import BaseVectorizer


class NumberVectorizer(BaseVectorizer):
    """Vectorizer for numbers

    Simply copies numbers.

    Parameters
    ----------
    dtype : optional (default=np.float_)
        NumPy compatible data type for feature matrix.

    Attributes
    ----------
    feature_names_ : list of str

    """
    def __init__(self, dtype=np.float_):
        self.dtype = np.dtype(dtype)

    def fit(self, values, **kwargs):
        """Fit vectorizer to the provided data

        Parameters
        ----------
        values : array-like, [n_samples]
        **kwargs
            Ignored keyword arguments.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If `values` is not a one-dimensional array.

        """
        values = np.asarray(values, dtype=self.dtype)
        if values.ndim != 1:
            raise ValueError(
                'values must be a one dimensional array, not with shape {}'
                .format(values.shape)
            )

        self.feature_names_ = ['']
        return self

    def transform(self, values):
        """Transform numbers to feature matrix

        Parameters
        ----------
        values : array-like, [n_samples]

        Returns
        -------
        X : ndarray, [n_samples, 1]

        Raises
        ------
        NotFittedError
            If the vectorizer has not yet been fitted.

        """
        if not hasattr(self, 'feature_names_'):
            raise utils.NotFittedError('Vectorizer has be yet been fitted')

        values = np.asarray(values, dtype=self.dtype).reshape(-1, 1)
        return values
