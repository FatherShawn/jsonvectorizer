import copy
import numpy as np
from sklearn import feature_extraction, utils

from .basevectorizer import BaseVectorizer
from ..utils import _validation


class StringVectorizer(BaseVectorizer):
    """Vectorizer for strings

    Tokenization using scikit-learn's :class:`CountVectorizer`.

    Parameters
    ----------
    min_df : int or float, optional (default=1)
        When using tokenization, ignore terms that have a document
        frequency strictly lower than this threshold. An integer is
        taken as an absolute count, and a float indicates the proportion
        of `n_total` passed to the :meth:`fit` method.
    dtype : optional (default=np.float_)
        NumPy compatible data type for feature matrix.
    **kwargs
        Passed to scikit-learn's :class:`CountVectorizer` class for
        initialization.

    Raises
    ------
    ValueError
        If `min_df` is not a positive number.

    Attributes
    ----------
    feature_names_ : list of str

    """

    def __init__(self, min_df=1, dtype=np.float_, **kwargs):
        _validation.check_positive(min_df, alias='min_df')
        self.dtype = np.dtype(dtype)
        self.params = dict(min_df=min_df, **kwargs)

    def fit(self, values, n_total=None, **kwargs):
        """Fit vectorizer to the provided data

        Parameters
        ----------
        values : array-like, [n_samples]
        n_total : int or None, optional (default=None)
            Total Number of documents that values are extracted from. If
            None, defaults to ``len(values)``.
        **kwargs:
            Ignored keyword arguments.

        Returns
        -------
        self or None
            Returns None if no features were generated, otherwise
            returns `self`.

        """
        if n_total is None:
            n_total = len(values)

        params = copy.copy(self.params)
        if isinstance(params['min_df'], float):
            params['min_df'] = max(int(params['min_df'] * n_total), 1)
        else:
            params['min_df'] = params['min_df']

        vectorizer = feature_extraction.text.CountVectorizer(
            dtype=self.dtype, **params
        )
        try:
            vectorizer.fit(values)
        except ValueError:
            return None

        self._vectorizer = vectorizer
        self.feature_names_ = [
            u'has "{}"'.format(feature_name)
            for feature_name in vectorizer.get_feature_names()
        ]
        if hasattr(self._vectorizer, 'stop_words_'):
            delattr(self._vectorizer, 'stop_words_')

        return self

    def transform(self, values):
        """Transform values and return the resulting feature matrix

        Parameters
        ----------
        values : array-like, [n_samples]

        Returns
        -------
        X : sparse matrix, shape [n_samples, n_features]

        Raises
        ------
        NotFittedError
            If the vectorizer has not yet been fitted.

        """
        if not hasattr(self, 'feature_names_'):
            raise utils.NotFittedError('Vectorizer has not yet been fitted')

        return self._vectorizer.transform(values)
