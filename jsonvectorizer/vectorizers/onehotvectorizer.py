import numpy as np
from sklearn import preprocessing, utils

from .basevectorizer import BaseVectorizer
from ..utils import _validation


class OneHotVectorizer(BaseVectorizer):
    """Vectorizer for categorical values.

    One-hot encoding using scikit-learn's :class:`OneHotEncoder`.

    Parameters
    ----------
    min_f : int or float, optional (default=1)
        Ignores categories sparser than this threshold. An integer is
        taken as an absolute count, and a float indicates the proportion
        of `n_total` passed to the :meth:`fit` method.
    min_categories : int, optional (default=1)
        Does not generate any features if the number of extracted
        categories is lower than this threshold.
    lowercase : bool, optional(default=True)
        Whether to convert strings to lowercase.

    Raises
    ------
    ValueError
        If `min_f` is not a positive number, or if `min_categories` is
        not a positive integer.

    Attributes
    ----------
    feature_names_ : list of str

    """

    def __init__(self, min_f=1, min_categories=1, lowercase=True):
        _validation.check_positive(min_f, alias='min_f')
        _validation.check_positive_int(min_categories, alias='min_categories')

        self.min_f = min_f
        self.min_categories = min_categories
        self.lowercase = bool(lowercase)

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

        if isinstance(self.min_f, float):
            min_f = max(int(self.min_f * n_total), 1)
        else:
            min_f = self.min_f

        lowercase = False
        if self.lowercase and isinstance(values[0], str):
            values = [value.lower() for value in values]
            lowercase = True

        values = np.asarray(values).reshape(-1, 1)
        categories, counts = np.unique(values, return_counts=True)
        categories = categories[counts >= min_f]
        if categories.shape[0] < self.min_categories:
            return None

        vectorizer = preprocessing.OneHotEncoder(
            categories=[categories], dtype=np.bool, handle_unknown='ignore'
        )
        vectorizer.fit(values)

        self._vectorizer = vectorizer
        self._lowercase = lowercase
        self.feature_names_ = [
            u'= {}'.format(category) for category in vectorizer.categories_[0]
        ]

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

        if self._lowercase:
            values = [value.lower() for value in values]

        values = np.asarray(values).reshape(-1, 1)
        return self._vectorizer.transform(values)
