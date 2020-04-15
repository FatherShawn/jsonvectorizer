import datetime
import dateutil.parser
import numpy as np
import pytz
import scipy.sparse as sp

from .basevectorizer import BaseVectorizer
from .binvectorizer import BinVectorizer
from ..utils import _validation


EPOCH = datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)


def parse_timestamp(timestamp):
    # Parse a timestamp and return a unix timestamp
    timestamp = dateutil.parser.parse(timestamp)
    if timestamp.tzinfo is None:
        timestamp = pytz.utc.localize(timestamp)

    return (timestamp - EPOCH).total_seconds()


class TimestampVectorizer(BaseVectorizer):
    """Vectorizer for timestamps

    Bins data into the specfied number of equiprobable bins, or using
    the provided bin edges, and uses one-hot encoding to create a binary
    feature matrix. After binning, the resulting bins are processed from
    left to right, and are merged into their right neighbor until all
    bins contain at least the specified number of items. If necessary,

    Parses and converts strings to unix timestamps, bins results into
    the specified number of equiprobable bins, or using the provided bin
    edges. and uses one-hot encoding to create a binary feature matrix.
    After binning, the resulting bins are processed from left to right,
    and are merged into their right neighbor until all bins contain at
    least the specified number of items. If necessary, the right-most
    bin is then merged into its left neighbor. Also, if at least `min_f`
    items are not valid timestamps, an additional bin (feature) is
    created for invalid timestamps.

    Parameters
    ----------
    bins : int or list
        Number of bins to generate, or a list of timestamps to use as
        bin edges (excluding -inf and inf).
    min_f : int or float, optional (default=1)
        Minimum number of samples in each generated bin. An integer is
        taken as an absolute count, and a float indicates the proportion
        of `n_total` passed to the :meth:`fit` method.
    dtype : optional (default=np.float_)
        NumPy compatible data type for feature matrix.

    Raises
    ------
    ValueError
        If `min_f` is not a positive number.

    Attributes
    ----------
    feature_names_ : list of str

    """

    def __init__(self, bins, min_f=1, dtype=np.float_):
        _validation.check_positive(min_f, alias='min_f')
        if not isinstance(bins, int):
            bins = [parse_timestamp(bin_edge) for bin_edge in bins]

        self.bins = bins
        self.min_f = min_f
        self.type = np.dtype(dtype)

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
        if n_total is None:
            n_total = len(values)

        if isinstance(self.min_f, float):
            min_f = max(int(self.min_f * n_total), 1)
        else:
            min_f = self.min_f

        timestamps = []
        n_invalid = 0
        for value in values:
            try:
                timestamps.append(parse_timestamp(value))
            except (ValueError, OverflowError):
                n_invalid += 1

        has_invalid_feature = (n_invalid >= min_f)
        vectorizer = BinVectorizer(self.bins, min_f=min_f, dtype=self.dtype)
        vectorizer = vectorizer.fit(timestamps, n_total=n_total)
        if vectorizer is None and not has_invalid_feature:
            return None

        feature_names = []
        if has_invalid_feature:
            feature_names.append('is not a valid timestamp')
        if vectorizer is not None:
            bin_edges = [
                datetime.datetime.fromtimestamp(bin_edge).isoformat() + 'Z'
                for bin_edge in vectorizer._bin_edges
            ]
            feature_names.append('in (-inf, {})'.format(bin_edges[0]))
            feature_names.extend([
                'in [{}, {})'.format(bin_edges[i], bin_edges[i+1])
                for i in range(len(bin_edges) - 1)
            ])
            feature_names.append('in [{}, inf)'.format(bin_edges[-1]))

        self._has_invalid_feature = has_invalid_feature
        self._vectorizer = vectorizer
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
            raise NotFittedError('Vectorizer has not yet been fitted')

        invalids = []
        valids = []
        timestamps = []
        for i, value in enumerate(values):
            try:
                timestamps.append(parse_timestamp(value))
                valids.append(i)
            except (ValueError, OverflowError):
                invalids.append(i)

        n_values = len(invalids) + len(valids)
        X = sp.lil_matrix(
            (n_values, len(self.feature_names_)), dtype=self.dtype
        )
        if self._has_invalid_feature:
            if invalids:
                X[invalids,0] = 1
            if valids and self._vectorizer is not None:
                X[valids,1:] = self._vectorizer.transform(timestamps)
        elif valids:
            X[valids,:] = self._vectorizer.transform(timestamps)

        return X
