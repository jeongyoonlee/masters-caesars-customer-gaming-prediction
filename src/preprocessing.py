from __future__ import division
from scipy import sparse
from scipy.signal import butter, lfilter
from scipy.stats import norm
from sklearn import base
from statsmodels.distributions.empirical_distribution import ECDF
import logging
import numpy as np
import pandas as pd
import logging

NAN_INT = 7535805

class LabelEncoder(base.BaseEstimator):
    """Label Encoder that groups infrequent values into one label.
    Attributes:
        min_obs (int): minimum number of observation to assign a label.
        label_encoders (list of dict): label encoders for columns
        label_maxes (list of int): maximum of labels for columns
    """

    def __init__(self, min_obs=10):
        """Initialize the OneHotEncoder class object.
        Args:
            min_obs (int): minimum number of observation to assign a label.
        """

        self.min_obs = min_obs

    def __repr__(self):
        return ('LabelEncoder(min_obs={})').format(self.min_obs)

    def _get_label_encoder_and_max(self, x):
        """Return a mapping from values and its maximum of a column to integer labels.
        Args:
            x (numpy.array): a categorical column to encode.
        Returns:
            label_encoder (dict): mapping from values of features to integers
            max_label (int): maximum label
        """

        # NaN cannot be used as a key for dict. So replace it with a random integer.
        x[pd.isnull(x)] = NAN_INT

        # count each unique value
        label_count = {}
        for label in x:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

        # add unique values appearing more than min_obs to the encoder.
        label_encoder = {}
        label_index = 1
        labels_not_encoded = 0
        for label in label_count.keys():
            if label_count[label] >= self.min_obs:
                label_encoder[label] = label_index
                label_index += 1
            else:
                labels_not_encoded += 1

        max_label = label_index - 1

        # if every label is encoded, then replace the maximum label with 0 so
        # that total number of labels encoded is (# of total labels - 1).
        if labels_not_encoded == 0:
            for label in label_encoder:
                # find the label with the maximum encoded value
                if label_encoder[label] == max_label:
                    # set the value of the label to 0 and decrease the maximum
                    # by 1.
                    label_encoder[label] = 0
                    max_label -= 1
                    break

        return label_encoder, max_label

    def _transform_col(self, x, col):
        """Encode one categorical column into labels.
        Args:
            x (numpy.array): a categorical column to encode
            col (int): column index
        Returns:
            x (numpy.array): a column with labels.
        """

        label_encoder = self.label_encoders[col]

        # replace NaNs with the pre-defined random integer
        x[pd.isnull(x)] = NAN_INT

        labels = np.zeros((x.shape[0], ), dtype=np.int64)
        for label in label_encoder:
            labels[x == label] = label_encoder[label]

        return labels

    def fit(self, X, y=None):
        self.label_encoders = [None] * X.shape[1]
        self.label_maxes = [None] * X.shape[1]

        for col in range(X.shape[1]):
            logging.info('processing {} ...'.format(col))
            self.label_encoders[col], self.label_maxes[col] = \
                self._get_label_encoder_and_max(X[:, col])

        return self

    def transform(self, X):
        """Encode categorical columns into sparse matrix with one-hot-encoding.
        Args:
            X (numpy.array): categorical columns to encode
        Returns:
            X (numpy.array): label encoded columns
        """

        for col in range(X.shape[1]):
            X[:, col] = self._transform_col(X[:, col], col)

        return X

    def fit_transform(self, X, y=None):
        """Encode categorical columns into label encoded columns
        Args:
            X (numpy.array): categorical columns to encode
        Returns:
            X (numpy.array): label encoded columns
        """

        self.label_encoders = [None] * X.shape[1]
        self.label_maxes = [None] * X.shape[1]

        for col in range(X.shape[1]):
            self.label_encoders[col], self.label_maxes[col] = \
                self._get_label_encoder_and_max(X[:, col])

            X[:, col] = self._transform_col(X[:, col], col)

        return X