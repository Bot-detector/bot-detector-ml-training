import logging
import os
import time

import numpy as np

import cudf
import cuml

import joblib

logger = logging.getLogger(__name__)


class classifier:
    """
    This class is a wrapper for RandomForestClassifier.
    It adds the ability to save and load the model.
    """

    working_directory = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(working_directory, "models")
    if not os.path.exists(path):
        os.makedirs(path)
    loaded = False

    def __init__(self, name, path=None, **kwargs):
        """
        Initialize the classifier object.
        :param path: path to the models
        :param name: name of the classifier
        :param kwargs: keyword arguments for RandomForestClassifier
        """
        self.name = name
        self.path = path if path is not None else self.path

    def __best_file_path(self, startwith: str):
        """
        This method will return the best model path based on accuracy.
        :param startwith: name of the classifier
        :return: classifier object
        """
        files = []
        for f in os.listdir(self.path):
            if f.endswith(".joblib") and f.startswith(startwith):
                # accuracy is in filename, so we have to parse it
                model_file = f.replace(".joblib", "")
                model_file = model_file.split(sep="_")
                # save to dict
                logger.debug(f"joined path: {os.path.join(self.path, f)}")
                d = {
                    "path": os.path.join(self.path, f),
                    "model": model_file[0],
                    "date": model_file[1],
                    "accuracy": model_file[2],
                }
                # add dict to array
                files.append(d)

        if not files:
            return None

        # array of dict can be used for pandas dataframe
        df_files = cudf.DataFrame(files)
        df_files.sort_values(by=["date"], ascending=False, inplace=True)
        path = df_files["path"].iloc[0]
        return joblib.load(path)

    def load(self):
        """
        Loads the model object from the file.
        :return: classifier object
        """

        model = self.__best_file_path(self.name)
        if model:
            self = model
            logger.debug(f"Loading: {self.name}, {self.path}")
        else:
            logger.debug("No model defined")
            return

        self.loaded = True
        return self

    def save(self):
        """
        Save the classifier object to the file.
        :return: None
        """
        logger.debug(f"Saving classifier: {self.name}")
        today = int(time.time())
        joblib.dump(
            self,
            f"{self.path}/{self.name}_{today}_{round(self.accuracy, 2)}.joblib",
            compress=3,
        )

    def score(self, test_y, test_x):
        """
        Calculate the accuracy and roc_auc score for the classifier.
        :param test_y: test data labels
        :param test_x: test data features
        :return: accuracy and roc_auc score
        """
        labels = test_y.unique().to_arrow().to_pylist()
        print(f"labels: {labels}")

        # make predictions
        pred_y = self.rfc.predict(test_x)
        pred_proba_y = self.rfc.predict_proba(test_x)

        if len(labels) == 2:
            pred_proba_y = pred_y  # pred_proba_y[:, 1]
            self.roc_auc = cuml.metrics.roc_auc_score(test_y, pred_proba_y)
        else:
            self.roc_auc = None

        self.accuracy = cuml.metrics.accuracy.accuracy_score(test_y, pred_y)

        # labels = ["Not bot", "bot"] if len(labels) == 2 else labels

        logger.info(cuml.metrics.confusion_matrix(test_y, pred_y, convert_dtype=True))
        return self.accuracy, self.roc_auc

    def fit(self, train_x, train_y, *args, **kwargs):
        kwargs = kwargs if len(kwargs) != 0 else {}
        args = args if len(args) != 0 else []
        # self.rfc = cuml.ensemble.RandomForestClassifier(*args,
        #     split_criterion=0, handle=0, verbose=False, output_type=None,
        #     **kwargs)
        self.rfc = cuml.ensemble.RandomForestClassifier(
            split_criterion=0, handle=None, n_estimators=100, verbose=True
        )
        start_time = time.time()
        self.rfc.fit(train_x, train_y)
        print("fit time:", time.time() - start_time)

    def predict(self, x):
        return self.rfc.predict(x)
