import pandas as pd
import numpy as np
from typing import Callable

from giskard.ml_worker.core.giskard_dataset import GiskardDataset
from giskard.ml_worker.core.model import GiskardModel
from giskard.ml_worker.generated.ml_worker_pb2 import SingleTestResult
from giskard.ml_worker.testing.abstract_test_collection import AbstractTestCollection


class StatisticalTests(AbstractTestCollection):

    @staticmethod
    def _predict_result(ds: GiskardDataset, model: GiskardModel, positive_idx: int):
        df = model.prepare_dataframe(ds)
        raw_prediction = model.prediction_function(df)
        if model.model_type == "regression":
            raise ValueError("Disparate Impact only works with classification models")
        elif model.model_type == "classification":
            labels = np.array(model.classification_labels)
            threshold = model.classification_threshold

            if threshold is not None and len(labels) == 2:
                return np.squeeze((raw_prediction[:, 1] > threshold).astype(int) == positive_idx)
            else:
                return np.squeeze(raw_prediction.argmax(axis=1) == positive_idx)

    def test_disparate_impact(self,
                              gsk_dataset: GiskardDataset,
                              protected_slice: Callable[[pd.DataFrame], pd.DataFrame],
                              unprotected_slice: Callable[[pd.DataFrame], pd.DataFrame],
                              model: GiskardModel,
                              positive_outcome,
                              threshold=0.8,
                              classification_label=None,
                              ) -> SingleTestResult:

        testing = gsk_dataset.df[gsk_dataset.target]

        if positive_outcome not in gsk_dataset.df[gsk_dataset.target].values:
            raise ValueError(
                f"The positive outcome chosen {positive_outcome} is not part of the dataset columns {gsk_dataset.columns}."
            )

        gsk_dataset.df.reset_index(drop=True, inplace=True)
        protected_ds=gsk_dataset.slice(protected_slice)
        unprotected_ds=gsk_dataset.slice(unprotected_slice)

        if protected_ds.df.equals(unprotected_ds.df):
            raise ValueError(
                f"The protected and unprotected datasets are equal. Please check that you chose different slices."
            )
        protected_ds_po = protected_ds.slice(lambda df: df[df[gsk_dataset.target] == positive_outcome])
        unprotected_ds_po = unprotected_ds.slice(lambda df: df[df[gsk_dataset.target] == positive_outcome])

        positive_idx = np.where(model.classification_labels == positive_outcome)

        protected_predictions = StatisticalTests._predict_result(protected_ds_po, model, positive_idx)
        unprotected_predictions = StatisticalTests._predict_result(unprotected_ds_po, model, positive_idx)

        protected_proba = np.count_nonzero(protected_predictions)/protected_predictions.shape[0]
        unprotected_proba = np.count_nonzero(unprotected_predictions)/unprotected_predictions.shape[0]

        DI = protected_proba/unprotected_proba

        return self.save_results(
            SingleTestResult(
                actual_slices_size=[len(protected_ds_po)],
                reference_slices_size=[len(unprotected_ds_po)],
                metric=DI,
                passed=DI > threshold,
            )
        )