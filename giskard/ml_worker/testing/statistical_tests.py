import pandas as pd

from giskard.ml_worker.core.giskard_dataset import GiskardDataset
from giskard.ml_worker.core.model import GiskardModel
from giskard.ml_worker.generated.ml_worker_pb2 import SingleTestResult
from giskard.ml_worker.testing.abstract_test_collection import AbstractTestCollection
import numpy as np

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
                              protected_ds: GiskardDataset,
                              unprotected_ds: GiskardDataset,
                              model: GiskardModel,
                              positive_outcome,
                              threshold=0.8,
                              classification_label=None,
                              ) -> SingleTestResult:

        #if positive_outcome not in protected_df.columns:
        #    raise ValueError(
        #        f"The positive outcome chosen {positive_outcome} is not part of the dataframe columns {protected_df.columns}."
        #    )

        positive_idx = np.where(model.classification_labels==positive_outcome)

        protected_ds.df.reset_index(drop=True, inplace=True)
        protected_ds.df = protected_ds.df[protected_ds.df[protected_ds.target] == positive_outcome]
        protected_predictions = StatisticalTests._predict_result(protected_ds,model,positive_idx)

        unprotected_ds.df.reset_index(drop=True, inplace=True)
        unprotected_ds.df = unprotected_ds.df[unprotected_ds.df[unprotected_ds.target] == positive_outcome]
        unprotected_predictions = StatisticalTests._predict_result(unprotected_ds,model,positive_idx)


        protected_proba = np.count_nonzero(protected_predictions)/protected_predictions.shape[0]
        unprotected_proba = np.count_nonzero(unprotected_predictions)/unprotected_predictions.shape[0]

        DI = protected_proba/unprotected_proba

        return self.save_results(
            SingleTestResult(
                actual_slices_size=[len(protected_ds)],
                reference_slices_size=[len(unprotected_ds)],
                metric=DI,
                passed=DI > threshold,
            )
        )