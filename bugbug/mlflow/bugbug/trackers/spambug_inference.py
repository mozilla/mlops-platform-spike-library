import mlflow
import pandas
from pandas import DataFrame


class SpambugInference(mlflow.pyfunc.PythonModel):
    """
    Inference code copied from MLFlow bugs.py
    """
    def __init__(self, extraction_pipeline, clf, le):
        self.extraction_pipeline = extraction_pipeline
        self.clf = clf
        self.le = le
    def predict(self, context, bugs):
        """
        Args:
            context ([type]): MLflow context where the model artifact is stored.
            model_input ([type]): the input data to fit into the model.
        """
        if isinstance(bugs, DataFrame):
            bugs = bugs.to_dict("records")
        probs = self.classify(bugs, True)
        indexes = probs.argmax(axis=-1)
        suggestions = self.le.inverse_transform(indexes)
        return {"probs": probs, "indexes": indexes, "suggestions": suggestions}
    def classify(
        self,
        items,
        probabilities=False
    ):
        assert items is not None
        assert (
            self.extraction_pipeline is not None and self.clf is not None
        ), "The module needs to be initialized first"

        if not isinstance(items, list):
            items = [items]
        assert isinstance(items[0], (dict, tuple))
        X = self.extraction_pipeline.transform(lambda: items)
        if probabilities:
            classes = self.clf.predict_proba(X)
        else:
            classes = self.clf.predict(X)
        classes = self.overwrite_classes(items, classes, probabilities)
        return classes
    def overwrite_classes(self, bugs, classes, probabilities):
        for i, bug in enumerate(bugs):
            if "@mozilla" in bug["creator"]:
                if probabilities:
                    classes[i] = [1.0, 0.0]
                else:
                    classes[i] = 0

        return classes
