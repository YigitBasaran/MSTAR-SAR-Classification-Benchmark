"""Traditional classification pipeline: StandardScaler -> RBF-SVM."""

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


def build_pipeline():
    """RBF-SVM pipeline matching the case-study specification.

    ``decision_function_shape='ovr'`` yields one score per class, which is used
    later for the one-vs-rest ROC curves.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                    decision_function_shape="ovr", random_state=42)),
    ])
