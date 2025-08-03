import mlflow
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from mlflow.models import infer_signature
import warnings


def evaluate_and_log_metrics(model, X_test, y_test):
    """
    Evaluates a trained model and logs key metrics and artifacts to MLflow.
    This function is the heart of your model validation.

    Args:
        model: The trained scikit-learn model or pipeline.
        X_test: Test feature data for evaluation.
        y_test: Test target data for evaluation.
    """
    print("    - Evaluating model...")

    # Make predictions on the held-out test set
    predictions = model.predict(X_test)

    # 1. Calculate the Confusion Matrix to get raw counts
    # The .ravel() function flattens the 2x2 matrix into a simple array.
    # For binary classification, the order is: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    # 2. Log the raw counts as individual MLflow metrics
    # This is extremely useful for sorting/filtering runs in the UI
    mlflow.log_metric("true_positives", tp)
    mlflow.log_metric("false_positives", fp)
    mlflow.log_metric("true_negatives", tn)
    mlflow.log_metric("false_negatives", fn)
    print(f"      - TP: {tp}, FP: {fp}")

    # 3. Log derived metrics like accuracy
    accuracy = round(float(accuracy_score(y_test, predictions)), 4)
    mlflow.log_metric("accuracy", accuracy)
    print(f"      - Accuracy: {accuracy}")

    # 4. Log the full classification report as a text artifact
    # This gives you precision, recall, and f1-score for each class.
    # Saving it as a .json file makes it easy to read and parse later.
    report_dict = classification_report(y_test, predictions, output_dict=True)
    if not isinstance(report_dict, dict):
        return

    report_dict: dict[str, dict | str]
    for pred, cf_rep in report_dict.items():
        if not isinstance(cf_rep, dict):
            continue
        for k, v in cf_rep.items():
            mlflow.log_metric(key=f"{pred}-{k}", value=round(v, 4))


def log_model_artifact(model, model_name: str, input_example):
    """
    Logs the trained model itself as an artifact to MLflow.

    Args:
        model: The trained scikit-learn model or pipeline.
        artifact_path (str): The name for the artifact folder where the model is saved.
        input_example: A sample of the training data (e.g., X_train.head()).
    """
    # Infer the model's "signature" - its expected input and output schema.
    # This is a best practice that makes reloading and using the model safer.
    # TODO: fix this
    # with warnings.catch_warnings():
    #     warnings.filterwarnings(
    #         "ignore",
    #         message="Hint: Inferred schema contains integer column(s).",
    #     )
    #     signature = infer_signature(
    #         model_input=input_example,
    #         model_output=model.predict(input_example),
    #     )

    # Log the model using MLflow's scikit-learn integration.
    mlflow.sklearn.log_model(
        sk_model=model,
        name=model_name,
        # signature=signature,
        input_example=input_example,
    )
