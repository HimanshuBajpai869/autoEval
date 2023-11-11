# autoEval : Automatic Evaluation of Machine Learning Models.

[![Documentation Status](https://readthedocs.org/projects/autoeval/badge/?version=latest)](https://autoeval.readthedocs.io/en/latest/?badge=latest)

The library helps evaluate the model and provides below reports :

   1. Model Performance Metrics
   2. Model Performance Report
   3. Model Fairness Report
   4. Model Testing Report


Function Usage
---------------
Example usage of ``auto_evaluate_model`` function

    from autoEval import autoevaluatemodel

    autoevaluatemodel.auto_evaluate_model(
        model, 
        train_output, 
        test_output,
        feature_columns, 
        target_column, 
        predicted_column)
