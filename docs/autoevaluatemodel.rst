Automatic Model Evaluation
==========================

.. currentmodule:: autoEval.autoevaluatemodel
.. autofunction:: auto_evaluate_model

Function Usage
---------------
Example usage of ``auto_evaluate_model`` function

.. code-block:: python

    from autoEval import autoevaluatemodel

    autoevaluatemodel.auto_evaluate_model(
        model, 
        train_output, 
        test_output,
        feature_columns, 
        target_column, 
        predicted_column)
    