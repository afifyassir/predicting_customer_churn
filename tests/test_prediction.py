"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from model.predict import make_prediction

# Add the root of your project to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def test_make_prediction(sample_input_data):
    # Given
    expected_no_predictions = len(sample_input_data[0])

    # When
    result = await make_prediction(input_data=sample_input_data[0])

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.int64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    _predictions = list(predictions)
    y_true = sample_input_data[1]
    accuracy = accuracy_score(_predictions, y_true)
    assert accuracy > 0.7
