# Machine Learning App Deployment

This repository contains the code for deploying a machine learning model that predicts whether a customer will churn or not. The model is trained on a dataset provided by a European company specialized in energy and uses a Random Forest classifier algorithm.

## Tools and Libraries

The app is built using several tools and libraries, including:

- FastAPI: for building the web API
- Pydantic: for data validation and settings management
- Uvicorn: for running the app
- Loguru: for logging
- Typing: for type annotations
- Docker: for containerization
- Tox: for testing and automation
- Pytest: for testing
- flake8: for code linting
- black: for code formatting

## Usage

To use the app, send a POST request to the `/predict` endpoint with a JSON payload containing the customer data. The app will return a prediction of whether the customer will churn or not.

## Testing

To run the tests, install Tox and run `tox` in the root directory of the repository. This will run the tests using Pytest and check that everything is working correctly.