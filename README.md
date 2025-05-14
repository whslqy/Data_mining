# Fake News Passive-Aggressive Classifier

## Overview
This Jupyter notebook, `fakenews-passive-aggressive.ipynb`, is designed to classify news articles as fake or real using a Passive-Aggressive classifier, a popular online learning algorithm suitable for large-scale text classification tasks. The project leverages a dataset sourced from Kaggle, which contains labeled news articles for training and evaluation. The notebook is structured to include data loading, preprocessing, feature engineering, model training, evaluation, and visualization steps. An attached image (`fake-news.png`) is included to provide visual insights, likely depicting model performance metrics, confusion matrices, or data distributions. The environment is configured for Python 3.10.12, running on a non-GPU setup with internet disabled, ensuring portability and offline usability.

## Features
- Implements a Passive-Aggressive classifier for binary text classification (fake vs. real news).
- Processes a Kaggle dataset with pre-labeled news articles.
- Includes data visualization using matplotlib (e.g., the `fake-news.png` attachment).
- Supports offline execution with all dependencies and data stored locally.
- Provides a modular structure for easy modification and experimentation with hyperparameters or additional features.

## Prerequisites
To run this notebook successfully, ensure the following are installed and configured:

- **Python Version**: 3.10.12 or higher.
- **Required Libraries**:
  - `pandas` for data manipulation and analysis.
  - `numpy` for numerical computations.
  - `scikit-learn` for the Passive-Aggressive classifier and preprocessing tools.
  - `matplotlib` for generating visualizations.
  - Optional: `seaborn` or `plotly` if additional plotting enhancements are desired (not mandatory).
- **Dataset**: Kaggle dataset must be downloaded and available locally. The dataset includes news text and corresponding labels.
- **System Requirements**: 
  - No GPU acceleration required (`isGpuEnabled`: false).
  - Minimum 4GB RAM recommended for handling the dataset.
  - Approximately 50MB of free disk space for the notebook, dataset, and output files.

## Installation
Follow these steps to set up the environment and run the notebook:

1. **Clone the Repository**:
   - Clone the project repository to your local machine:
     ```
     git clone hhttps://github.com/whslqy/Data_mining.git
     ```

2. **Install Dependencies**:
   - Create a virtual environment (optional but recommended):
     ```
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install the required Python packages:
     ```
     pip install pandas numpy scikit-learn matplotlib
     ```


3. **Launch Jupyter Notebook**:
   - Start the Jupyter Notebook server:
     ```
     jupyter notebook
     ```
   - Open `fakenews-passive-aggressive.ipynb` in your browser.

4. **Verify Setup**:
   - Run the first cell to check for missing dependencies or dataset issues. The notebook will fail gracefully if prerequisites are not met, providing error messages for troubleshooting.

## Usage
The notebook is divided into executable cells. Follow these steps to use it:

1. **Data Loading**:
   - Execute the initial cells to load the Kaggle dataset. Ensure the file path matches your local setup.
   - The dataset is expected to contain columns for news text and labels (e.g., `text` and `label`).

2. **Data Exploration**:
   - Review the exploratory data analysis (EDA) cells, which may include summary statistics, missing value checks, and a preview of the `fake-news.png` visualization.
   - This section helps understand the dataset's structure and identify preprocessing needs.

3. **Preprocessing**:
   - Run cells for text cleaning (e.g., removing stopwords, punctuation) and feature extraction (e.g., TF-IDF vectorization).
   - Adjust parameters like `max_features` or `ngram_range` in the TF-IDF transformer if needed.

4. **Model Training**:
   - Execute the training cells to initialize and fit the Passive-Aggressive classifier.
   - Default hyperparameters are used, but you can modify `C` (regularization parameter) or `max_iter` for optimization.

5. **Evaluation**:
   - Run the evaluation cells to generate performance metrics (e.g., accuracy, precision, recall, F1-score) and visualize results using the attached image or additional plots.
   - Check for overfitting or underfitting and adjust the model accordingly.

6. **Experimentation**:
   - Modify the code to experiment with different classifiers (e.g., SVM, Logistic Regression) or add cross-validation.
   - Save outputs or models using `joblib` or `pickle` if desired (add relevant cells).

## Troubleshooting
- **Dataset Not Found**: Verify the file path in the notebook matches the dataset location. Update the path or move the file.
- **Module Errors**: Ensure all libraries are installed. Use `pip list` to check versions and update if necessary (e.g., `pip install --upgrade scikit-learn`).
- **Performance Issues**: If the notebook runs slowly, reduce the dataset size or increase `max_iter` cautiously.
- **Visualization Errors**: Confirm `matplotlib` is installed and the `fake-news.png` file is correctly referenced.


## Additional Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html) for Passive-Aggressive classifier details.
- [Kaggle Dataset Page](https://www.kaggle.com/datasets/rajatkumar30/fake-news) for dataset information.
- [Jupyter Notebook Tips](https://jupyter-notebook.readthedocs.io/en/stable/) for optimizing your workflow.
- [Github repository](https://github.com/whslqy/Data_mining#) for viewing github repository.
- [Kaggle Dataset Page](https://www.kaggle.com/code/alkidiarete/fakenews-passive-aggressive) for viewing algorithm.