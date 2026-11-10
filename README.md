# Student Academic Performance Prediction Using Machine Learning

## Project Overview

This project develops a machine learning application to estimate students' final academic grades using demographic, family, school, and study-related information.

The project uses the Student Performance dataset from the UCI Machine Learning Repository. The Portuguese language dataset contains 649 student records.

The target variable is **G3 (final grade)**, measured on a scale of 0–20. The earlier period grades G1 and G2 were excluded from the primary prediction experiment to investigate whether academic performance can be estimated without using those previous grades.

## Objectives

- Explore and understand student academic performance data.
- Perform data cleaning and preprocessing.
- Train and compare multiple regression algorithms.
- Evaluate model performance using MAE, RMSE, and R².
- Interpret important features influencing model predictions.
- Develop a web application using Flask.

## Machine Learning Models

The following regression models were implemented and evaluated:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The tuned Gradient Boosting model achieved the following test results:

| Metric | Result |
|---|---:|
| MAE | 2.049 |
| RMSE | 2.768 |
| R² | 0.214 |

The results indicate that the model has predictive value but also has limitations.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Flask
- Joblib
- HTML and CSS
- Google Colab
- GitHub

## Project Structure

```text
student-data-preprocessing/
│
├── app.py
├── requirements.txt
├── student_performance_model.pkl
├── templates/
│   └── index.html
├── README.md
└── Project Notebooks
```

## How to Run the Application

1. Clone this repository:

```bash
git clone https://github.com/bhavyashah1306/student-data-preprocessing.git
```

2. Open the project directory:

```bash
cd student-data-preprocessing
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Start the Flask application:

```bash
python app.py
```

5. Open your browser and visit:

```text
http://127.0.0.1:5000/
```

Enter the student information in the web application to receive an estimated final grade and general academic monitoring suggestions.

## Dataset Source

UCI Machine Learning Repository:

[Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

Cortez, P., and Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance.

## Important Limitations

- The dataset represents students from only two Portuguese schools.
- The model has not been validated for Indian students or other school systems.
- The model's predictions are estimates, not guaranteed final grades.
- Feature importance indicates model reliance, not causation.
- The application should not be used as the sole basis for academic decisions or student support.

This project is an educational machine learning research prototype.

## Author

**Bhavya Shah**

GitHub: [bhavyashah1306](https://github.com/bhavyashah1306)
