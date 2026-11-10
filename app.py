
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the already-trained machine learning model
model = joblib.load("student_performance_model.pkl")

# Numerical input columns
numeric_columns = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout", "Dalc",
    "Walc", "health", "absences"
]

def get_support_recommendation(grade):

    if grade < 10:
        return {
            "category": "Priority Support",
            "action": (
                "Arrange a teacher check-in, identify possible learning "
                "difficulties, and consider additional academic support."
            )
        }

    elif grade < 12:
        return {
            "category": "Needs Monitoring",
            "action": (
                "Monitor academic progress, discuss areas of difficulty, "
                "and provide guidance where needed."
            )
        }

    elif grade < 15:
        return {
            "category": "Routine Monitoring",
            "action": (
                "Encourage consistent study habits and continue regular "
                "academic monitoring."
            )
        }

    else:
        return {
            "category": "Positive Progress",
            "action": (
                "Encourage continued progress, maintain support, "
                "and provide opportunities for further development."
            )
        }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data received."}), 400

        input_data = pd.DataFrame([data])

        # Convert numerical fields into numbers
        for col in numeric_columns:
            input_data[col] = pd.to_numeric(
                input_data[col], errors="raise"
            )

        # Predict the final grade
        prediction = float(model.predict(input_data)[0])

        # Get the support recommendation
        recommendation = get_support_recommendation(prediction)

        return jsonify({
            "predicted_final_grade": round(prediction, 2),
            "grade_scale": "0-20",
            "factors_to_consider": [
                {
                    "factor": "Previous Class Failures",
                    "explanation": "Previous academic difficulties were an influential feature in the model."
                },
                {
                    "factor": "School",
                    "explanation": "The school attended was influential in this dataset. This may reflect differences between the two schools."
                },
                {
                    "factor": "Interest in Higher Education",
                    "explanation": "Whether the student plans to pursue higher education contributed to the model's predictions."
                },
                {
                    "factor": "Study Time",
                    "explanation": "The student's reported weekly study time was among the influential features."
                },
                {
                    "factor": "Extra School Support",
                    "explanation": "Whether the student receives extra educational support was also relevant to the model."
                }
            ],
            "support_category": recommendation["category"],
            "suggested_action": recommendation["action"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
