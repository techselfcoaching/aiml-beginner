#Goal: Predict a student’s score using multiple factors: study hours, attendance, and past exam performance.

#Step 1: Install required packages

#pip install pandas matplotlib scikit-learn ipywidgets

#Step 2: Import packages

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from ipywidgets import interact, IntSlider

#Step 3: Create sample dataset

data = {
"Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
"Attendance": [60, 65, 70, 75, 80, 85, 90, 95, 100, 100],
"Past_Performance": [50, 55, 60, 62, 65, 70, 75, 80, 85, 90],
"Score": [10, 18, 25, 32, 40, 50, 60, 70, 80, 90]
}

df = pd.DataFrame(data)
print(df)

#Step 4: Train Multiple Linear Regression Model

X = df[["Study_Hours", "Attendance", "Past_Performance"]]
y = df["Score"]

model = LinearRegression()
model.fit(X, y)

#Step 5: Define interactive prediction function

def predict_score(study_hours, attendance, past_perf):
predicted = model.predict([[study_hours, attendance, past_perf]])[0]
print(f"Predicted Score: {predicted:.2f}")
# Optional: visualize study hours vs score
plt.scatter(df["Study_Hours"], df["Score"], color='blue', label='Data')
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Student Score Prediction (Study Hours vs Score)")
plt.show()

#Step 6: Add interactive sliders

interact(predict_score,
study_hours=IntSlider(min=0, max=12, step=1, value=5),
attendance=IntSlider(min=50, max=100, step=5, value=80),
past_perf=IntSlider(min=40, max=100, step=5, value=70))