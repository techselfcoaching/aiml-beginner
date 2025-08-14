#Interactive Mini AI/ML Project: Predict Student Scores with Slider

#Step 1: Install required packages

pip install pandas matplotlib scikit-learn ipywidgets

#Step 2: Import packages

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from ipywidgets import interact, IntSlider

#Step 3: Create sample dataset

data = {
"Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
"Score": [10, 20, 28, 35, 45, 50, 61, 68, 79, 85]
}

df = pd.DataFrame(data)

#Step 4: Train Linear Regression Model

X = df[["Hours_Studied"]]
y = df["Score"]

model = LinearRegression()
model.fit(X, y)

#Step 5: Define interactive prediction function

def predict_score(hours):
    predicted = model.predict([[hours]])[0]
    print(f"Predicted Score for {hours} hours studied: {predicted:.2f}")
    plt.scatter(df["Hours_Studied"], df["Score"], color='blue')
    plt.plot(df["Hours_Studied"], model.predict(X), color='red')
    plt.scatter(hours, predicted, color='green', s=100)
    plt.xlabel("Hours Studied")
    plt.ylabel("Score")
    plt.title("Linear Regression: Hours Studied vs Score")
    plt.show()

#Step 6: Add interactive slider

interact(predict_score, hours=IntSlider(min=0, max=12, step=1, value=5))