#Create a new Jupyter Notebook.
#Install and import packages:
#pip install pandas matplotlib

import pandas as pd
import matplotlib.pyplot as plt

data = {
"Name": ["Alice", "Bob", "Charlie", "David"],
"Age": [24, 30, 22, 35],
"Score": [85, 90, 78, 92]
}

df = pd.DataFrame(data)
print(df)

#Simple visualization

df.plot(x='Name', y='Score', kind='bar', title='Scores of Students')
plt.show()