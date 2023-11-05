# Using the support vector
# Classification model to predict whether a pastry is a muffin or cupcake
# based on its sugar and flour content.

import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.2)

recipes = pd.read_csv("Cupcakes vs Muffins.csv")
print(recipes.head(), "\n")

# to create a seaborn plot from a dataframe
sns.lmplot(data=recipes, x='Flour', y='Sugar', hue='Type',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70})

# creating a binary data series based off the condition Muffin or not

type_label = np.where(recipes['Type'] == 'Muffin', 0, 1)

# finding out all the recipe features(independent variables) based on the column names
# all the factors to consider
recipe_features = recipes.columns.values[1:].tolist()

# we will focus on two ingredients or two features or factors

ingredients = recipes[['Flour', 'Sugar']].values
# print(ingredients)

# create and fit model
# sklearn has multiples such as SVC, svr...

model = svm.SVC(kernel='linear')
model.fit(ingredients, type_label)

# to get the separating hyperplane
w = model.coef_[0]

a = -w[0] / w[1]  # Slope

xx = np.linspace(30, 60)

yy = a * xx - (model.intercept_[0] / w[1])

# Plot the parallels to the separating hyperline that pass through the support vectors

b = model.support_vectors_[0]

yy_down = a * xx + (b[1] - a * b[0])

b = model.support_vectors_[-1]

yy_up = a * xx + (b[1] - a * b[0])

plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(xx, yy_down, "k--")
plt.plot(xx, yy_up, "k--")


# plt.show()


# creating a function to determine if we have muffin or cupcake depending on the
# two ingredients

def muffin_or_cupcake(flour, sugar):
    if (model.predict([[flour, sugar]])) == 0:
        print("This is a muffin recipe!")
    else:
        print("This is a cupcake recipe!")


muffin_or_cupcake(50, 20)