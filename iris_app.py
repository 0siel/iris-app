#This is my fisrt AI/ML project
#This is a a simple, interactive web application that uses the Iris dataset to predict flower species based on user input
##    Concepts covered: 
# Classification: a fundamental machine learning task where you categorize an input into a predefined class
# Model Training: the process of teaching a machine learning model to make predictions based on labeled data
# Data Preprocessing: preparing raw data for analysis, including normalization and encoding categorical variables
# Interactive application: using a simple framework to create a user-friendly interface for model interaction
##    Technologies used:
# Python: the programming language used for building the application
# Pandas: the library used for data manipulation and analysis
# Streamlit: the framework used for creating the web application
# Scikit-learn: the library used for implementing machine learning algorithms

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

# Streamlit configurations
# --- Page Configuration (Must be the first Streamlit command) ---
# Set the page configuration to have the sidebar expanded by default
st.set_page_config(    
    layout="centered",
    initial_sidebar_state="auto" # This is the key!
)

# --- 1. Load and prepare the Dataset 🔄️ ---

# Load the Iris dataset
iris = load_iris()

# iris.data contains the measurements (features)
# iris.target contains the species labels (0, 1, 2 for Setosa, Versicolor, Virginica)
# iris.feature_names contains the names of the features
# iris.target_names contains the names of the species

# Create a DataFrame for easier manipulation (optional but useful for understanding)

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map target numbers to species names for better readability
df['species'] = df['species'].apply(lambda x: iris.target_names[x])

# --- 2. Let's Train the Model baby! ⚙️---

# Split the dataset into training and testing sets
X = iris.data # Features (measurements)
y = iris.target # Target (species labels)


# The model learns from the training set and is evaluated on the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Random state ensures reproducibility

#    K-Nearest Neighbors (KNN) Classifier explanation: 
# KNN is a simple, instance-based learning algorithm that classifies new instances based on the majority class among their k-nearest neighbors in the feature space. 

# Create a KNN classifier with 3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model on the training data
knn.fit(X_train, y_train)



# --- 3. Build the Streamlit App 🖥️---

# Set the title of the app
st.title("My First AI/ML Project")
st.header("Iris Flower Species Prediction 🌸")
st.write("Hey there! I'm Osiel, this is my first AI/ML app, it predicts the species of an Iris flower based on its measurements.")
# Create sliders in the sidebar for user input
st.sidebar.header("Input Flower Measurements (cm)")

def user_input_features():
  sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
  sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
  petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
  petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)

  data={'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width}
  features = pd.DataFrame(data, index=[0])
  return features

# Get user input features
input_df = user_input_features()



# --- 4. Make Predictions and display the results 🔮---
# Make predictions using the trained model

# Display the user's input
st.subheader("Your Input")
st.write("Use the sliders in the sidebar (top-left corner) to adjust the flower's measurements:")
st.write(input_df)

# Predict the species using the KNN model
prediction = knn.predict(input_df)
prediction_proba = knn.predict_proba(input_df)

# Display the prediction
st.subheader("Prediction")
st.write(f"The model predicts this flower is a **{iris.target_names[prediction][0].capitalize()}**.")

# Display the prediction probabilities
st.subheader("Prediction Probability:")
st.write("How confident is the model about this prediction?")
st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))

# Display a corresponding image of the predicted species
st.subheader("Species Image")
if iris.target_names[prediction][0] == 'setosa':
  st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Irissetosa1.jpg/1024px-Irissetosa1.jpg", caption="Iris Setosa")

elif iris.target_names[prediction][0] == 'versicolor':
  st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Iris_versicolor_15-p.bot-iris.versi-20.jpg/800px-Iris_versicolor_15-p.bot-iris.versi-20.jpg", caption="Iris Versicolor")

else: 
  st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Iris_virginica_2.jpg/1024px-Iris_virginica_2.jpg", caption="Iris Virginica")

st.subheader("How does it work? 🤔")
st.write("This app uses a K-Nearest Neighbors (KNN) classifier to predict the species of an Iris flower based on its measurements. The model was trained on the Iris dataset, which contains measurements of different Iris flowers and their corresponding species labels.")
st.subheader("Tech Stack 🌼")
st.write("This app is built using the following technologies:")
st.write("- Python: the programming language used for building the application")
st.write("- Streamlit: the framework used for creating the web application")
st.write("- Scikit-learn: the library used for implementing machine learning algorithms")
st.write("- Pandas: the library used for data manipulation and analysis")

st.subheader("My portfolio 👨‍💻")
st.write("Check out my portfolio to see more of my work:")
st.write("[My Portfolio](https://portfolio-seven-rose-88.vercel.app/)")
