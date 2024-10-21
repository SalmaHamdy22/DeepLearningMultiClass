
# Multi Classification Using Artificial Neural Networks (ANN) in PyTorch

The objective of this project is to classify different species of iris flowers using a neural network implemented with PyTorch. The goal is to achieve a model with an accuracy of ≥ 95% on the test dataset.



# Dataset description

The dataset contains 150 samples of iris flowers with the following features:

Sepal Length
Sepal Width
Petal Length
Petal Width

The target variable is species, which can belong to one of three classes:

Iris-setosa
Iris-versicolor
Iris-virginica

The dataset is processed by checking for null values, normalizing the features, and then being used to train a neural network model for multiclass classification.



## Steps to Run the Code in Jupyter Notebook

-Open Anaconda:

You can upload this code directly to a Jupyter notebook.

-Upload or Load the Dataset:

Use the following command to load the dataset from a CSV file:

iris = pd.read_csv("path_to_your_file/iris.csv")

-Run the Code:

Once the dataset is uploaded or loaded, execute the code cells sequentially in the Jupyter notebook.
The code will handle preprocessing, model training, evaluation, and visualization.
## Dependencies and Installation Instructions
bash

!!pip install torch torchvision numpy pandas matplotlib scikit-learn sklearn metrics

## Example Output

After training the neural network, you will visualize the results including:

Confusion Matrix: A heatmap representation showing the performance of the classification model.
ROC Curves: Plots for each class, illustrating the true positive rate against the false positive rate.
Model Accuracy: The overall accuracy of the model on the test dataset.
