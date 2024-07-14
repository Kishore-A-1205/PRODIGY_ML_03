Cat and Dog Image Classification using Support Vector Machine (SVM)
This repository contains my project completed at Prodigy InfoTech, where I implemented an image classification model using a Support Vector Machine (SVM) to classify images of cats and dogs from the CIFAR-10 dataset.

Project Overview
In this project, I focused on building an effective image classification model. The key steps involved were:

Data Loading & Preprocessing:

Loaded the CIFAR-10 dataset.
Filtered the dataset to include only images of cats and dogs.
Flattened the images for easier processing.
Converted the labels to a binary format for classification.
Dimensionality Reduction:

Applied Principal Component Analysis (PCA) to reduce the dimensionality of the image data, enhancing the efficiency of the model.
Model Training:

Trained an SVM classifier on the PCA-transformed dataset.
Used a linear kernel to balance between training time and model performance.
Model Evaluation:

Evaluated the model on validation and test sets.
Achieved promising accuracy, demonstrating the model's effectiveness.
Key Learnings
Support Vector Machine (SVM): Effective for classification tasks, especially in high-dimensional spaces.
Dimensionality Reduction with PCA: Essential for managing high-dimensional data efficiently, reducing training time while retaining crucial features.
Binary Classification: Efficient preprocessing techniques are vital for handling image data effectively.
Tools & Technologies
Python
TensorFlow & Keras
Scikit-learn
Jupyter Notebook
