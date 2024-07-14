import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

print("Filtering for cats and dogs...")
cat_dog_train_idx = np.where((y_train == 3) | (y_train == 5))[0]
cat_dog_test_idx = np.where((y_test == 3) | (y_test == 5))[0]

x_train, y_train = x_train[cat_dog_train_idx], y_train[cat_dog_train_idx]
x_test, y_test = x_test[cat_dog_test_idx], y_test[cat_dog_test_idx]

print(f"Filtered x_train shape: {x_train.shape}")
print(f"Filtered y_train shape: {y_train.shape}")
print(f"Filtered x_test shape: {x_test.shape}")
print(f"Filtered y_test shape: {y_test.shape}")

print("Flattening images...")
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

print(f"Flattened x_train shape: {x_train.shape}")
print(f"Flattened x_test shape: {x_test.shape}")

print("Converting labels to binary...")
y_train = (y_train == 5).astype(int).ravel()
y_test = (y_test == 5).astype(int).ravel()

print(f"Binary y_train shape: {y_train.shape}")
print(f"Binary y_test shape: {y_test.shape}")

print("Splitting data into training and validation sets...")
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print(f"Training set shape: {x_train.shape}")
print(f"Validation set shape: {x_val.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Validation labels shape: {y_val.shape}")

print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=10)  
x_train_pca = pca.fit_transform(x_train)
x_val_pca = pca.transform(x_val)
x_test_pca = pca.transform(x_test)

print(f"Reduced x_train shape: {x_train_pca.shape}")
print(f"Reduced x_val shape: {x_val_pca.shape}")
print(f"Reduced x_test shape: {x_test_pca.shape}")

subset_size = 500  
x_train_subset = x_train_pca[:subset_size]
y_train_subset = y_train[:subset_size]

print("Training the SVM classifier on PCA-transformed subset data...")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(x_train_subset, y_train_subset)

print("SVM training complete.")

print("Validating the model...")
y_val_pred = log_reg.predict(x_val_pca)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

print("Testing the model...")
y_test_pred = log_reg.predict(x_test_pca)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
