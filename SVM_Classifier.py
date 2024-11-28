# In[0]

import os
import numpy as np
import librosa
from scipy.stats import skew, kurtosis
from noisereduce import reduce_noise

def preprocess_and_extract_file_level_features(data_path, sample_rate=48000):
    """
    Preprocess audio files by denoising and extracting global statistics
    for the entire file (e.g., mean, median, skew of MFCCs).

    Args:
        data_path (str): Path to the directory containing .wav files.
        sample_rate (int): Sample rate for audio processing.

    Returns:
        features (list): List of global feature vectors for each audio file.
        labels (list): List of user labels corresponding to each feature vector.
    """
    features = []
    labels = []

    # Iterate through all .wav files in the directory
    for file_name in os.listdir(data_path):
        if file_name.endswith('.wav'):
            # Load the audio file
            file_path = os.path.join(data_path, file_name)
            audio, sr = librosa.load(file_path, sr=sample_rate)

            # Step 1: Denoise the audio
            audio_denoised = reduce_noise(y=audio, sr=sr, prop_decrease=0.8)

            # Step 2: Calculate MFCC coefficients for the entire audio sequence
            mfcc_data = librosa.feature.mfcc(y=audio_denoised, sr=sr, n_mfcc=3, n_fft=960, hop_length=480)

            # Step 3: Calculate global statistics for MFCCs
            mean_mfcc = np.mean(mfcc_data, axis=1)
            median_mfcc = np.median(mfcc_data, axis=1)
            std_mfcc = np.std(mfcc_data, axis=1)
            skew_mfcc = skew(mfcc_data, axis=1)
            kurt_mfcc = kurtosis(mfcc_data, axis=1)
            maximum_mfcc = np.amax(mfcc_data, axis=1)
            minimum_mfcc = np.amin(mfcc_data, axis=1)

            # Step 4: Combine all statistics into a single feature vector
            feature_vector = np.concatenate(
                (mean_mfcc, median_mfcc, std_mfcc, skew_mfcc, kurt_mfcc, maximum_mfcc, minimum_mfcc)
            )
            features.append(feature_vector)

            # Step 5: Extract the label (username) from the file name
            label = file_name.split('_')[0]  # Extract "farmerjack" from "farmerjack_10.wav"
            labels.append(label)

    return np.array(features), np.array(labels)

# Example usage
train_data_path = 'traindata'
test_data_path = 'testdata'

# Extract file-level features for training and testing datasets
X_train, Y_train = preprocess_and_extract_file_level_features(train_data_path, sample_rate=48000)
X_test, Y_test = preprocess_and_extract_file_level_features(test_data_path, sample_rate=48000)

# Save features to a file (optional)
pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
pd.DataFrame(Y_train, columns=['label']).to_csv('Y_train.csv', index=False)
pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
pd.DataFrame(Y_test, columns=['label']).to_csv('Y_test.csv', index=False)

# In[1]
from sklearn.preprocessing import StandardScaler

# Normalize the training and testing features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[2]

from sklearn import svm

# Initialize and train the SVM
svm_model = svm.SVC(kernel='rbf', decision_function_shape='ovr')
svm_model.fit(X_train, Y_train)

# Save the trained model (optional)
import joblib
joblib.dump(svm_model, 'svm_voice_recognition_model.pkl')

# In[3]

# Predict the labels for the test dataset
predictions = svm_model.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(Y_test, predictions))
print(classification_report(Y_test, predictions))

# In[]

from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import numpy as np

# Perform ANOVA test
f_values, p_values = f_classif(X_train, Y_train)

# Sort features by F-value for visualization
sorted_indices = np.argsort(f_values)[::-1]  # Descending order
sorted_f_values = f_values[sorted_indices]
sorted_p_values = p_values[sorted_indices]

# Plot the top N features by F-value
top_n = 20  # Number of features to display
plt.figure(figsize=(12, 6))
plt.bar(range(top_n), sorted_f_values[:top_n], align='center')
plt.xticks(range(top_n), sorted_indices[:top_n], rotation=90)
plt.xlabel("Feature Index")
plt.ylabel("F-value")
plt.title("Top 20 Features by ANOVA F-value")
plt.show()

# In[]

# For 1 MFCC
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import numpy as np

# Perform ANOVA to calculate F-values and p-values
f_values, p_values = f_classif(X_train, Y_train)

# Sort features by F-value for better visualization
sorted_indices = np.argsort(f_values)[::-1]  # Descending order
sorted_f_values = f_values[sorted_indices]
sorted_p_values = p_values[sorted_indices]

# Plot both F-values and p-values
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot F-values on the left y-axis
color = 'tab:blue'
ax1.set_xlabel("Feature Index (Sorted by F-value)")
ax1.set_ylabel("F-value", color=color)
ax1.bar(range(len(sorted_f_values)), sorted_f_values, color=color, alpha=0.7, label="F-value")
ax1.tick_params(axis='y', labelcolor=color)

# Add a secondary y-axis for p-values
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel("p-value", color=color)
ax2.plot(range(len(sorted_p_values)), sorted_p_values, color=color, marker='o', linestyle='-', label="p-value")
ax2.axhline(y=0.05, color='green', linestyle='--', label="Significance Threshold (p=0.05)")
ax2.tick_params(axis='y', labelcolor=color)

# Add a legend and title
fig.tight_layout()
plt.title("F-values and p-values for Features with n_mfcc=1")
fig.legend(loc="upper right", bbox_to_anchor=(1, 0.85))
plt.show()

# In[]

#Visualizing MFCC Values
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualize_feature_variance(feature_number, features, labels, feature_name="MFCC Mean"):
    """
    Visualize the variance of a specific feature across classes.

    Args:
        features (np.ndarray): Feature matrix (samples x features).
        labels (np.ndarray): Corresponding labels for each sample.
        feature_name (str): Name of the feature being visualized (e.g., "MFCC Mean").
    """
    # Convert labels to a pandas DataFrame for grouping
    df = pd.DataFrame(features)
    df['label'] = labels

    # Sort labels for better grouping
    df = df.sort_values(by='label')

    # Plot each sample's feature value grouped by speaker
    unique_labels = sorted(np.unique(labels))
    plt.figure(figsize=(12, 6))

    for label in unique_labels:
        class_samples = df[df['label'] == label].iloc[:, feature_number]  # Assume MFCC mean is the first column
        x_values = [label] * len(class_samples)
        plt.scatter(x_values, class_samples, label=f"Speaker {label}", alpha=0.6)

    plt.xlabel("Speaker (Class)")
    plt.ylabel(f"{feature_name} Value")
    plt.title(f"{feature_name} Distribution Across Speakers")
    plt.grid(True)
    plt.show()

# Example usage
for i in sorted_indices[:5]:
    print(i)
    visualize_feature_variance(i, X_train[:, :i+1], Y_train, feature_name=i)



# In[]

# plot confusion matrix
from sklearn.metrics import confusion_matrix
import scipy.stats as stats

cm = confusion_matrix(Y_test, predictions)
print("Confusion Matrix:\n", cm)



chi2, p, dof, expected = stats.chi2_contingency(cm)
print(f"Chi2: {chi2}, p-value: {p}")

from sklearn.model_selection import cross_val_score

f1_scores = cross_val_score(svm_model, X_train, Y_train, cv=5, scoring='f1_macro')
print("F1-scores across folds:", f1_scores)

from scipy.stats import ttest_1samp

baseline_f1 = 0.5  # Example baseline (chance-level performance)
t_stat, p_value = ttest_1samp(f1_scores, baseline_f1)
print(f"T-test p-value: {p_value}")
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_normalized_confusion_matrix(y_true, y_pred, label_names, classifier_name="SVM"):
    """
    Custom visualization for a normalized confusion matrix.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        label_names (list): List of class label names.
        classifier_name (str): Name of the classifier (default: "SVM").
    """
    # Compute the confusion matrix
    confusion_mat = confusion_matrix(np.array(y_true), y_pred, labels=label_names)

    # Normalize the confusion matrix by row (percentage per true class)
    row_sums = confusion_mat.sum(axis=1, keepdims=True)
    normalized_confusion_mat = confusion_mat / row_sums

    # Create the plot
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(normalized_confusion_mat, cmap=plt.cm.Blues, interpolation='nearest')

    # Set the x, y, and title labels for the plot
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.title(f"Normalized Confusion Matrix of {classifier_name} Classifier", fontsize=14)

    # Arrange the label names on the x and y axis
    plt.xticks(np.arange(len(label_names)), label_names, rotation=45, fontsize=10)
    plt.yticks(np.arange(len(label_names)), label_names, fontsize=10)
    plt.tick_params(axis='both', labelsize='10')

    # Display the values in the normalized confusion matrix
    for (y, x), value in np.ndenumerate(normalized_confusion_mat):
        if row_sums[y] != 0:  # Avoid division by zero for empty rows
            plt.text(x, y, f"{value:.2f}", ha='center', va='center', size='10', color='black')

    # Add a colorbar
    plt.colorbar()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# Example usage
label_names = sorted(set(Y_test))  # Use the unique class names from your dataset
plot_normalized_confusion_matrix(Y_test, predictions, label_names, classifier_name="SVM")
# %%

from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np

# Initialize the SVM classifier
svm_model = svm.SVC(kernel='rbf', C=1, gamma='scale', decision_function_shape='ovr')

# Perform 5-fold cross-validation
cv_scores = cross_val_score(svm_model, X_train, Y_train, cv=5, scoring='accuracy')

# Print results
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean accuracy: {np.mean(cv_scores):.2f}")
print(f"Standard deviation: {np.std(cv_scores):.2f}")
# %%
