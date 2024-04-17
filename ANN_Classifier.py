import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, log_loss, recall_score, f1_score


# Binary Classification using a Neural Network==========================================================================
# ======================================================================================================================

# Load the dataset
train_df = pd.read_csv('dota2Train.csv', header=None)
test_df = pd.read_csv('dota2Test.csv', header=None)

# Combine train and test data to support random samples
dota2_data = pd.concat([train_df, test_df], ignore_index=True)

# Separate target and features
X = dota2_data.drop(0, axis=1)
y = dota2_data[0]

# Feature Selection ==========================
# Recursive Feature Elimination RFE to reduce number of features based on importance
# random forest classifier is used because it inherently provides feature importance scores

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=90)
# rfe.fit(X_train, y_train)
# X_train = rfe.transform(X_train)
# X_test = rfe.transform(X_test)

# Create and train the ANN classifier
# MLPClassifier parameters:
# activation function = rectified linear unit function
# optimization for weights = 'adam' is a stochastic gradient-based optimizer that works well with large datasets
# alpha = strength of the L2 regularization term
# max_iter = max epochs to use
# learning_rate_init = controls initial step-size when updating the weights for solver = 'adam' or 'sgd'
classifier = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=200, random_state=42, activation='relu',
                           solver='adam', alpha=0.01, learning_rate_init=0.0001)

# Store training and validation losses
training_losses = []
validation_losses = []

# calculate losses during fitting
for epoch in range(classifier.max_iter):
    classifier.partial_fit(X_train, y_train, classes=np.unique(y_train))

    # Calculate training loss
    train_loss = classifier.loss_
    training_losses.append(train_loss)

    # Calculate validation loss
    y_val_pred = classifier.predict_proba(X_test)
    val_loss = log_loss(y_test, y_val_pred)
    validation_losses.append(val_loss)

# Predict on the test data
y_pred = classifier.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Generate the confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print("Confusion Matrix:")
print(confusion_mat)

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(training_losses, label="Training Loss", linestyle='-', linewidth=3)
plt.plot(validation_losses, label="Validation Loss", linestyle='-', linewidth=3)
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.title("Training vs Validation Loss", fontsize=16)
plt.show()

# Confusion Matrix Heatmap
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in confusion_mat.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     confusion_mat.flatten()/np.sum(confusion_mat)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=labels, fmt='', cmap="Blues", square=True,
            xticklabels=[0, 1],
            yticklabels=[0, 1])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('ANN Confusion Matrix')
plt.show()

# Bar graph of classification metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
fig, ax = plt.subplots()
ax.bar(metrics, values)
ax.set_ylabel('Score')
ax.set_title('Performance Metrics for ANN')
ax.set_ylim(0, 1)

for i in range(len(values)):
    ax.text(i, values[i] + 0.01, f'{values[i]:.2f}', ha='center')

plt.show()
