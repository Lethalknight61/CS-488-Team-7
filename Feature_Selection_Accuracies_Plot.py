import matplotlib.pyplot as plt

# accuracies derived from looping RFE runs with different numbers of retained features
# RFE estimator: RandomForestClassifier
# Classifier: MLPClassifier
# plot accuracies for different numbers of features
accuracies = [0.5289, 0.5282, 0.5310, 0.5547, 0.5626, 0.5810, 0.5865, 0.5924, 0.5997, 0.5867]
features = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.figure(figsize=(10, 6))
plt.plot(features, accuracies, label='MLPClassifier', linestyle='-', linewidth=3)
plt.xlabel("Number of Features", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.title("Average Accuracy for Different Numbers of Features", fontsize=14)
plt.show()
