import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Function to parse a single row
def parse_row(row):
    # Extracting the features
    team_won = int(row[0])
    heroes = [int(hero) for hero in row[4:]]
    
    # Returning the parsed data
    return {
        'team_won': team_won,
        'heroes': heroes
    }

# Path to the train and test CSV files
train_csv_file_path = 'dota2Train.csv'
test_csv_file_path = 'dota2Test.csv'

# List to store parsed train and test data
parsed_train_data = []
parsed_test_data = []

# Reading the train CSV file
with open(train_csv_file_path, newline='') as train_csvfile:
    reader = csv.reader(train_csvfile)
    for row in reader:
        parsed_train_data.append(parse_row(row))

# Reading the test CSV file
with open(test_csv_file_path, newline='') as test_csvfile:
    reader = csv.reader(test_csvfile)
    for row in reader:
        parsed_test_data.append(parse_row(row))

# Prepare train and test features and labels
X_train = []
y_train = []
X_test = []
y_test = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=None)

for game in parsed_train_data:
    # Concatenate team1 and team2 heroes for training
    features = game['heroes']
    label = game['team_won']
    X_train.append(features)
    y_train.append(label)

for game in parsed_test_data:
    # Concatenate team1 and team2 heroes for testing
    features = game['heroes']
    label = game['team_won']
    X_test.append(features)
    y_test.append(label)

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the SVM classifier
svm_classifier.fit(X_train_scaled, y_train)

# Predictions on the test set
y_pred = svm_classifier.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
