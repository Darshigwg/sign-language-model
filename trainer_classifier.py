import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

try:
    with open('./data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print("File not found. Make sure the path to the pickle file is correct.")
    exit()

if 'data' not in data_dict or 'labels' not in data_dict:
    print("Invalid format of pickle file. 'data' and 'labels' keys are required.")
    exit()

data = data_dict['data']
max_length = max(len(seq) for seq in data)
data_padded = [np.pad(seq, (0, max_length - len(seq)), mode='constant') for seq in data]
data = np.asarray(data_padded)
labels = np.asarray(data_dict['labels'])

# Count occurrences of each class
class_counts = {label: np.sum(labels == label) for label in np.unique(labels)}

# Find classes with only one member
single_member_classes = [label for label, count in class_counts.items() if count == 1]

# Remove samples corresponding to single-member classes
indices_to_keep = np.where(~np.isin(labels, single_member_classes))
data = data[indices_to_keep]
labels = labels[indices_to_keep]

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

with open('model.p', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully as 'model.p'")
