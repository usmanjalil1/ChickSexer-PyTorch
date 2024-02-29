import torch
import torch.nn as nn
from chicksexer import NameClassifier
import pickle
from chicksexer import CLASS2DEFAULT_CUTOFF, POSITIVE_CLASS, NEGATIVE_CLASS, NEUTRAL_CLASS

_CLASS2PROB = {
    POSITIVE_CLASS: 1.,
    NEUTRAL_CLASS: 0.5,
    NEGATIVE_CLASS: 0.,
}

def _get_training_data(train_data_path):
    """Load training data from the file and return them."""
    names = list()
    y = list()
    with open(train_data_path, 'rb') as pickle_file:
        name2proba = pickle.load(pickle_file)

    for name, proba in name2proba.items():
        names.append(name)
        y.append(proba)

    return names, y
x_train,y_train = _get_training_data('name2proba_train.pkl')



names = x_train
probabilities = y_train

# POSITIVE_CLASS = "Male"
# NEGATIVE_CLASS = "Female"

classified_names = {POSITIVE_CLASS: [], NEGATIVE_CLASS: []}
for name, prob in zip(names, probabilities):
    if prob >= 0.8:
        classified_names[POSITIVE_CLASS].append(name)
    else:
        classified_names[NEGATIVE_CLASS].append(name)

def name_to_onehot(name):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    onehot = [0] * (len(alphabet) * 2)  # Multiply by 2 for both lower and upper case
    for char in name.lower():
        if char in alphabet:
            index = alphabet.index(char)
            onehot[index] = 1
    return onehot

encoded_names = {}
for class_name, names_list in classified_names.items():
    encoded_names[class_name] = [name_to_onehot(name) for name in names_list]

data = []
labels = []
for class_name, encoded_list in encoded_names.items():
    data.extend(encoded_list)
    labels.extend([1 if class_name == POSITIVE_CLASS else 0] * len(encoded_list))

data = torch.tensor(data, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)  # Reshape for proper shape



input_size = len(data[0])
model = NameClassifier(input_size=input_size)
model.load_state_dict(torch.load('name_classifier.pth'))

def predict(name, model):
    name_encoded = torch.tensor([name_to_onehot(name)], dtype=torch.float32)
    with torch.no_grad():
        output = model(name_encoded)
        prob = output.item()
    if prob >= CLASS2DEFAULT_CUTOFF[POSITIVE_CLASS]:
        return POSITIVE_CLASS
    elif prob <= CLASS2DEFAULT_CUTOFF[NEGATIVE_CLASS]:
        return NEGATIVE_CLASS
    else:
        return NEUTRAL_CLASS
def test_predict(name:str):
    name_list = [name]
    for name in name_list:
        predicted_gender = predict(name, model)
        print(f'Its a {predicted_gender} name')


name = input('Enter your name: ')
test_predict(name)