import os
import pickle
from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy as np
from chicksexer import CLASS2DEFAULT_CUTOFF, POSITIVE_CLASS, NEGATIVE_CLASS, NEUTRAL_CLASS
from Preprocessor import gen_name_gender_from_csv
from Preprocessor import Name2Proba
from chicksexer import NameClassifier
from Preprocessor import compute_gender_probas


PACKAGE_ROOT = os.path.dirname('.')


_RAW_DATA_ROOT = 'Data'
_PROCESSED_DATA_PATH = 'name2proba_{}.pkl'
_NEUTRAL_NAME_AUGMENTATION_NUM = 100000
_FEMALE_NAME_AUGMENTATION_NUM = 85000
_TEST_DATA_SIZE = 10000  # the size of the whole dataset is ~400,000

_CLASS2PROB = {
    POSITIVE_CLASS: 1.,
    NEUTRAL_CLASS: 0.5,
    NEGATIVE_CLASS: 0.,
}

def _process_csv(name2probfa):
    """Process csv files that list names and their gender."""
    file_names = ['Black-Female-Names.csv', 'Black-Male-Names.csv', 'White-Male-Names.csv', 'White-Female-Names.csv']

    for file_name in file_names:
        for name, gender in gen_name_gender_from_csv(os.path.join(_RAW_DATA_ROOT, file_name)):
            proba = _CLASS2PROB[gender]
            name2probfa[name] = proba
    return name2probfa



def _process_us_stats(name2proba, start_year=1940):
    """Process yobxxxx.txt files that list first names and their gender."""
    dir_path = os.path.join(_RAW_DATA_ROOT, 'names')
    name2proba_stats = compute_gender_probas(dir_path, start_year)
    for name, proba in name2proba_stats.items():
        name2proba.set_fix_item(name, proba)
    return name2proba



name2proba = Name2Proba()

# name2proba = _process_dbpedia(name2proba)

name2proba = _process_us_stats(name2proba)

name2proba = _process_csv(name2proba)



# name2proba = _process_common_names(name2proba)

# name2proba = _augment_full_names(name2proba, 'neutral')

# name2proba = _augment_full_names(name2proba, 'female')


# randomly split into train/test set
name2proba = dict(name2proba)
assert len(name2proba) > _TEST_DATA_SIZE, 'Whole dataset size is not larger than test set size.'
ids = list(range(len(name2proba)))
shuffle(ids)
test_ids = set(ids[:_TEST_DATA_SIZE])
name2proba_train = dict()
name2proba_test = dict()

for id_, (name, proba) in enumerate(name2proba.items()):
    if id_ in test_ids:
        name2proba_test[name] = proba
    else:
        name2proba_train[name] = proba

# write to pickle files
with open(_PROCESSED_DATA_PATH.format('train'), 'wb') as train_file:
    pickle.dump(name2proba_train, train_file)
with open(_PROCESSED_DATA_PATH.format('test'), 'wb') as test_file:
    pickle.dump(name2proba_test, test_file)
with open(_PROCESSED_DATA_PATH.format('all'), 'wb') as all_file:
    pickle.dump(name2proba, all_file)


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

POSITIVE_CLASS = "Male"
NEGATIVE_CLASS = "Female"

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
model = NameClassifier(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'name_classifier.pth')


