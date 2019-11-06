__author__ = 'Venushka Thisara'
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Transformations
NRM = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
TT = transforms.ToTensor()

# Transforms object for testset with NO augmentation
transform_no_aug = transforms.Compose([TT, NRM])

# Downloading/Louding CIFAR10 data
trainset = CIFAR10(root='./data', train=True, download=True)  # , transform = transform_with_aug)
testset = CIFAR10(root='./data', train=False, download=True)  # , transform = transform_no_aug)
classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
             'truck': 9}

# Separating train set/ test set data/label
x_train = trainset.data
x_test = testset.data
y_train = trainset.targets
y_test = testset.targets


# Define a function to separate CIFAR classes by class index

def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i


class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc=transform_no_aug):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class


# ================== Usage ================== #

plane = get_class_i(x_train, y_train, classDict['plane'])
# for training data 60%
start_index = int(len(plane)*0.6)
end_index = int(len(plane))
train_plane = plane[0:start_index]
# for testing data 40%
test_plane = plane[start_index:end_index]


car = get_class_i(x_train, y_train, classDict['car'])
start_index = int(len(car)*0.6)
end_index = int(len(car))
# for training data 60%
train_car = car[0:start_index]
# for testing data 40%
test_car = car[start_index:end_index]

bird = get_class_i(x_train, y_train, classDict['bird'])
# 50% of bird data
bird = bird[0:int(len(bird)/2)]
start_index = int(len(bird)*0.6)
end_index = int(len(bird))
# for training data 60%
train_bird = bird[0:start_index]
# for testing data 40%
test_bird = bird[start_index:end_index]


cat = get_class_i(x_train, y_train, classDict['cat'])
start_index = int(len(cat)*0.6)
end_index = int(len(cat))
# for training data 60%
train_cat = cat[0:start_index]
# for testing data 40%
test_cat = cat[start_index:end_index]


deer = get_class_i(x_train, y_train, classDict['deer'])
# 50% of deer data
deer = deer[0:int(len(deer)/2)]
start_index = int(len(deer)*0.6)
end_index = int(len(deer))
# for training data 60%
train_deer = deer[0:start_index]
# for testing data 40%
test_deer = deer[start_index:end_index]


dog = get_class_i(x_train, y_train, classDict['dog'])
start_index = int(len(dog)*0.6)
end_index = int(len(dog))
# for training data 60%
train_dog = dog[0:start_index]
# for testing data 40%
test_dog = dog[start_index:end_index]

frog = get_class_i(x_train, y_train, classDict['frog'])
start_index = int(len(frog)*0.6)
end_index = int(len(frog))
# for training data 60%
train_frog = frog[0:start_index]
# for testing data 40%
test_frog = frog[start_index:end_index]

horse = get_class_i(x_train, y_train, classDict['horse'])
start_index = int(len(horse)*0.6)
end_index = int(len(horse))
# for training data 60%
train_horse = horse[0:start_index]
# for testing data 40%
test_horse = horse[start_index:end_index]

ship = get_class_i(x_train, y_train, classDict['ship'])
start_index = int(len(ship)*0.6)
end_index = int(len(ship))
# for training data 60%
train_ship = ship[0:start_index]
# for testing data 40%
test_ship = ship[start_index:end_index]

truck = get_class_i(x_train, y_train, classDict['truck'])
# 50% of truck data
truck = truck[0:int(len(truck)/2)]
start_index = int(len(truck)*0.6)
end_index = int(len(truck))
# for training data 60%
train_truck = cat[0:start_index]
# for testing data 40%
test_truck = cat[start_index:end_index]

trainset = \
    DatasetMaker(
        [train_plane, train_car, train_bird, train_deer, train_frog, train_horse, train_truck],
        transform_no_aug
    )

testset = \
    DatasetMaker(
        [test_plane, test_car, test_bird, test_deer, test_frog, test_horse, test_truck],
        transform_no_aug
    )

kwargs = {'num_workers': 8, 'pin_memory': False}

# Create datasetLoaders from train set and test set
dataloader = DataLoader(trainset, batch_size=64, shuffle=True, **kwargs)
testloader = DataLoader(testset, batch_size=64, shuffle=False, **kwargs)
