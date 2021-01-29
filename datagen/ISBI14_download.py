import numpy as np
from scipy.io import loadmat
import h5py
import os
import matplotlib.pyplot as plt
import wget
import zipfile

# download and extract data
compressed_file1 = wget.download('https://cs.adelaide.edu.au/~carneiro/isbi14_challenge/Train45Test90.zip')
compressed_file2 = wget.download('https://cs.adelaide.edu.au/~carneiro/isbi14_challenge/Dataset.zip')

with zipfile.ZipFile(compressed_file1, 'r') as zip_ref:
    zip_ref.extractall('data/ISBI14/')
with zipfile.ZipFile(compressed_file2, 'r') as zip_ref:
    zip_ref.extractall('data/ISBI14/')

# remove compressed files after extraction
os.remove("Train45Test90.zip")
os.remove("Dataset.zip")

# locations of images and masks
trainset_filepath = 'data/ISBI14/isbi_train.mat'
testset90_filepath = 'data/ISBI14/isbi_test90.mat'
testset810_filepath = 'data/ISBI14/Dataset/Synthetic/testset.mat'
trainset_GT_filepath = 'data/ISBI14/isbi_train_GT.mat'
testset90_GT_filepath = 'data/ISBI14/isbi_test90_GT.mat'
testset810_GT_filepath = 'data/ISBI14/Dataset/Synthetic/testset_GT.mat'

# the data will be stored in this file
export_filepath = 'data/ISBI14_images.hdf5'
compression = 'gzip'

f = h5py.File(export_filepath, 'w')
trainset_labels_group = f.create_group("trainset_labels")
testset90_labels_group = f.create_group("testset90_labels")
testset810_labels_group = f.create_group("testset810_labels")

# trainset images
file = loadmat(trainset_filepath)
num_images = file['ISBI_Train'].shape[0]
height = file['ISBI_Train'][0][0].shape[0]
width = file['ISBI_Train'][0][0].shape[1]

trainset_images = np.empty((num_images, height, width), dtype='float32')
trainset_images = np.expand_dims(trainset_images, 1)
for i in range(num_images):
    trainset_images[i, 0] = file['ISBI_Train'][i][0]

f.create_dataset('trainset', data=trainset_images, compression=compression)

# trainset labels
file = loadmat(trainset_GT_filepath)

trainset_labels = []
for i in range(num_images):
    num_objects = file["train_Cytoplasm"][i][0].shape[0]
    labels = np.empty((num_objects, height, width), dtype='bool')
    for j in range(num_objects):
        labels[j] = file['train_Cytoplasm'][i][0][j][0]
    trainset_labels.append(labels)

for i in range(len(trainset_labels)):
    trainset_labels_group.create_dataset(str(i), data=trainset_labels[i], compression=compression)

# testset90 images
file = loadmat(testset90_filepath)
num_images = file["ISBI_Test90"].shape[0]

testset90_images = np.empty((num_images, height, width), dtype='float32')
testset90_images = np.expand_dims(testset90_images, 1)
for i in range(num_images):
    testset90_images[i, 0] = file['ISBI_Test90'][i][0]

f.create_dataset('testset90', data=testset90_images, compression=compression)

# testset90 labels
file = loadmat(testset90_GT_filepath)

testset90_labels = []
for i in range(num_images):
    num_objects = file["test_Cytoplasm"][i][0].shape[0]
    labels = np.empty((num_objects, height, width), dtype='bool')
    for j in range(num_objects):
        labels[j] = file['test_Cytoplasm'][i][0][j][0]
    testset90_labels.append(labels)

for i in range(len(testset90_labels)):
    testset90_labels_group.create_dataset(str(i), data=testset90_labels[i], compression=compression)

# testset810 images
file = h5py.File(testset810_filepath, "r")
references = np.array(file[list(file.keys())[1]])[0]

num_images = references.shape[0]

testset810_images = np.empty((num_images - 90, 1, height, width), dtype='float32')

for i in range(90, num_images):
    testset810_images[i - 90, 0] = file[references[i]][()]

f.create_dataset('testset810', data=testset810_images, compression=compression)

# testset810 labels 
file = h5py.File(testset810_GT_filepath, "r")
references = np.array(file['test_Cytoplasm'])[0]

testset810_labels = []

for i in range(90, num_images):
    num_cells = file[references[i]].shape[1]
    labels = np.empty((num_cells, height, width), dtype='bool')
    for j in range(num_cells):
        labels[j] = file[file[references[i]][0][j]][()]

    testset810_labels.append(labels)

for i in range(len(testset810_labels)):
    testset810_labels_group.create_dataset(str(i), data=testset810_labels[i], compression=compression)


# real EDF images
images_list = sorted(os.listdir("data/ISBI14/Dataset/EDF/"))

num_images = len(images_list)
height = plt.imread(os.path.join("data/ISBI14/Dataset/EDF/", images_list[0])).shape[0]
width = plt.imread(os.path.join("data/ISBI14/Dataset/EDF/", images_list[0])).shape[1]

edf_images = np.empty((num_images, height, width), dtype='float32')

for i, x in enumerate(images_list):
    edf_images[i] = plt.imread(os.path.join("data/ISBI14/Dataset/EDF/", x))

edf_images = np.expand_dims(edf_images, 1)

f.create_dataset("edf", data=edf_images, compression=compression)