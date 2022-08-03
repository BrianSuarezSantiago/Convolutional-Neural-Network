from keras.preprocessing.image import ImageDataGenerator
from skimage import io
from PIL import Image
import numpy as np
import os

datagen = ImageDataGenerator(rotation_range = 80,
                             shear_range = 0.2,
                             horizontal_flip = True,
                             brightness_range = (0.5, 1.5))

image_directory = r'./NaturalDataset/Training/airplane/'
dataset = []
SIZE = 100    # Size of the dataset images
my_images = os.listdir(image_directory)

for i, image_name in enumerate(my_images):
    if image_name.split('.')[1] == 'jpg':
        image = io.imread(image_directory + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
x = np.array(dataset)
i = 0

for batch in datagen.flow(x, batch_size = 4,
                          save_to_dir = r'../Augmentation/',
                          save_prefix = r'airplane',
                          save_format = 'jpg'):
    i += 1
    if i > 560:    # Number of images to be generated
        break
