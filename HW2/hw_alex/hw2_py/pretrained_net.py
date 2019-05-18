import torch
import torchvision
import torchvision.models as models
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch.nn as nn

import requests


import PIL
from PIL import Image

import matplotlib.pyplot as plt

import numpy as np
from numpy import asarray


from matplotlib import image
from matplotlib import pyplot

vgg16 = models.vgg16(pretrained=True)

# normalization of the data as adviced in pytorch tutorials
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


crop_transform = transforms.CenterCrop((224, 224))



def create_batch_image(addresses_list, specific_transforms):

    pics_mb = list()

    for address in addresses_list:
        pic = Image.open(address)
        resized_pic = pic.resize([224, 224])

        pic_data = specific_transforms(resized_pic)

        # reshape to create pseudo minibatch
        pic_data_mb = pic_data.reshape((1, pic_data.shape[0], pic_data.shape[1], pic_data.shape[2]))

        pics_mb.append(pic_data_mb)

    return pics_mb

def display_transformed_img(addresses_list, specific_transforms):

    for address in addresses_list:
        pic = Image.open(address)
        resized_pic = pic.resize([224, 224])

        pic_data = specific_transforms(resized_pic)


        plt.imshow(pic_data.permute(1, 2, 0))
        plt.show()


def print_predictions(minibatch_list):
    softmax_layer = nn.Softmax(dim=1)

    for sample_idx, sample in enumerate(minibatch_list):
        # print(bird.shape)
        # we have to normalize the data
        output = vgg16(sample)
        max_index = torch.argmax(output).item()
        # max_index = output.data.numpy().argmax()

        softmax_vals = softmax_layer(output)

        # print('Sum of output values: ' + str(torch.sum(softmax_vals).item()))

        print('For sample #' + str(sample_idx) + ' the precicted output is ' +
              str(labels[max_index]) + '\n\twith certainty of : '+ str('{0:.2f}'.format(torch.max(softmax_vals).item()*100)) + ' [%] ')




# Code from github
# Class labels used when training VGG as json, courtesy of the 'Example code' link above.
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
# Let's get our class labels.
response = requests.get(LABELS_URL)  # Make an HTTP GET request and store the response.
labels = {int(key): value for key, value in response.json().items()}





# 1. The birds example

address_list = list()
address_list.append('./CNN Images/birds/bird_0.jpg')
address_list.append('./CNN Images/birds/bird_1.jpg')

# define transformations
specific_transforms = transforms.Compose([transforms.ToTensor(), normalize])

birds_mb = create_batch_image(address_list, specific_transforms)

print('For the input birds images:')
print_predictions(birds_mb)




# Using the internet image


address_list = list()
address_list.append('./data/internet_images/bear.jpg')

# define transformations
specific_transforms = transforms.Compose([transforms.ToTensor(), normalize])

bear_mb = create_batch_image(address_list, specific_transforms)
print('For a random bear image from the internet:')
print_predictions(bear_mb)



###############
# 1. Using a geometric transformation - SHEAR + ROTATION
affine_tr = transforms.RandomAffine(degrees=50, shear=45)

address_list = list()
address_list.append('./data/internet_images/bear.jpg')


# display without the normalization
specific_transforms = transforms.Compose([affine_tr,
                                          transforms.ToTensor()])

display_transformed_img(address_list, specific_transforms)

# get output with normalization

specific_transforms = transforms.Compose([affine_tr,
                                          transforms.ToTensor(),
                                          normalize])

bear_mb = create_batch_image(address_list, specific_transforms)
print ('For an image with geometric transformation:')
print_predictions(bear_mb)


###################
# 2. Using a color transformation - we ue the color jitter transformation
jitter_tr = transforms.ColorJitter(hue=(0.5, 0.5))

address_list = list()
address_list.append('./data/internet_images/bear.jpg')


# display without the normalization
specific_transforms = transforms.Compose([jitter_tr,
                                          transforms.ToTensor()])

display_transformed_img(address_list, specific_transforms)
# get output with normalization

specific_transforms = transforms.Compose([jitter_tr,
                                          transforms.ToTensor(),
                                          normalize])

bear_mb = create_batch_image(address_list, specific_transforms)
print ('For an image with color transformation ( hue jitter):')
print_predictions(bear_mb)

##################
# 3. Using a filter on our image - let us choose the

# defining a small filter of size 3*3, with some random parameters, imitating noise
# without brightness reduction, i.e. all values sum to 1



def create_noise_filt_2d(size, mean, std):
    custom_filter = torch.randn(size, size)  # mean is 0, var is 1
    custom_filter = custom_filter*std + mean

    # make the central pixel fill till 1
    custom_filter[1][1] = torch.tensor(0)
    custom_filter[1][1] = 1 - torch.sum(custom_filter)

    return custom_filter

def create_noise_filt_3d(size, depth, mean, std):
    custom_filter = torch.randn(size, size, depth)  # mean is 0, var is 1
    custom_filter = custom_filter*std + mean

    for depth in range(3):

        # make the central pixel fill till 1
        custom_filter[depth][1][1] = torch.tensor(0)
        custom_filter[depth][1][1] = 1 - torch.sum(custom_filter[depth][:][:])

    return custom_filter

def apply_filter(address_list, noise_filter):

    filtered_images = list()

    for address in address_list:
        pic = Image.open(address)
        pic_np = np.array(pic)
        pic_torch = torch.from_numpy(pic_np).float()

        # need to reshape into tensor of BHWC
        # totensor = transforms.ToTensor()
        # pic = totensor(pic)

        noise_filter = noise_filter.reshape((1, noise_filter.shape[0], noise_filter.shape[1], noise_filter.shape[2]))

        pic_data_mb = pic_torch.reshape((1, pic_torch.shape[2], pic_torch.shape[0], pic_torch.shape[1]))

        output = F.conv2d(pic_data_mb, noise_filter)

        filtered_images.append(output)

    return filtered_images


address_list = list()
address_list.append('./data/internet_images/bear.jpg')


noise_filter = create_noise_filt_3d(size=3, depth=3,  mean=0.02, std=0.005) # central will be around 0.85

filtered_images = apply_filter(address_list, noise_filter)




