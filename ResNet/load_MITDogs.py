import numpy as np
import torch
import os
from PIL import Image
from tqdm import tqdm

#load maximum number of breeds to be considered, to be called once
def load_dataset(dir = "MITDogs/Images/", dir_annotations = "MITDogs/Annotation/", max_breeds=120):
    database = {}
    i=0
    #check through each breed folder, up to the maximum number of breeds chosen
    for folder in tqdm(os.listdir(dir)[:max_breeds]):
        ID, breed = folder.split('-', 1)
        folder+='/'
        #create a dictionary for each breeds information
        breed_dict={}
        breed_dict['breed_ID'] = str(ID)
        breed_dict['breed'] = str(breed)
        breed_dict['image_count'] = len(os.listdir(dir+folder))

        n=0
        #check each image in the chosen breed folder
        for filename in os.listdir(dir+folder):
            img_dict = {}
            #try and load image and data for sample
            try:
                img = prep_image(dir+folder+filename)
                img_dict['img_array'] = img
                img_dict['img_annotation'] = read_annotation(dir_annotations+folder+filename)
                img_dict['annotation_count'] = len(img_dict['img_annotation'])
                #store image data in breed dictionary
                breed_dict[str(n)] = img_dict
                img_dict['img_title'] = filename
                n+=1
            #fix the number of samples to accomodate for failed loading
            except(ValueError):
                breed_dict['image_count'] -= 1

        #combine all breeds into a single database
        database[str(i)] = breed_dict
        i+=1


    return(database)

#load then perform transformations to allow images to be handled by model without error
def prep_image(filename):
    img = np.array(Image.open(filename))
    blank = np.zeros((1,500,500, 3))
    x,y, z = np.shape(img)
    blank[0,:x,:y]=img
    blank = np.moveaxis(blank, -1,1)
    return(blank)

#parse annotation data for each images boundary box
def read_annotation(filename):
    filename = filename.split('.')[0]
    annotations = open(filename, "rt")
    annotations = annotations.read()
    annotations = annotations.split('<bndbox>')[1:]
    parsed = []
    for annot in annotations:
        annot = annot.split('</bndbox>')[0]
        xmin = annot.split('<xmin>')[1].split('</xmin>')[0]
        ymin = annot.split('<ymin>')[1].split('</ymin>')[0]
        xmax = annot.split('<xmax>')[1].split('</xmax>')[0]
        ymax = annot.split('<ymax>')[1].split('</ymax>')[0]
        parsed +=[(xmin, ymin, xmax, ymax)]

    return(parsed)

#turn full dataset into deep learning ready data (tensors, normalized, and split for train/test), call anytime number of breeds being considered changes
def create_breed_dataset(dataset, number_of_breeds=120):
    train_x, train_y, test_x, test_y = [],[],[],[]
    #run through each breed being considered
    for i in tqdm(range(number_of_breeds)):
        image_count = dataset[str(i)]['image_count']
        #set division point between training and test samples
        div = image_count *0.7 //1
        for n in range(image_count):
            if n < div:
                train_x += [dataset[str(i)][str(n)]['img_array']]
                train_y += [i]
            if n >= div:
                test_x += [dataset[str(i)][str(n)]['img_array']]
                test_y += [i]
    train_x, train_y, test_x, test_y =np.array(train_x), train_y, np.array(test_x), test_y
    #normalize image data
    train_x = train_x.astype('float64')*1/np.max(train_x)
    test_x = test_x.astype('float64')*1/np.max(test_x)
    #convert from numpy to tensors
    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y)
    test_x = torch.Tensor(test_x)
    test_y = torch.Tensor(test_y)

    return(train_x, train_y, test_x, test_y)



if __name__ == '__main__':
    print('Loading dataset')
    database = load_dataset(max_breeds= 10)
    print("creating trainging/test data")
    train_x, train_y, test_x, test_y =create_breed_dataset(database, 2)
