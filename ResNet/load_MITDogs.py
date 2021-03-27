import numpy as np 
import torch 
import os
from PIL import Image



def load_dataset(dir = "MITDogs/Images/", dir_annotations = "MITDogs/Annotation/", max_breeds=120):
    database = {}
    i=0
    for folder in os.listdir(dir)[:max_breeds]:
        # print((folder.split('-', 1), i))
        id, breed = folder.split('-', 1)
        folder+='/'

        breed_dict={}
        breed_dict['breed_ID'] = str(id)
        breed_dict['breed'] = str(breed)
        breed_dict['image_count'] = len(os.listdir(dir+folder))

        n=0
        for filename in os.listdir(dir+folder):
            img_dict = {}
            try:
                img = prep_image(dir+folder+filename)
                img_dict['img_array'] = img
                img_dict['img_annotation'] = read_annotation(dir_annotations+folder+filename)
                img_dict['annotation_count'] = len(img_dict['img_annotation'])
                breed_dict[str(n)] = img_dict
                img_dict['img_title'] = filename
                n+=1
            except(ValueError):
                breed_dict['image_count'] -= 1
            

        database[str(i)] = breed_dict
        i+=1


    return(database)


def prep_image(filename):
    img = np.array(Image.open(filename))
    blank = np.zeros((1,500,500, 3))
    x,y, z = np.shape(img)
    blank[0,:x,:y]=img
    blank = np.moveaxis(blank, -1,1)
    return(blank)


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


def create_breed_dataset(dataset, number_of_breeds=120):
    train_x, train_y, test_x, test_y = [],[],[],[]
    for i in range(number_of_breeds):
        # print(i)
        image_count = dataset[str(i)]['image_count']
        div = image_count *0.7 //1
        for n in range(image_count):
            # print((i,n))
            # print(dataset.keys())
            # print(dataset[str(i)].keys())
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



    # train_y = train_y.flatten()
    # test_y = test_y.flatten()

    #onehot encode truths
    classes = int(np.max(train_y).item()+1)
    # for i in range(len(train_y)):
    #     # print(classes)
    #     blank = np.zeros(classes)
    #     # print(blank)
    #     # print(train_y[i])
    #     blank[int(train_y[i])] = 1
    #     train_y[i] = blank
    # for i in range(len(test_y)):
    #     blank = np.zeros(classes)
    #     blank[int(test_y[i])] = 1
    #     test_y[i] = blank
    
    # train_y = np.expand_dims(train_y,1)
    # test_y = np.expand_dims(test_y,1)

    train_x = torch.Tensor(train_x)#.to(device)
    train_y = torch.Tensor(train_y)#.to(device)
    test_x = torch.Tensor(test_x)#.to(device)
    test_y = torch.Tensor(test_y)#.to(device)

    return(train_x, train_y, test_x, test_y)


def create_annotation_dataset(dataset, number_of_breeds=120):
    pass


database = load_dataset(max_breeds= 10)
# print('make breed dataset')
train_x, train_y, test_x, test_y =create_breed_dataset(database, 2)
# print(np.shape(train_x))
# print(train_y)


# for i in train_x:
#     print(np.shape(i))  # therefore should use 500x500 model with padding to make all images same size
