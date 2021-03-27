import numpy as np
import torch
from torchsummary import summary
from random import randint
from matplotlib import pyplot as plt

from model import ResNet50
from load_MITDogs import load_dataset, create_breed_dataset
from tqdm import tqdm

# Set Parameters for Session
lr = 1e-5                   #learning rate - float effects training sensitivity
steps_per_epoch =  50       #Number of times model sees data and is updated per epoch
epochs = 600                #Number of epochs model will be trained for (epochs*steps_per_epoch), Loss results are averaged per epoch
classes = 10                #Number of breeds being considered
batch_size = 6              #Number of images seen by model per step


# Select device, either 'cpu' or 'cuda:(gpu-id)'
device = 'cuda:0'

#name for training session, records details
session_name = "classes"+str(classes)+"_epoch"+str(epochs)+"_steps"+str(steps_per_epoch)+"_lr1e"+str(lr)+"_bs"+str(batch_size)
print(session_name)


#train model to classify images
def train(model, epochs, trainx, trainy, steps_per_epoch , batch_size=2, results_dir="./results" ):
    max_sample = len(trainx)
    loss_log = []
    #Model is trained for i-many epochs
    for i in tqdm(range(epochs)):
        epoch_loss = 0
        #Model makes s-many training steps per epoch
        for s in range(steps_per_epoch):
            c, x, y = np.shape(train_x[0,0])
            #Create new tensors to store
            x_in, y_true = torch.tensor(np.zeros((batch_size, c, x, y))), torch.Tensor(np.zeros((batch_size)))
            for bs in range(batch_size):
                #pick a random training sample and put the correlated image and class_id into the tensors for usage
                sample = randint(0, max_sample-1)
                x_in[bs] = train_x[sample]
                y_true[bs] = torch.Tensor([train_y[sample]])
            #Set datatype and move tensors to GPU
            x_in , y_true = x_in.float().cuda(), y_true.long().cuda()
            optimizer.zero_grad()                        #Reset gradient on optimizer
            y_out = model(x_in)                          #Apply model to
            loss = criterion(y_out, y_true)              #
            loss.backward()                              #
            optimizer.step()                             #
            epoch_loss += loss.detach().cpu().numpy()    #

        epoch_loss = epoch_loss/steps_per_epoch
        loss_log+=[epoch_loss]
        print((i, epoch_loss))
    return(model,loss_log)

# Predict model on data without training, returns
def test_model(model, test_x, test_y):
    true_count = np.zeros((classes))
    total_count = np.zeros((classes))
    for i in tqdm(range(len(test_x))):
        guess = np.argmax(model(test_x[i].cuda()).detach().cpu().numpy())
        truth = int(test_y[i].numpy())
        if guess == truth:
            true_count[truth]+=1
        total_count[truth]+=1
    return(true_count, total_count)


if __name__ == '__main__':


    # load dataset
    print("Loading Dataset.")
    database = load_dataset(max_breeds= classes)
    print('Making breed classification dataset.')
    train_x, train_y, test_x, test_y =create_breed_dataset(database, classes)

    #load Resnet50 model from local file
    model = ResNet50(classes=classes)


    #setup loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #Convert model to cuda then train
    model = model.cuda()
    print('Starting training session. Please Wait.')
    model, log = train(model, epochs, train_x, train_y, steps_per_epoch, batch_size)
    print('Training Complete: '+session_name)

    #log average loss vs epoch on training data
    plt.plot(range(epochs), log)
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
    plt.savefig('results/'+session_name+'.png')
    plt.show()

    #Apply model to test data, then print out accuracy by breed
    print("Testing model Performance")
    true_count, total_count=test_model(model, test_x, test_y)
    print('results')
    print(true_count)
    print(total_count)
    accuracy = true_count*100/total_count
    print(accuracy)
    breeds=[]
    for i in range(classes):
        breeds+=[database[str(i)]['breed']]
    print('Accuracy by breed:')
    for i in range(len(breeds)):
        print(str(accuracy[i])+"% -"+str(breeds[i]))
