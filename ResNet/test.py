import numpy as np
import torch
from torchsummary import summary
import torchvision.models as models
from random import randint
from matplotlib import pyplot as plt

from model import ResNet50
from load_MITDogs import load_dataset, create_breed_dataset
from tqdm import tqdm

lr = 1e-5
steps_per_epoch =  50
epochs = 50#200
classes = 2 #10
batch_size = 4


# device='cpu'
device = 'cuda:0'

session_name = "classes"+str(classes)+"_epoch"+str(epochs)+"_steps"+str(steps_per_epoch)+"_lr1e"+str(lr)[-1]+"_bs"+str(batch_size)
print(session_name)
#define training function
def train(model, epochs, trainx, trainy, steps_per_epoch , batch_size=2, results_dir="./results" ):
    max_sample = len(trainx)
    loss_log = []
    for i in tqdm(range(epochs)):
        epoch_loss = 0
        for s in range(steps_per_epoch):
            sample = randint(0, max_sample-1)
            c, x, y = np.shape(train_x[0,0])
            x_in, y_true = torch.tensor(np.zeros((batch_size, c, x, y))), torch.Tensor(np.zeros((batch_size)))
            for bs in range(batch_size):
                x_in[bs] = train_x[sample]
                y_true[bs] = torch.Tensor([train_y[sample]])
            x_in , y_true = x_in.float().cuda(), y_true.long().cuda()
            optimizer.zero_grad()

            y_out = model(x_in)
            loss = criterion(y_out, y_true)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().numpy()

        epoch_loss = epoch_loss/steps_per_epoch
        loss_log+=[epoch_loss]
        print((i, epoch_loss))
    return(model,loss_log)



#load premade resnet50 as sample

if __name__ == '__main__':
    resnet50 = models.resnet50()


    # load dataset
    print("Loading Dataset.")
    database = load_dataset(max_breeds= classes)
    print('Making breed classification dataset.')
    train_x, train_y, test_x, test_y =create_breed_dataset(database, classes)

    #load Resnet50 model from local folder
    model = ResNet50(classes=classes)


    #setup loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.cuda()
    print('Starting training session. Please Wait.')
    model, log = train(model, epochs, train_x, train_y, steps_per_epoch, batch_size)
    print('Training Complete')
    plt.plot(range(epochs), log)
    print(session_name)

    plt.savefig('results/'+session_name+'.png')
    plt.show()
