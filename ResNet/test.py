import numpy as np
import torch
from torchsummary import summary
import torchvision.models as models 
from random import randint
from matplotlib import pyplot as plt

from model import ResNet50
from load_MITDogs import load_dataset, create_breed_dataset
from tqdm import tqdm

lr = 1e-4
steps_per_epoch =  50
epochs = 20
classes = 5 #10


# device='cpu'
device = 'cuda:0'


#define training function
def train(model, epochs, trainx, trainy, steps_per_epoch , results_dir="./results" ):
    max_sample = len(trainx)
    loss_log = []
    for i in tqdm(range(epochs)):
        epoch_loss = 0
        for s in range(steps_per_epoch):
            sample = randint(0, max_sample-1)
            x_in,y_true = train_x[sample], torch.Tensor([train_y[sample]]).long()
            x_in , y_true = x_in.cuda(), y_true.cuda() # x_in.to(device), y_true.to(device)
            optimizer.zero_grad()

            # print(next(model.parameters()).device)
            # print(x_in.device)
            # print(y_true.device)

            y_out = model(x_in)
            # print('loss sizes')
            # y_out = y_out.flatten()
            # print(y_out.size())
            # print(y_true.size())
            # print(y_out)
            # print(y_true)
            loss = criterion(y_out, y_true)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().numpy()

        epoch_loss = epoch_loss/steps_per_epoch
        loss_log+=[epoch_loss]
    return(model,loss_log)



#load premade resnet50 as sample
resnet50 = models.resnet50()


# load dataset

database = load_dataset(max_breeds= classes)
print('make breed dataset')
train_x, train_y, test_x, test_y =create_breed_dataset(database, classes)
print(train_y.size())

#load Resnet50 model from local folder
model = ResNet50(classes=classes)


#setup loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print(model)
for layer in model.parameters():
    print(layer.device)

print('switch to cuda')

model = model.cuda()

for layer in model.parameters():
    print(layer.device)
# print(3/0)

model = model.cuda()
model, log = train(model, epochs, train_x, train_y, steps_per_epoch)

plt.plot(range(epochs), log)
plt.show()


