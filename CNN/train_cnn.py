import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from cnn import CNN
from dataset import CatsAndDogsDataset
from tqdm import tqdm
from torch import nn, optim



transform = transforms.Compose(
        [
            transforms.ToTensor()
          
        ]
    )



num_epochs = 10
learning_rate = 0.00001
train_CNN = False
batch_size = 1
shuffle = True
pin_memory = True
num_workers = 1



dataset = CatsAndDogsDataset("train","train_csv.csv",transform=transform)
train_set, validation_set = torch.utils.data.random_split(dataset,[10, 6])

trainloader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
testloader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)


model=CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochs=500
train_losses=[]
test_losses=[]


for e in range(epochs):
    print("Currently in epoch:{}/{}".format(e+1,epochs))
    print("\n")

    running_loss=0
    print("In training")
    for images,labels in trainloader:
       
        print("\n")

        optimizer.zero_grad()
       

       
        logps=model(images)
        loss=criterion(logps,labels.long())
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()

    else:
        print("In validation")
        print("\n")

        test_loss=0
        accuracy=0

        with torch.no_grad():
            log_ps=model(images)
            test_loss+=criterion(log_ps,labels.long())

            ps=torch.exp(log_ps)
            top_p,top_class=ps.topk(1,dim=1)
            equals=top_class==labels.view(*top_class.shape)
            accuracy+=torch.mean(equals.type(torch.FloatTensor))

        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))

        print("Epoch number {} out of {}".format(e+1,epochs),"\n",
                "Training loss is {:.4f}".format(running_loss/len(trainloader)),"\n",
                "Test loss is {:.4f}".format(test_loss/len(testloader)),"\n",
                "Test accuracy is {:.4f}".format(accuracy/len(testloader)))


