from dataset import Data
from torch.utils.data import DataLoader
from torch import nn, optim
from get_data import get_data
import matplotlib.pyplot as plt 
from model import Classifier
import torch

X_train, X_test, y_train, y_test=get_data()



train_data = Data(X_train, y_train)
test_data = Data(X_test, y_test)

# dataloaders
try:
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = DataLoader(test_data, batch_size=64, shuffle=False)
    print("Creating dataloaders")
    print("\n")

except:
    print("Error: Dataloaders could not be created")


model=Classifier()
criterion=nn.NLLLoss()
optimizer=optim.Adam(model.parameters(),lr=0.003)

epochs=50
train_losses=[]
test_losses=[]


for e in range(epochs):
    print("Currently in epoch:{}/{}".format(e+1,epochs))
    print("\n")

    running_loss=0

    for images,labels in trainloader:
        print("In training")
        print("\n")

        optimizer.zero_grad()
        logps=model(images)
        loss=criterion(logps,labels)
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
            test_loss+=criterion(log_ps,labels)

            ps=torch.exp(log_ps)
            top_p,top_class=ps.topk(1,dim=1)
            equals=top_class==labels.view(*top_class.shape)
            accuracy+=torch.mean(equals.type(torch.FloatTensor))

        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))

        print("Epoch number {} out of {}".format(e+1,epochs),
                "Training loss is {:.4f}".format(running_loss/len(trainloader)),
                "Test loss is {:.4f}".format(test_loss/len(testloader)),
                "Test accuracy is {:.4f}".format(accuracy/len(testloader)))


