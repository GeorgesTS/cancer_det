from dataset import Data
from torch.utils.data import DataLoader
from torch import nn, optim
from get_data import get_data
import matplotlib.pyplot as plt 
from model import Classifier
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import numpy as np

X_train, X_test, y_train, y_test=get_data()



train_data = Data(X_train, y_train)
test_data = Data(X_test, y_test)
transforms=transforms.ToTensor()
validation_size=0.1


#Obtain training and validation indices
num_train=len(train_data)
index=list(range(num_train))
np.random.shuffle(index)
split=int(np.floor(validation_size*num_train))
train_idx,valid_idx=index[split:],index[:split]

batch_size=60

train_sampler=SubsetRandomSampler(train_idx)
valid_sampler=SubsetRandomSampler(valid_idx)

# dataloaders
try:
    trainloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    testloader = DataLoader(test_data, batch_size=batch_size)
    validloader=DataLoader(train_data,batch_size=batch_size, sampler=valid_sampler)
    print("Creating dataloaders")
    print("\n")

except:
    print("Error: Dataloaders could not be created")



model=Classifier()
criterion=nn.NLLLoss()
optimizer=optim.Adam(model.parameters(),lr=0.003)

epochs=1
train_losses=[]
test_losses=[]
valid_loss_min=np.Inf

for e in range(epochs):
    print("Currently in epoch:{}/{}".format(e,epochs))
    print("\n")

    running_loss=0.0
    valid_loss=0.0

    for images,labels in trainloader:
        print("In training")
        print("\n")

        optimizer.zero_grad()
        logps=model(images)
        loss=criterion(logps,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()

    #Evaluation mode
    with torch.no_grad():
        model.eval()
        instances=0
        valid_corr=0
        for images,labels in validloader:
            print("In validation")
            print("\n")

            
            
            output=model(images)
            loss=criterion(output,labels)
            _, predicted = torch.max(output.data, 1)
            instances+=labels.size(0)
            valid_loss+=loss.item()*images.size(0)
            valid_corr+=(predicted==labels).sum().item()

        epoch_acc = valid_corr / instances

        train_losses=(running_loss/len(trainloader.dataset))
        valid_loss=(valid_loss/len(testloader.dataset))

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \t Validation accuracy {:.6f}'.format(
            e+1, 
            train_losses,
            valid_loss,
            epoch_acc
            ))
    

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save ({
                'epochs': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, 'model.pt')
            valid_loss_min = valid_loss


PATH="/home/jorje/cancer/fully_connected/model.pt"
model = Classifier()

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
#.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epochs']
loss = checkpoint['loss']

model.eval()


correct_res=0
instances=0
class_0_corr=0
class_1_corr=0
instances_0=0
instances_1=0
for data in testloader:
    images, labels = data
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels.data).squeeze()
    instances=instances+len(labels)
       

    for i in range(len(labels)):
    
        if predicted[i]==labels[i]:
            correct_res=correct_res+1

            if (labels[i].item()==0):
                class_0_corr=class_0_corr+1

            if (labels[i].item()==1):
                class_1_corr=class_1_corr+1
            
        if labels[i].item()==0:
            instances_0+=1

        if labels[i].item()==1:
            instances_1+=1

class_0_acc=class_0_corr/instances_0
class_1_acc=class_1_corr/instances_1
total_acc=correct_res/instances

print("The total testing accuracy is {:.4f}".format(total_acc))
print("The accuracy for the first and second class are {:.2f} , {:.2f}".format(class_0_acc,class_1_acc))