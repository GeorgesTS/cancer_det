import torch
from model import Classifier
from get_data import get_data
from dataset import Data
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
PATH="/home/jorje/cancer/fully_connected/model.pt"
model = Classifier()

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
#.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epochs']
loss = checkpoint['loss']

model.eval()


X_train, X_test, y_train, y_test=get_data()



train_data = Data(X_train, y_train)
test_data = Data(X_test, y_test)
transforms=transforms.ToTensor()
validation_size=0.5


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
    trainloader = DataLoader(train_data,batch_size=batch_size, sampler=train_sampler)
    testloader = DataLoader(test_data,batch_size=batch_size)
    validloader=DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
    print("Creating dataloaders")
    print("\n")

except:
    print("Error: Dataloaders could not be created")



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