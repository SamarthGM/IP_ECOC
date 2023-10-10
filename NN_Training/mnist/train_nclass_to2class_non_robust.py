import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import grad
from torch.autograd import Variable
import pickle
from torch.utils.data.sampler import *
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import numpy as np
import sys
import random
import scipy.io as sio


########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
torch.manual_seed(2)

'''
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
		
def mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model




class NeuralNet(nn.Module):

	def __init__(self, input_size,hidden_size,output_size):
		super(NeuralNet, self).__init__()
		self.layer1 = nn.Linear(input_size, hidden_size)
		self.layer2 = nn.Linear(hidden_size, hidden_size)
		self.layer3 = nn.Linear(hidden_size, output_size)
		self.relu = nn.ReLU()

	def forward(self, x):
		output = self.layer1(x)
		output = self.relu(output)
		output = self.layer2(output)
		output = self.relu(output)
		output = self.layer2(output)
		return output
'''

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
    
    def forward(self, input):
        return self.func(input)

class Model(nn.Module):
    def __init__(self, i_c=1, n_c=2):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(i_c, 32, 5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=0)

        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=0)


        self.flatten = Expression(lambda tensor: tensor.view(tensor.shape[0], -1))
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=True)
        self.fc2 = nn.Linear(1024, n_c)


    def forward(self, x_i, _eval=False):

        if _eval:
            # switch to eval mode
            self.eval()
        else:
            self.train()
            
        x_o = self.conv1(x_i)
        x_o = torch.relu(x_o)
        x_o = self.pool1(x_o)

        x_o = self.conv2(x_o)
        x_o = torch.relu(x_o)
        x_o = self.pool2(x_o)

        x_o = self.flatten(x_o)

        x_o = torch.relu(self.fc1(x_o))

        self.train()

        return self.fc2(x_o)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

indices_dict = pickle.load(open("indices_dict", "rb"))

test_indices=[]
train_indices= []

bs = 128
transform = transforms.Compose(
    [transforms.ToTensor()#,
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    
class1 = [int(a) for a in sys.argv[1].split('_')] #[0, 3, 4]
class2=  [int(a) for a in sys.argv[2].split('_')] #[1, 5, 6]
print(class1,class2)

if class1 != sorted(class1):
	print('wrong class1 order:', class1)
	class1 = sorted(class1)

if class2 != sorted(class2):
	print('wrong class2 order:', class2)
	class2 = sorted(class2)
	
for c in class1 + class2:
	test_indices.extend(indices_dict['test'][c])
	train_indices.extend(indices_dict['train'][c])

print('# of test images:', len(test_indices))
print('# of train images:', len(train_indices))


trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, sampler=SubsetRandomSampler( train_indices), shuffle=False , num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform) 
testloader = torch.utils.data.DataLoader(testset, batch_size=bs,  sampler=SubsetRandomSampler( test_indices), shuffle=False , num_workers=2)

criterion = nn.CrossEntropyLoss().cuda()
epsilon = 0.3

inputSize = 28*28

outputSize = 2


#net = NeuralNet(inputSize,20, outputSize)
net = Model()

net.to(device) 
train_loss=[]
state ={}


epoch_schedule = [25 ]#, 100, 100]
lr_schedule    = [0.01]# , 0.001, 0.0001 ] 

for sch_id in range(len(lr_schedule)):
	no_epochs = epoch_schedule[sch_id]
	for epoch in range(no_epochs):  # loop over the dataset multiple times
		print('epoch:', epoch)
		net.train()
		running_loss = 0.0		
		for iter, data in enumerate(trainloader, 0):
			# get the inputs
			x, y = data
			#print("x.shape",x.shape)
			#print(y)
			#x=x.view(x.shape[0],-1)   # x.shape[0] is the batch size
			x=x.cuda()
			for l in range(len(y)):
				if y[l] in class1:
				   y[l] = 0
				elif y[l] in class2:
				   y[l] = 1 
				else:
				   print('~~~~~~~~~something is wrong~~~~~~~~~~~~~')


			x_var=torch.autograd.Variable(x.data,requires_grad = False).cuda()
			y_var= torch.autograd.Variable(y.data,requires_grad = False)

			y_var = y_var.cuda()

			train_acc=0

			#optimizer1 = optim.Adam(net.parameters(), lr=0.05)
			optimizer1 = optim.SGD(net.parameters(), lr=lr_schedule[sch_id])

			optimizer1.zero_grad()

			y_pred = net(x_var)

			_, predicted = torch.max(y_pred,1)
			loss = criterion(y_pred, y_var)	
			loss.backward()
			optimizer1.step()
			train_loss.append(loss.cpu().item())
		
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Now evaluating on TEST SET~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

		net.eval()
		test_loss = 0
		total_nat_acc = 0
		for iter, data in enumerate(testloader):
			x, y = data
			#x=x.view(x.shape[0],-1)   # x.shape[0] is the batch size
			x= x.cuda()
			for l in range(len(y)):
				if y[l] in class1:
				   y[l] = 0
				elif y[l] in class2:
				   y[l] = 1 
				else:
				   print('~~~~~~~~~something is wrong~~~~~~~~~~~~~')

			
			x_var=torch.autograd.Variable(x.data,requires_grad = False).cuda()
			y_var= torch.autograd.Variable(y.data,requires_grad = False)

			y_var = y_var.cuda()
			y_pred = net(x_var)
			_, predicted = torch.max(y_pred,1)
			total_nat_acc = total_nat_acc + predicted.eq(y_var).sum().item()
			
		print('classes:  ', class1,' ',class2,  'total_nat_acc:', total_nat_acc)
		




print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~final evaluation on TEST SET and saving~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

net.eval()
test_loss = 0
total_adv_acc = 0
total_nat_acc = 0
for iter, data in enumerate(testloader):
	x, y = data
	#x=x.view(x.shape[0],-1)   # x.shape[0] is the batch size
	x= x.cuda()
	for l in range(len(y)):
		if y[l] in class1:
		   y[l] = 0
		elif y[l] in class2:
		   y[l] = 1 
		else:
		   print('~~~~~~~~~something is wrong~~~~~~~~~~~~~')


	x_var=torch.autograd.Variable(x.data,requires_grad = True).cuda()
	y_var= torch.autograd.Variable(y.data,requires_grad = False)

	y_var = y_var.cuda()
	x_min = torch.autograd.Variable ( torch.max( x.data - epsilon, torch.zeros(x.shape).cuda() ) , requires_grad = False)
	x_max = torch.autograd.Variable ( torch.min( x.data + epsilon, torch.ones(x.shape).cuda() ) , requires_grad = False )


	params = {x_var}
	optimizer = optim.SGD(params, lr=0.1)

	tmp_adv_acc = 100000
	for i in range(20):
		optimizer.zero_grad()

		y_pred = net(x_var)

		_, predicted = torch.max(y_pred,1)
		adv_acc = predicted.eq(y_var).sum().item()
		
		if i == 0:
			total_nat_acc = total_nat_acc + predicted.eq(y_var).sum().item()
		
		
		#print('adv_acc:',adv_acc)
		if adv_acc < tmp_adv_acc:
			tmp_adv_acc = adv_acc

		loss = -1.0*criterion(y_pred,y_var)

		loss.backward()
		x_var.grad = x_var.grad/torch.abs(x_var.grad)
		optimizer.step()
		x_var.data = torch.max(torch.min(x_var.data, x_max.data), x_min.data)

	total_adv_acc = total_adv_acc + tmp_adv_acc


print('classes:  ', class1,' ',class2, ' total_adv_acc', total_adv_acc, 'total_nat_acc', total_nat_acc)



state = {
	    'net': net.state_dict(),
	    'adv_acc': total_adv_acc,
	    'nat_acc': total_nat_acc,
	    'no_test_images': len(test_indices)
	}
			


print('Saving....................')
class_id=   '_'.join([ str(a) for a in class1]) + '__' +'_'.join([ str(a) for a in class2])
n_class =str( len(class1) + len(class2))
torch.save(state,  'non_robust_'+class_id+'.ckpt' )

