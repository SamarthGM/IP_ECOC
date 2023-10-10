import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import grad
from torch.autograd import Variable
from resnet_2class import *
#from resnet_10class import *
import pickle
from torch.utils.data.sampler import *
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import numpy as np
import sys
import random
import time




torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

indices_dict = pickle.load(open("indices_dict", "rb"))

test_indices=[]
train_indices= [] 

bs = 64#128
transform = transforms.Compose(
    [transforms.ToTensor()#,
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
class1 = [int(a) for a in sys.argv[1].split('_')] #[0, 3, 4]
class2=  [int(a) for a in sys.argv[2].split('_')] #[1, 5, 6]


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


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, sampler=SubsetRandomSampler( train_indices), shuffle=False , num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform) 
testloader = torch.utils.data.DataLoader(testset, batch_size=bs,  sampler=SubsetRandomSampler( test_indices), shuffle=False , num_workers=2)

criterion = nn.CrossEntropyLoss().cuda()



net = ResNet18().to(device) 
#net = ResNet34().to(device) 
train_loss=[]

net.train()

l_rate = 0.001
for epoch in range(20):  # loop over the dataset multiple times
	net.train()
	print('epoch:', epoch)
	running_loss = 0.0

	if epoch >9:
		l_rate = 0.0005
	t1 =time.time()
	for iter, data in enumerate(trainloader, 0):
		# get the inputs
		x, y = data
		x=x.cuda()
		for l in range(len(y)):
			if y[l] in class1:
			   y[l] = 0
			elif y[l] in class2:
			   y[l] = 1 
			else:
			   print('~~~~~~~~~something is wrong~~~~~~~~~~~~~')


		x_var=torch.autograd.Variable(255*x.data,requires_grad = False).cuda()  # Note that the requires_grad has been set to False!!
		y_var= torch.autograd.Variable(y.data,requires_grad = False)

		y_var = y_var.cuda()

		train_acc=0
		#net.train()

		#optimizer1 = optim.SGD(net.parameters(), lr=0.01)
		optimizer1 = optim.Adam(net.parameters(), lr=l_rate)		


		optimizer1.zero_grad()

		y_pred = net(x_var)

		_, predicted = torch.max(y_pred,1)
		nat_acc = predicted.eq(y_var).sum().item()
		#print('_____train nat acc_______:',nat_acc)
		loss = criterion(y_pred, y_var)
		loss.backward()
		optimizer1.step()
		train_loss.append(loss.cpu().item())

	print("time taken for epcoh ", epoch, " = ", time.time() -t1 )
	net.eval()
	nat_acc =0 
	test_loss =0
	for iter, data in enumerate(testloader, 0):
		# get the inputs
		x, y = data
		x=x.cuda()
		for l in range(len(y)):
			if y[l] in class1:
			   y[l] = 0
			elif y[l] in class2:
			   y[l] = 1 
			else:
			   print('~~~~~~~~~something is wrong~~~~~~~~~~~~~')


		x_var=torch.autograd.Variable(255*x.data,requires_grad = False).cuda()  # Note that the requires_grad has been set to False!!
		y_var= torch.autograd.Variable(y.data,requires_grad = False)

		y_var = y_var.cuda()
		y_pred = net(x_var)

		_, predicted = torch.max(y_pred,1)
		nat_acc = nat_acc + predicted.eq(y_var).sum().item()
		loss = criterion(y_pred, y_var)
		test_loss = test_loss + loss.item()



	print('natural accuracy on test set:', nat_acc)
	print('test loss', test_loss)





#--------------------- Evalauting  Natural Accuracy---------------------
print("--------------------- Evalauting  Natural Accuracy---------------------")
net.eval()
nat_acc =0 
test_loss =0
for iter, data in enumerate(testloader, 0):
	# get the inputs
	x, y = data
	x=x.cuda()
	for l in range(len(y)):
		if y[l] in class1:
		   y[l] = 0
		elif y[l] in class2:
		   y[l] = 1 
		else:
		   print('~~~~~~~~~something is wrong~~~~~~~~~~~~~')


	x_var=torch.autograd.Variable(255*x.data,requires_grad = False).cuda()  # Note that the requires_grad has been set to False!!
	y_var= torch.autograd.Variable(y.data,requires_grad = False)

	y_var = y_var.cuda()
	y_pred = net(x_var)

	_, predicted = torch.max(y_pred,1)
	nat_acc = nat_acc + predicted.eq(y_var).sum().item()
	loss = criterion(y_pred, y_var)
	test_loss = test_loss + loss.item()


print('natural accuracy on test set:', nat_acc)
print('test loss', test_loss)


'''

#---------------- Evaluating Adversarial Accuracy in case we need to report it--------------------------------
epsilon = 8
print("---------------- Evaluating Adversarial Accuracy in case we need to report it-------")
net.eval()
total_adv_acc = 0
for iter, data in enumerate(testloader):
	x, y = data
	x= x.cuda()
	for l in range(len(y)):
		if y[l] in class1:
		   y[l] = 0
		elif y[l] in class2:
		   y[l] = 1 
		else:
		   print('~~~~~~~~~something is wrong~~~~~~~~~~~~~')


	x_var=torch.autograd.Variable(255*x.data,requires_grad = True).cuda()
	y_var= torch.autograd.Variable(y.data,requires_grad = False)

	y_var = y_var.cuda()
	x_min = torch.autograd.Variable ( torch.max( 255*x.data - epsilon, torch.zeros(x.shape).cuda() ) , requires_grad = False)
	x_max = torch.autograd.Variable ( torch.min( 255*x.data + epsilon, 255*torch.ones(x.shape).cuda() ) , requires_grad = False )


	params = {x_var}
	optimizer = optim.SGD(params, lr=1.0)
	#print('-------------------------------------------------')
	tmp_adv_acc = 1000000
	for i in range(10):
		optimizer.zero_grad()

		y_pred = net(x_var)

		_, predicted = torch.max(y_pred,1)
		adv_acc = predicted.eq(y_var).sum().item()

		#print('adv_acc:',adv_acc)
		if adv_acc < tmp_adv_acc:
			tmp_adv_acc = adv_acc

		loss = -1.0*criterion(y_pred,y_var)

		loss.backward()
		x_var.grad = x_var.grad/torch.abs(x_var.grad)
		optimizer.step()
		x_var.data = torch.max(torch.min(x_var.data, x_max.data), x_min.data)

	total_adv_acc = total_adv_acc + tmp_adv_acc


print('------Adversarial Acc:----',total_adv_acc)
'''

state = {
    'net': net.state_dict(),
    #'adv_acc': total_adv_acc*1.0/len(test_indices),
    'nat_acc' : nat_acc*1.0/len(test_indices)	
}
		
#state['loss'] = train_loss

print('Saving....................')
class_id=   '_'.join([ str(a) for a in class1]) + '__' +'_'.join([ str(a) for a in class2])
n_class =str( len(class1) + len(class2))
torch.save(state, 'resnet18_non_robust_'+class_id+'.ckpt')
