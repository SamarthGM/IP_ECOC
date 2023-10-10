import multiprocessing
import time
import pickle
import sys
import numpy as np
from subprocess import PIPE, Popen

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import grad
from torch.autograd import Variable
#from resnet_2class import *
import pickle
from torch.utils.data.sampler import *
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def print_code_book(code_book):
	for i in code_book:
		print(i)


def code_book(n_class):
	n_classes = n_class
	n_bits =pow(2,n_classes-1) -1
	book=[]
	for c in range(1,n_classes+1):
		if c == 1:
			code = [1 for j in range(n_bits)]
			#print(c-1,code)
			book.append(code)	
		else:
			zeros =[ 0 for j in range(int(pow(2,n_classes-c)))]
			ones  =[ 1 for j in range(int(pow(2,n_classes-c)))]
			code =[]
		
			for j in range( int(n_bits/pow(2,n_classes-c))):
				code.extend(zeros)
				code.extend(ones)
				   
			code=code[0:n_bits]
			#print(c-1, code)
			book.append(code)
	return book

def code_book_to_combo_ternary(code_book):
	n_combo = len(code_book[0])
	n_classes = len(code_book)
	classes = [ i for i in range(n_classes)] # chnage this if want to use diff set of classes
	combo=[]
	for j in range(n_combo):
		class1=[]
		class2=[]
		for i in range(n_classes):
			if(code_book[i][j] == 1):
				class1.append(i) 	
			elif(code_book[i][j] == -1):
				class2.append(i)
		class1 = "_".join(str(k) for k in class1)	
		class2 = "_".join(str(k) for k in class2)			
		#print(class1,class2)
		combo.append([class1, class2])
	return combo


def writeHeader(eps_):

	print("import torch")
	print("import torchvision")
	print("import torchvision.transforms as transforms")
	print("import torch.optim as optim")
	print("from torch.autograd import grad")
	print("from torch.autograd import Variable")
	print("#from resnet_2class import *")
	print("import pickle")
	print("from torch.utils.data.sampler import *")
	print("import torch.nn as nn")
	print("import torch.nn.functional as F")
	print("import matplotlib.pyplot as plt")
	print("import numpy as np")
	print("import sys")
	#print("sys.path.insert(0, '/home/samarth/nsga2-gnuplot-v1.1.6_m_instances')")
	print("import random")
	#print("import nsga2r_mi")
	print("\n\n")



	print("class Expression(nn.Module):")
	print("    def __init__(self, func):")
	print("        super(Expression, self).__init__()")
	print("        self.func = func")
	    
	print("    def forward(self, input):")
	print("        return self.func(input)")

	print("\n\n")

	print("class Model(nn.Module):")
	print("    def __init__(self, i_c=1, n_c=2):")
	print("        super(Model, self).__init__()")

	print("        self.conv1 = nn.Conv2d(i_c, 32, 5, stride=1, padding=2, bias=True)")
	print("        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=0)")

	print("        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2, bias=True)")
	print("        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=0)")


	print("        self.flatten = Expression(lambda tensor: tensor.view(tensor.shape[0], -1))")
	print("        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=True)")
	print("        self.fc2 = nn.Linear(1024, n_c)")


	print("    def forward(self, x_i, _eval=False):")

	print("        if _eval:")
		    # switch to eval mode
	print("            self.eval()")
	print("        else:")
	print("            self.train()")
	print("	    		     ")
	print("        x_o = self.conv1(x_i)")
	print("        x_o = torch.relu(x_o)")
	print("        x_o = self.pool1(x_o)")

	print("        x_o = self.conv2(x_o)")
	print("        x_o = torch.relu(x_o)")
	print("        x_o = self.pool2(x_o)")

	print("        x_o = self.flatten(x_o)")

	print("        x_o = torch.relu(self.fc1(x_o))")

	print("        self.train()")

	print("        return self.fc2(x_o)")

	print("\n\n")


	print("device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')")

	print("indices_dict = pickle.load(open(\"indices_dict\", \"rb\"))")

	print("test_indices=[]")
	print("test_indices.extend(indices_dict['test'][0]) ")
	print("test_indices.extend(indices_dict['test'][1]) ")
	print("test_indices.extend(indices_dict['test'][2]) ")
	
	print("test_indices.extend(indices_dict['test'][3]) ")
	print("test_indices.extend(indices_dict['test'][4]) ")
	print("test_indices.extend(indices_dict['test'][5]) ")
	print("test_indices.extend(indices_dict['test'][6]) ")
	print("test_indices.extend(indices_dict['test'][7]) ")
	print("test_indices.extend(indices_dict['test'][8]) ")
	print("test_indices.extend(indices_dict['test'][9]) ")
	print("print(\"# of test images:\", len(test_indices)) ")
	

	print("\n\n")

	print("transform = transforms.Compose([transforms.ToTensor()#, #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) \n\t\t])")
	print("testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform) ")
	print("testloader0_1_2_3_4_5_6_7_8_9 = torch.utils.data.DataLoader(testset, batch_size=256,    sampler=SubsetRandomSampler( indices_dict['test'][0] +indices_dict['test'][1] + indices_dict['test'][2] + indices_dict['test'][3] + indices_dict['test'][4] + indices_dict['test'][5]+indices_dict['test'][6]+indices_dict['test'][7]+indices_dict['test'][8]+indices_dict['test'][9]), shuffle=False, num_workers=2) ")
	print("\n\n")

	print("criterion = nn.CrossEntropyLoss().cuda()")
	print("epsilon = "+str(eps_))
	print("print(\"epsilon = \",epsilon)")



#if __name__ == '__main__':
def write_code(codebook_name, eps_):
	writeHeader(eps_)
	code_book_name = codebook_name #"sample_codebook_K_10_L_20.npy"
	code_book = np.load(code_book_name)
	#code_book[ code_book == -1 ] = 0	
	code_book = (code_book).tolist()

	#print('#---------------------------Code_book--------------------------------')
	#print_code_book(code_book)
	
	code_book_mod = np.load(code_book_name)   #code_book_modified
	code_book_mod[code_book_mod == 0 ] = -1.0			### !!!!! need to check this 
	code_book_mod[code_book_mod == 1 ] = 1.0			### !!!!! need to check this	
	code_book_mod = code_book_mod.tolist()
	#print('#---------------------------Code_book modified-----------------------')
	#print_code_book(code_book_mod)
	
	
	print('#---------------------------------------')
	all_combo = code_book_to_combo_ternary( code_book)  

	print('#------------------Code to load trained classifiers-------------------')
	for combo in all_combo:
		
		net_name = 'non_robust_'+combo[0]+'__'+ combo[1] +'.ckpt'
		net_var = 'net_'+combo[0]+'__'+combo[1] 
		code = 'saved_dict = torch.load(\''+net_name+ '\')'+'\n' + net_var +'  = Model().to(device)\n'
		code = code +   net_var+'.load_state_dict( saved_dict[\'net\'] )\n'
		code = code + net_var +' = '+net_var+'.eval()\n'
		code = code + net_var +'.zero_grad()\n'	
		print(code)
	
	print("total_adv_acc=0")
	print("total_nat_acc=0")

	print("total_adv_vote_acc=0")
	print("total_nat_vote_acc=0")
	
	print('#------------------Code to compute classwise score/loss-------------------')
	print("for iter,data in enumerate(testloader0_1_2_3_4_5_6_7_8_9):\n\tx,y=data\n\tx=x.cuda()\n\tx_var=torch.autograd.Variable(x.data,requires_grad = True).cuda()\n\ty_var= torch.autograd.Variable(y.data,requires_grad = False)\n\ty_var = y_var.cuda()\n\tx_min = torch.autograd.Variable ( torch.max( x.data - epsilon, torch.zeros(x.shape).cuda() ) , requires_grad = False)\n\tx_max = torch.autograd.Variable ( torch.min( x.data + epsilon, torch.ones(x.shape).cuda() ) , requires_grad = False )")
	
	#forward pass	
	for combo in all_combo:
		net_var = 'net_'+combo[0]+'__'+combo[1] 
		#code = 'y'+ combo[0]+'__'+combo[1]+' = F.softmax('+net_var+'(x) , dim =0) # ensure that the dimension of softmax is correct'
		code = 'y'+ combo[0]+'__'+combo[1]+' = '+net_var+'(x_var)'
		print('\t'+code)

	print('\n')
	no_classes =10
	
	# y_hat	
	for k in range(no_classes):
		code = 'y'+str(k)+' = '
		for j,combo in enumerate(all_combo):
		    if str(k) in combo[0].split('_') or str(k) in combo[1].split('_'):
			loss_margin = 'y'+ combo[0]+'__'+combo[1]+'[:,'+   (str(0) if str(k) in combo[0].split('_') else str(1) )  +']'
			#loss_margin =   '(1 - code_book_mod['+str(k)+']['+str(j)+']*(y'+ combo[0]+'__'+combo[1]+'[:,'+   str(1)  +'] -0.5) )^2 +'  # with square loss
			
			if j == len(all_combo)-1:
				code = code +  loss_margin 			
			else :
				code = code +  loss_margin + ' + '	
		
		code = code + '\n'
		print('\t'+code)
		
					
	tmp = "y_pred = torch.stack( ("  
	for k in range(no_classes):
		if k == no_classes -1:
			tmp =tmp+ "y"+str(k)
		else:
			tmp =tmp+ "y"+str(k)+", "
	tmp =tmp+ "),1)"
	print('\t'+tmp)

	print("\n\n")
	print('\t'+"_, predicted = torch.max(y_pred,1)")
	print('\t'+"acc = predicted.eq(y_var).sum().item()")
	print('\t'+"total_nat_acc= total_nat_acc + acc")
	print("\n\n")	



	print("\tparams = {x_var}\n\toptimizer = optim.SGD(params, lr=0.01)")	

	for combo in all_combo:
		code = 'net_'+combo[0]+'__'+combo[1] +".eval()"
		print("\t"+code)
		
	print("\tadv_acc=1000\n\tvote_adv_acc =1000\n\tfor i in range(100):")
	print("\t\toptimizer.zero_grad()")	

	#forward pass
	for combo in all_combo:
		net_var = 'net_'+combo[0]+'__'+combo[1] 
		code = 'y'+ combo[0]+'__'+combo[1]+' = '+net_var+'(x_var)'
		print('\t\t'+code)



	# y_hat	
	for k in range(no_classes):
		code = 'y'+str(k)+' = '
		for j,combo in enumerate(all_combo):
			loss_margin = 'y'+ combo[0]+'__'+combo[1]+'[:,'+   (str(0) if str(k) in combo[0].split('_') else str(1) )  +']'
			#loss_margin =   '(1 - code_book_mod['+str(k)+']['+str(j)+']*(y'+ combo[0]+'__'+combo[1]+'[:,'+   str(1)  +'] -0.5) )^2 +'  # with square loss
			
			if j == len(all_combo)-1:
				code = code +  loss_margin 			
			else :
				code = code +  loss_margin + ' + '	
		
		code = code + '\n'
		print('\t\t'+code)


	tmp = "y_pred = torch.stack( ("  
	for k in range(no_classes):
		if k == no_classes -1:
			tmp =tmp+ "y"+str(k)
		else:
			tmp =tmp+ "y"+str(k)+", "
	tmp =tmp+ "),1)"
	print('\t\t'+tmp)


	print("\n\n\n\n")
	print("\t\t"+"_, predicted = torch.max(y_pred,1) ")
	print("\t\t"+"tmp_acc = predicted.eq(y_var).sum().item()")
	print("\t\t"+"if tmp_acc < adv_acc:")
	print("\t\t"+"\tadv_acc = tmp_acc")
	print("\t\t"+"#print('adv_acc:',tmp_acc)")
	print("\n\n")




	print("\t\t"+"loss = -1.0*criterion(y_pred,y_var)")
	print("\t\t"+"loss.backward()")
	print("\t\t"+"x_var.grad = x_var.grad/torch.abs(x_var.grad)")
	print("\t\t"+"optimizer.step()")
	print("\t\t"+"x_var.data = torch.max(torch.min(x_var.data, x_max.data), x_min.data)")

	print("\n\n")
	print("\t#print('-----------------------------------')")
	print("\ttotal_adv_acc=total_adv_acc+adv_acc")

	print("\n\n")
	print("print('Test set nat acc, vote_acc:', total_nat_acc*1.0/10000,total_nat_vote_acc*1.0/10000)")
	print("print('Test set adv acc, vote_acc:', total_adv_acc*1.0/10000,total_adv_vote_acc*1.0/10000)")
	


	

#print('-------------------------------------------------------------------------------------------------------------')

if __name__ == '__main__':
	codebook_name = sys.argv[1]
	eps      = float(sys.argv[2] )
	write_code(codebook_name,eps)


