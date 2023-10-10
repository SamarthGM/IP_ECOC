import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import pickle


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,  shuffle=False, num_workers=1)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)


train_dict={}
test_dict={}

for i in range(10):
  train_dict[i] = []
  test_dict[i]  = []


for i, data in enumerate(trainloader, 0):

	input_, label = data
	#print(i, label.data.item())
	train_dict[label.data.item()].append(i)



for i, data in enumerate(testloader, 0):

	input_, label = data
	#print(i, label.data.item())
	test_dict[label.data.item()].append(i)


data_indices_dict={'train': train_dict,
	   'test' : test_dict
	  }

pickle.dump(data_indices_dict, open("indices_dict", "wb"))  

indices_dict = pickle.load(open("indices_dict", "rb"))
