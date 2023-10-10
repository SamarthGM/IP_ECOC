import multiprocessing
import time
import pickle
import sys
from subprocess import PIPE, Popen
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

def code_book_to_combo(code_book):
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
			elif(code_book[i][j] == 0):
				class2.append(i)
		class1 = "_".join(str(k) for k in class1)	
		class2 = "_".join(str(k) for k in class2)			
		#print(class1,class2)
		combo.append([class1, class2])
	return combo
	

def system(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0]

def worker(num):
    """thread worker function"""
    device_id = num[0]	
    class_id=  num[1][0] + ' '+ num[1][1]
    command = "CUDA_VISIBLE_DEVICES="+str(device_id)+"  python train_nclass_to2class_non_robust.py " +class_id + ' > output_log_'+num[1][0]+'__'+num[1][1]+'.txt'
    print ('Worker:', num, command)
    
    ### !!!!! uncomment the following line !!!!!
    system(command)
    

    time.sleep(1)
    return


if __name__ == '__main__':
	
	## change the filename to the codebook filename for which binary classifiers are to be trained.
	code = np.load("sample_codebook_K_10_L_20.npy")
	
	
	print(code)
	code[code == -1] = 0
	print(code,code.shape)

	code_book = (code).tolist()
	print('----------------------------Code_book--------------------------------')
	print_code_book(code_book)


	print('---------------------------------------')
	all_combo = code_book_to_combo( code_book)  

	for all_combo_ in all_combo:
		print(all_combo_)
	print('--------------------------------------------------------------')
	
	## Set this to the number of gpu in  your machine
	no_gpu=1
    
    
	for batch_id in range(0, len(all_combo),no_gpu):  
		print('batch_id:', batch_id)
		print('------------------------------------------------')
		batch = all_combo[batch_id:batch_id+no_gpu]			# Note batch can be less than no_gpu
		#[print( i) for i in batch]
		print('------------------------------------------------')
		print('batch len:',len(batch))
    		
		jobs = []
		for i in range(len(batch)):
			p = multiprocessing.Process(target=worker, args=([i, batch[i] ],) )
			jobs.append(p)
			p.start()
		
		for job in jobs:
			job.join()
			
    		
