import os
import copy
import csv
from sklearn.svm import SVC
import numpy
from sklearn import svm
from sklearn import preprocessing
import random
import itertools
import pyexcel
from csv import reader
T=10#15
labeled=20#10 or 15
test_size=36#leukemia


#FUNCTIONS
def string_to_float(dts,rows,cols):
	for i in range(rows):
		for j in xrange(cols):
			dts[i][j]=float(dts[i][j])

def label_to_int(dts):	
	for i in range(len(dts)):
		if dts[i][0]=='ALL':
			dts[i][0]=1
		elif dts[i][0]=='AML':
			dts[i][0]=-1

def scaling(dataset):
	features=[]
	for i in range(len(dataset)):
		features.append(dataset[i][1:len(dataset[0])])
	scaled_dts=numpy.array(features)
	max_abs_scalar = preprocessing.MaxAbsScaler()
	l=list(max_abs_scalar.fit_transform(scaled_dts))
	for i in range(len(l)):
		l[i]=list(l[i])
	return l

def Intersection(A,B):
	intersect=[]
	for tup_A in A:
		for tup_B in B:
			if tup_A==tup_B:
				intersect.append(tup_A)
	return intersect

def Difference(A,B):
	differ=[]
	for tup_A in A:
		count=0
		for tup_B in B:
			count =count+1
			if tup_A==tup_B:
				break
		if(count==len(B)):
			differ.append(tup_A)

	return differ


def write_into_file(f_label,f_features,label_content,feature_content):
	print("writing")
	print "feature to be written: ",(len(feature_content))
	for label in label_content:
		#print("writing : ",label)
		f_label.write(str(label)+"\n")

	for tup in feature_content:
		content=""
		for index,text in enumerate(tup):
			content=content+str(index+1)+":"+str(text)+" "
		f_features.write(content+"\n")
	f_features.close()
	f_label.close()




#read from csv file
dts=[]#original dataset


with open('leukemia.csv','r') as file:
	csv_reader=reader(file)
	for row in csv_reader:
		if not row:
			continue
		dts.append(row)


#remove column names
#del dts[0]
#del dts[0]
#del dts[0]
#random.shuffle(dts)

#writing into csv file
#pyexcel.save_as(array=dts,dest_file_name='dtsafterShuffle1.csv')


label_to_int(dts)

#string_to_float(dts,len(dts),len(dts[0]))


#dataset having only features obtained from feature selection
tds=[]

print "-----------------",len(dts[0])
#exit()
print "len(dts[0]):",len(dts[0]),dts[0]
for i in range(len(dts)):
	print "i:",i
	temp=[]
	temp.append(dts[i][0])
	temp.append(dts[i][1375])
	temp.append(dts[i][238])
	temp.append(dts[i][3617])
	#temp.append(dts[i][5152])
	#temp.append(dts[i][583])
	tds.append(temp)

dts=copy.deepcopy(tds)#no of features=2 for leukemia dataset

#test set 
testDs=[]
actual_testDs_labels=[]
for i in range(test_size):
	testDs.append(dts[0][1:len(dts[0])])
	actual_testDs_labels.append(dts[0][0])
	del(dts[0])
#present - len(dts)=36


#IMPORTANT - SCALING MUST BE DONE FOR ALL THE FEATURES (-1 to 1)....
#NO SCALING FOR THE CLASS LABEL - FIRST COLUMN OF THE DATASET
scaled_train_dataset=scaling(dts)
#add labels to the train dataset
for i in range(len(dts)):
	scaled_train_dataset[i].insert(0,dts[i][0])
print "____scaled :",scaled_train_dataset[0]
scaled_test_dataset=scaling(dts)#test without labels

S=[]
V=[]
V_actual_labels=[]
dscaled_train_dataset=copy.deepcopy(scaled_train_dataset)
for i in range(labeled):
	S.append(dscaled_train_dataset[i])
for i in xrange(labeled,len(dscaled_train_dataset)):
	V.append(dscaled_train_dataset[i][1:len(dscaled_train_dataset[0])])
	V_actual_labels.append(dscaled_train_dataset[i][0])
print "S : ",S," LENGTH : ",len(S)
#print "V : ",V," LENGTH : ",len(V)

W=copy.deepcopy(S)
Aprev=[]

#TRAIN SVM CLASSIFIER WITH W
x=[]
y=[]
dW=copy.deepcopy(W)
for i in range(len(dW)):
	x.append(dW[i][1:len(dW[0])])
	y.append(int(dW[i][0]))
string_to_float(x,len(x),len(x[0]))
print x[0]
#x=numpy.array(x)
print x[0]
print "x : ",x,len(x)
print "y : ",y,len(y)

Z=[]
model = SVC(kernel='linear', C=1, gamma=1)
model.fit(x, y)
print(model.score(x, y))
Z= model.predict(V)
confidence_scores=[]
confidence_scores=model.decision_function(V)
#print "Scores:",confidence_scores
#print "Predicted labels for testset using SVM :",Z


#ITERATIONS
Wprev=copy.deepcopy(W)
D=[]
Dprev=[]
for t in range(10):
	print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
	print "Iteration : ",t
	B_minus=[]
	B_plus=[]
	B=[]
	#indices of B_plus and B_minus ....not the count of them
	plus=-1
	minus=-1
	for i in range(len(Z)):
		if confidence_scores[i]<=1.0 and confidence_scores[i]>=0.0:
			plus =	plus  + 1
			dV=copy.deepcopy(V)
			B_plus.append(dV[i])
			B_plus[plus].insert(0,Z[i])
			dB_plus=copy.deepcopy(B_plus)
			B.append(dB_plus[plus])
		elif confidence_scores[i]>=-1.0 and confidence_scores[i]<0.0:
			minus = minus + 1
			dV=copy.deepcopy(V)
			B_minus.append(dV[i])
			B_minus[minus].insert(0,Z[i])
			dB_minus=copy.deepcopy(B_minus)
			B.append(dB_minus[minus])

	print "B_plus: ",B_plus,"--",plus
	print "B_minus: ",B_minus,"--",minus
	print "B: ",B,"--",len(B),plus+minus

	
	if(len(Aprev)==0):
		W=Wprev + B
		print "len(W) if part:",len(W)
		D=copy.deepcopy(B)
		print "len(D) if part:",len(D)
	else:
		D = Intersection(Aprev,B)
		print "len(D) else part:",len(D)
		W = Difference(Wprev,Dprev) + D
		print "len(W) else part:",len(W)

	Aprev=copy.deepcopy(B)
	Dprev=copy.deepcopy(D)
	Wprev=copy.deepcopy(W)
	count1=0
	count2=0
	feature_content=[]
	label_content=[]
	W1=copy.deepcopy(W)
	for i in range(len(W1)):
		count1 = count1 + 1
		feature_content.append(W1[i][1:len(W1[0])])
		label_content.append(str(W1[i][0]))
	print "done "
	V1=copy.deepcopy(V)
	for i in range(len(V1)):
		count2 = count2 + 1
		feature_content.append((V1[i]))
		label_content.append(str(0))
		

	total_count = count1 + count2


	f_features=open("train_features","w")
	f_label=open("train_label","w")
	write_into_file(f_label,f_features,label_content,feature_content)
	os.system("./svmlin -A 2 -W 0.001 -U 1 -R 0.2 train_features train_label")

	Tf_features=open("test_features","w")
	Tf_label=open("test_label","w")
	write_into_file(Tf_label,Tf_features,V_actual_labels,V)
	os.system("./svmlin -f  train_features.weights test_features test_label")
	
	count=0	
	f_test_output=open("test_features.outputs","r")
	for line in f_test_output:
		confidence_scores[count]=float(line)
		if(confidence_scores[count]<0.0):
			Z[count]=-1
		else:
			Z[count]=1
		count=count+1
	


	print "len of confidence_scores:",len(confidence_scores),len(V),len(Z),confidence_scores
	print "==============================================================================================="


testf=open("test_examples","w")
testlabel=open("test_examples_label","w")
write_into_file(testlabel,testf,actual_testDs_labels,testDs)

os.system("./svmlin -f  train_features.weights test_examples test_examples_label")

