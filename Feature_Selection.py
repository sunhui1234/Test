import csv
import math
import random
import sys
from collections import Counter
import copy

def Consistency(reduct):
	if not reduct:
		return 0
	x=0
	trds=trainingds

	visited=[0]*len(trainingds)
	for i in xrange(len(trainingds)):
		if visited[i]==0:
			c1=c2=total=0
			if trds[i][0]=='AML':
				c1=c1+1
			else:
				c2=c2+1
			total=1
			visited[i]=1
			k=0
			
			for j in xrange(len(trainingds)):
				flag=0
				if visited[j]==0:
					for k in xrange(len(reduct)):
						if(float(trds[i][reduct[k]])!=float(trds[j][reduct[k]])):
							flag=1
							break

					if flag==0:
						visited[j]=1
						total=total+1
						if trds[j][0]=='AML':
							c1=c1+1
						else:
							c2=c2+1

			if total!=1:
				if c1>c2:
					max1=c1
				else:
					max1=c2
				x=x+total-max1
				
	delta=float(float((len(trainingds)-x))/float(len(trainingds)))
	return delta

ds=[]
trainingds=[]
testingds=[]
with open('leukemia.csv','rb') as csvfile:
	reader=csv.reader(csvfile,delimiter=' ',quotechar='|')
	rows=[x for x in reader]
	for j in xrange(73):
		r=rows[j][0].split('\t')
		
		ds.append(r)


del ds[0]
del ds[0]
del ds[0]

random.shuffle(ds)

for x in range(len(ds)):
	trainingds.append(ds[x])



red=[]
attrs=[]
for i in xrange(1,len(ds[4])):
	attrs.append(i)

noOfAttrs=len(trainingds[2])
SIG=[sys.float_info.min]*noOfAttrs
count=0
Boolean=True
while(Boolean):
	maximum=-sys.float_info.max
	for ai in list(set(attrs)-set(red)):

		redUai=copy.copy(red)
		redUai.append(ai)
		
		SIG[ai]=Consistency(redUai)-Consistency(red)
		if SIG[ai]==1.0:
			attrs.remove(ai)
			continue


		if SIG[ai]>maximum:
			maximum=SIG[ai]
			ak=ai
	if SIG[ak]>0:
		for ai in list(set(attrs)-set(red)):
			if SIG[ai]==SIG[ak]:
				red.append(ai)
	else:
		Boo1ean=False
	count=count+1
f=open("FS_leukemia","w")
for i in range(len(red)):
	f.write(red[i])s