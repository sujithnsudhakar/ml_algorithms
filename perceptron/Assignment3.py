import numpy as np
import csv
import argparse
import pandas as pd

def gradient_descent(x,y,flag):
	
	w_curr = np.zeros(len(x),dtype=float)
	w0_curr = 0.0
	i = 1
	error = 0
	while (i <= 101):
		#Check the flag if it is constant or annealing and set accordingly:
		if flag == 'constant' :
			learning_rate = 1
		else :
			learning_rate = 1 / i
		y_pred = w0_curr 

		weight_index = 0;
		while weight_index < len(x):
			y_pred = y_pred + (w_curr[weight_index] * x[weight_index])
			weight_index += 1;

		#Replace the value of predicted class labels to zeros and ones so as to match the actual class labels
		y_pred[y_pred <= 0] = 0
		y_pred[y_pred > 0] = 1

		#Python syntax for each values in y-y_pred, compute the square of values
		error = sum([val**2 for val in (y-y_pred)])

		#Output list is created only for writing the rows into tsv file, otherwise not required
		if flag == 'constant' :
			output_list1.insert(i,int(error))
		else :
			output_list2.insert(i,int(error))
		weight_index = 0
		w0_grad = sum(y-y_pred)
		while weight_index < len(x):
			w_grad = sum(x[weight_index] * (y-y_pred))
			w_curr[weight_index] = w_curr[weight_index] + learning_rate * w_grad
			weight_index += 1
		w0_curr = w0_curr + learning_rate * w0_grad
		i = i+1
	
#Implemented using panda dataframe ---Start
#Read data from command line arguments using arg parser so as to fetch values using keys
parser = argparse.ArgumentParser()
parser.add_argument("--data")
parser.add_argument("--output")

#file_name = 'Example.tsv'
args = parser.parse_args()
file_name = args.data
output_file = args.output

output_list1 = []
output_list2 = []


#Read the data from tsv file
data_tsv = np.genfromtxt(file_name,delimiter='\t',dtype=None,encoding=None)
df = pd.DataFrame(data_tsv)

#print(df)

#Replace the values of class labels with zeros and ones
df.iloc[:,0] = df.iloc[:,0].replace('A',1)
df.iloc[:,0] = df.iloc[:,0].replace('B',0)

#In iloc first parameter indicates range of rows and second one indicates the range of columns
class_lab = df.iloc[:,0].values
#Restricting the dataset to accept only two features as requirement is not clear about preprocessing
#Will not work if there is more than two features
features = df.iloc[:,1:3].values.T
#print('Shape',features.shape)
gradient_descent(features,class_lab,'constant');
gradient_descent(features,class_lab,'annealing');
#Implemented using panda dataframe ---End
#Write list of each rows into tsv file
with open(output_file, "w", newline="") as tsvfile:
	#create csv writer object having tab space as delimiter
	tsv_writer = csv.writer(tsvfile, delimiter = '\t')
	tsv_writer.writerow(output_list1)
	tsv_writer.writerow(output_list2)
