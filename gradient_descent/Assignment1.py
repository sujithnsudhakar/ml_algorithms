import numpy as np
import csv
import argparse

def gradient_descent(x,y,learning_rate,threshold):
	w_curr = np.zeros((len(x)),dtype=float)
	print(x)

	w0_curr = 0.0
	output_list = []
	
	i = 1

	prev_error = error = 0

	while (i):
		y_pred = w0_curr 

		weight_index = 0;
		while weight_index < len(x):
			y_pred += w_curr[weight_index] * x[weight_index]
			weight_index += 1; 

		#Python syntax for each values in y-y_pred, compute the square of values
		error = sum([val**2 for val in (y-y_pred)])

		if (prev_error != 0) and ((prev_error - error) < threshold):
			break			
		prev_error = error

		#Add weights error and iteration values into a single array
		#Temporary arrays for storing consolidated records
		w_curr_round = np.around(w_curr, decimals=4)
		temp_array1 = np.insert(w_curr_round, 0, i-1)
		temp_array2 = np.insert(temp_array1, 1, round(w0_curr,4))
		temp_array3 = np.insert(temp_array2, (len(x)+2),round(error,4))

		#Add the consolidated array into single list so as to write into csv file
		output_list.insert(i,temp_array3)

		weight_index = 0
		w0_grad = sum(y-y_pred)
		while weight_index < len(x):
			w_grad = sum(x[weight_index] * (y-y_pred))
			w_curr[weight_index] = w_curr[weight_index] + learning_rate * w_grad
			weight_index += 1
		w0_curr = w0_curr + learning_rate * w0_grad

		i = i+1	
	#Write list of each rows into csv file
	print(output_list)
	with open("output.csv", "w", newline="") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(output_list)

#Read data from command line arguments using arg parser so as to fetch values using keys
parser = argparse.ArgumentParser()
parser.add_argument("--data")
parser.add_argument("--learningRate")
parser.add_argument("--threshold")

args = parser.parse_args()
file_name = args.data
learning_rate_input = args.learningRate
threshold_input = args.threshold

learning_rate = float(learning_rate_input)
threshold = float(threshold_input)

#Read the data from csv file
data_csv = np.genfromtxt(file_name,delimiter=',')
#Transposing tp convert into columns because we fetch rows from csv
data_array = data_csv.T
gradient_descent(data_array[:-1],data_array[-1],learning_rate,threshold);
