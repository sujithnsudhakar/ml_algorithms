import numpy as np
import math
import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
import argparse

def find_entropy(c):
    #Calculate Entropy of entire data set:
    c_vals = np.array(c,dtype = 'object')
    total_rows = float(len(c_vals))
    entropy = 0.0
    unique_c_values = np.unique(c_vals)

    #For each unique_values in data,calculate entropy
    for val in unique_c_values:
        count_val = np.count_nonzero(c_vals == val)
        entropy += -(count_val / total_rows) * math.log((count_val / total_rows),logbase)
    return entropy

def find_entropy_gain(feature,class_lab):
    feature = np.array(feature,dtype = 'object')
    class_lab = np.array(class_lab,dtype = 'object')
    total_pos_neg = len(class_lab)
    average = 0.0
    unique_c = np.unique(class_lab)
    index = 0
    attribute_cpair = np.column_stack((feature, class_lab))
    unique_values = np.unique(feature)
    entropy_dict = {}
    for values in unique_values:
        b = attribute_cpair[attribute_cpair[:,0] == values]
        b_transpose = b.T
        no_pos_neg_attribute = len(b_transpose[0])
        entropy = find_entropy(b_transpose[1])
        entropy_dict[values] = entropy 
        average = average + (no_pos_neg_attribute / total_pos_neg) * entropy
    info_gain = find_entropy(class_lab) - average
    return entropy_dict,info_gain

def find_root(df,node):
    index = 0
    info_gain_list = []
    col_count = len(df.columns)
    val_entropy_list = []
    while index < col_count - 1:
        #Pass last column to find entropy, information gain
        val_entropy_dict,info_gain = find_entropy_gain(df.iloc[:,index],df.iloc[:,-1])
        info_gain_list.insert(index,info_gain)
        val_entropy_list.insert(index,val_entropy_dict)
        index = index + 1
    #Root node is max of the list and its index plus one
    max_gain_index = info_gain_list.index(max(info_gain_list))
    dicttest = val_entropy_list[max_gain_index]
    att_vals = dicttest.keys()
    for value in att_vals:
        if dicttest.get(value) == 0:
            new_node = ET.SubElement(node, 'node')
            new_node.set('feature', 'att'+ str(max_gain_index))
            new_node.set('value',value)
            new_node.set('entropy',str(0))
            new_df = df.loc[df[max_gain_index] == value]           
            new_node.text = str(new_df.iloc[:,-1].unique()[0])
        elif dicttest.get(value) != 0:
            #To fetch the dataset having some specific attribute
            new_node = ET.SubElement(node, 'node')

            new_df = df.loc[df[max_gain_index] == value]
            new_class_lab = new_df.iloc[:,-1] 
            new_entropy = find_entropy(new_class_lab)
            new_node.set('feature', 'att'+ str(max_gain_index))
            new_node.set('value',value)
            new_node.set('entropy',str(dicttest.get(value)))           
            subtree = find_root(new_df,new_node)


#Read data from CSV file
parser = argparse.ArgumentParser()
parser.add_argument("--data")
parser.add_argument("--output")
args = parser.parse_args()
file_name = args.data
output = args.output

data_csv = np.genfromtxt(file_name,delimiter=',',dtype=None,encoding=None)
#print(data_csv)
df = pd.DataFrame(data_csv)
tree = ET.Element('tree');
class_lab = df.iloc[:,-1]
unique_class_lab = np.unique(class_lab)
logbase = len(unique_class_lab)

tree.set('entropy',str(find_entropy(class_lab)))
find_root(df,tree)
with open(output, "w") as f:
        f.write(ET.tostring(tree).decode("utf-8"))



