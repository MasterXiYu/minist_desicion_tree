import csv
import tensorflow as tf
import numpy as np
from sklearn import tree


def load_train_data(data_train_dir):
	with open(data_train_dir,'r',encoding='utf8') as reader:
		next(reader)
		csv_reader = csv.reader(reader, delimiter = ',')
		x_train, y_train =[], []
		for row in csv_reader:
			x_train.append(row[1:])
			y_train.append(row[0])
		return np.array(x_train), np.array(y_train)

def load_test_data(data_test_dir):
	with open(data_test_dir,'r',encoding='utf8') as reader:
		next(reader)
		csv_reader = csv.reader(reader, delimiter = ',')
		x_test = []
		for row in csv_reader:
			x_test.append(row[:])
		return np.array(x_test)

data_train_dir = r"D:\python\kaggle_complete\minist\data\train.csv"
data_test_dir = r"D:\python\kaggle_complete\minist\data\test.csv"
# load_data("D:\python\kaggle_complete\minist\data\\train.csv")
mnist = tf.keras.datasets.mnist
x_train, y_train = load_train_data(data_train_dir)
x_test = load_test_data(data_test_dir)


classifier = tree.DecisionTreeClassifier(criterion='entropy', splitter='random',max_depth=21,min_samples_split=3, random_state=40)
classifier.fit(x_train, y_train)

ans = classifier.predict(x_test)
count = 1
for ele in ans:
	print("{},{}".format(count, ele))
	count+=1