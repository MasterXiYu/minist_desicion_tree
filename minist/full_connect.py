import csv
import tensorflow as tf
import numpy as np

def load_train_data(data_train_dir):
	with open(data_train_dir,'r',encoding='utf8') as reader:
		next(reader)
		csv_reader = csv.reader(reader, delimiter = ',')
		x_train, y_train =[], []
		for row in csv_reader:
			x_train.append(row[1:])
			y_train.append(row[0])
		return np.array(x_train).astype(np.int), np.array(y_train).astype(np.int)

def load_test_data(data_test_dir):
	with open(data_test_dir,'r',encoding='utf8') as reader:
		next(reader)
		csv_reader = csv.reader(reader, delimiter = ',')
		x_test = []
		for row in csv_reader:
			x_test.append(row[:])
		return np.array(x_test).astype(np.int)

data_train_dir = r"D:\python\kaggle_complete\minist\data\train.csv"
data_test_dir = r"D:\python\kaggle_complete\minist\data\test.csv"
# load_data("D:\python\kaggle_complete\minist\data\\train.csv")
mnist = tf.keras.datasets.mnist
x_train, y_train = load_train_data(data_train_dir)
x_test = load_test_data(data_test_dir)

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(10)]
)
predictions = model(x_train).numpy()
ans = tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

prediction = model(x_train)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

probability_model = tf.keras.Sequential([
	model,
	tf.keras.layers.Softmax()
])
ans_2 = probability_model(x_test[:])
index_id = np.argmax(ans_2, axis=1)
print("ImageId,Label")
count = 1
for ele in index_id:
	print("{},{}".format(count, ele))
	count+=1

#
