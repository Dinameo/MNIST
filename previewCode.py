from PIL import Image
import numpy as np
import os
import random

train_link = "dataset/trainingSample/trainingSample"
X = []
Y = []
for i in range(10):
    for image_name in os.listdir(f"{train_link}/{i}"):
        image_path = f"{train_link}/{i}/{image_name}"

        img = Image.open(image_path)
        img = img.resize((28, 28))
        img = np.array(img)
        img = img / 255
        img = img.reshape(1, 784)
        img = img[0]
        X.append(img)
        Y.append(np.eye(10)[i])

# gộp X và Y thành 1 mảng
combined = list(zip(X, Y))
# trộn mảng
random.shuffle(combined)
# tách mảng
X[:], Y[:] = zip(*combined)

# X từng hàng là toàn bộ pixel của một ảnh kích thước 28x28
# Y là nhãn của ảnh, từng hàng gồm 10 phần tử, phần tử thứ i là 1 và các phần tử còn lại là 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)


learning_rate = 0.1

input_size = 28 * 28  # 784
hidden_size_1 = 100   # Lớp ẩn thứ nhất
hidden_size_2 = 50    # Lớp ẩn thứ hai
output_size = 10      # Lớp đầu ra (cho phân loại 10 lớp)


np.random.seed(0)
weights_input_hidden1 = np.random.rand(input_size, hidden_size_1)
weights_hidden1_hidden2 = np.random.rand(hidden_size_1, hidden_size_2)

weights_hidden2_output = np.random.rand(hidden_size_2, output_size)

for input_data, target in combined:
    # Input -> hidden layer1
    hidden_layer_input_1 = np.dot(input_data, weights_input_hidden1)
    hidden_layer_output_1 = sigmoid(hidden_layer_input_1)
    # hidden layer1 -> hidden layer2
    hidden_layer_input_2 = np.dot(hidden_layer_output_1, weights_hidden1_hidden2)
    hidden_layer_output_2 = sigmoid(hidden_layer_input_2)
    # hidden layer2 -> ouput
    final_input = np.dot(hidden_layer_output_2, weights_hidden2_output)
    final_output = sigmoid(final_input)

    # cost hay error độ lệch so với output đầu ra
    cost = -(target*np.log10(final_output) + (1-target)*np.log10(1-final_output))
    loss = np.mean(cost)

    #Backpropagation
    # Update loss history
    print(loss)