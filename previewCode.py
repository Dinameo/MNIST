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
print(X)
print(Y)