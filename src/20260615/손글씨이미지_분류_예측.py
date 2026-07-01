from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

mnist = datasets.load_digits() # 손글씨 이미지
features  = mnist['data']
print(len(features[0]))
print(len(features))  # 1797
labels = mnist['target']
print(len(labels)) # 1797
print(np.unique(labels, return_counts=True))

print(features[0].reshape(8,8))

print(features.shape)
# ==> (1797,8,8,1)  ==> (batch_size,이미지가로,이미지세로, 채널)
features = features.reshape(-1,8,8,1) / 255.0  # 사이즈 변경 + 스케일정규화

# features 와 labels 을  train_x , val_x 로 분할 ( 분할비율은 0.2 )
from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = \
    train_test_split(features, labels , test_size=0.2 ,random_state=42)
print(len(train_x))
print(len(val_x))

from tensorflow.keras.models import load_model

# 소숫점 이하 8자리까지 출력
np.set_printoptions(precision=8, suppress=True)
mnistmodel = load_model('mnist_bestmodel.keras')
mnistmodel.summary()

print(mnistmodel.predict(val_x[0:1]) ) # predict ==> 6
print(val_y[0])  # 6

# newdata = np.array([[ 0.  0.  5. 13.  9.  1.  0.  0.]
#  [ 0.  0. 13. 15. 10. 15.  5.  0.]
#  [ 0.  3. 15.  2.  0. 11.  8.  0.]
#  [ 0.  4. 12.  0.  0.  8.  8.  0.]
#  [ 0.  5.  8.  0.  0.  9.  8.  0.]
#  [ 0.  4. 11.  0.  1. 12.  7.  0.]
#  [ 0.  2. 14.  5. 10. 12.  0.  0.]
#  [ 0.  0.  6. 13. 10.  0.  0.  0.]])