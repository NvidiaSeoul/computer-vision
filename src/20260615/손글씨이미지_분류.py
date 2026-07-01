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

# print(features[8].reshape(8,8))
# plt.imshow(features[8].reshape(8,8),cmap='gray')
# plt.savefig('mnist_0.jpeg')

print(features.shape)
# ==> (1797,8,8,1)  ==> (batch_size,이미지가로,이미지세로, 채널)
features = features.reshape(-1,8,8,1) / 255.0  # 사이즈 변경 + 스케일정규화



# features 와 labels 을  train_x , val_x 로 분할 ( 분할비율은 0.2 )
from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = \
    train_test_split(features, labels , test_size=0.2 ,random_state=42)
print(len(train_x))
print(len(val_x))
# 데이터 전처리 및 데이터준비 완료

# 모델 준비 ==> 이미지를 분류하는 모델 설계 ( 10개 클래스를 분류 , 다중분류 )
# 이미지 분류에 특화된 모델 --> cnn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Flatten

mnist_model = Sequential()
mnist_model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", input_shape=(8,8,1), activation='leaky_relu'))
mnist_model.add( MaxPooling2D(2,2) )
mnist_model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation='leaky_relu'))
mnist_model.add( MaxPooling2D(2,2) )
mnist_model.add( Flatten() )
mnist_model.add( Dense(units=100, activation='leaky_relu') )
mnist_model.add( Dropout(0.3) )
mnist_model.add( Dense(units=60, activation='leaky_relu') )
mnist_model.add( Dense(units=10, activation='softmax') ) # 출력층 ( 10개 분류 )
mnist_model.summary()
# 2개의 Conv , 2개의 Pooling 
# Flatten,  drop-out,  FC layer 층 추가
# 마지막 출력은 10개의 뉴런으로 설정
# 모델 summary()
mnist_model.compile(loss = "sparse_categorical_crossentropy" , 
                    optimizer = "adam" , metrics = ['accuracy'] )
# 모델 compile()
# 손실함수 ==> categorical_crossentropy

# 모델설계 이후 모델 학습 ( fit )
# target(정답) ==> 정수형태로 그대로 사용 ==> sparse_categorical_crossentropy
# categorical_crossentropy ==> 정답을 onehot encoding 상태로 변경해서 전달
# val_x , val_y 전달해서 val_loss 모니터링으로 조기 종료 콜백추가
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
modelcheck_cb = ModelCheckpoint(filepath='./mnist_bestmodel.keras', save_best_only=True)
earlystopping_cb = EarlyStopping(patience=3,verbose=1, restore_best_weights=True)

mnist_model.fit(train_x, train_y, validation_data=(val_x, val_y) , batch_size = 4 , epochs = 100 , verbose = 1,
                callbacks=[modelcheck_cb, earlystopping_cb])