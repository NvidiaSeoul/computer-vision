train_dir = '/home/sckit/deeplearning_prj/20260615/cnn_cats_and_dogs_dataset/train'
test_dir = '/home/sckit/deeplearning_prj/20260615/cnn_cats_and_dogs_dataset/test'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train 이미지 증강 유형 생성
train_image_generator = ImageDataGenerator( rescale = 1.0/255.,
                   rotation_range = 20,
                   height_shift_range=0.2 )

# test 이미지는 스케일 변환만해줌
test_image_generator = ImageDataGenerator( rescale = 1.0/255.)

# train image를 읽어 들이면서 image를 증강 시켜주는 제너레이터 생성
train_data_gen = train_image_generator.flow_from_directory(
    train_dir, # 불러올 이미지 경로
    batch_size = 20,
    shuffle = True,
    # 디렉토리 내부 이미지를 불러올때 어떤 형식으로 라벨링 해서 불러 올꺼냐?
    class_mode = 'binary',    #  이진 분류 할 때 binary , 다중분류 떄는 ==> 'categorical'
    # save_to_dir = '/home/sckit/deeplearning_prj/20260615/cnn_cats_and_dogs_dataset/temp', # 증강이미지 저장위치
    # save_prefix = 'gen', # 증강이미지 파일명 앞에 'gen'을 붙여서 만들어라
    # save_format = 'jpg', # 저장할 이미지 확장자 명시
    target_size = (150, 150) # CNN 모델 입력 사이즈로 리사이즈 해라
)

# test image를 읽어 들이면서 image를 증강 시켜주는 제너레이터 생성
test_data_gen = test_image_generator.flow_from_directory(
    test_dir, # 불러올 이미지 경로
    batch_size = 20,
    shuffle = True,
    # 디렉토리 내부 이미지를 불러올때 어떤 형식으로 라벨링 해서 불러 올꺼냐?
    class_mode = 'binary',    #  이진 분류 할 때 binary , 다중분류 떄는 ==> 'categorical'
    # save_to_dir = '/home/sckit/deeplearning_prj/20260615/cnn_cats_and_dogs_dataset/temp', # 증강이미지 저장위치
    # save_prefix = 'gen', # 증강이미지 파일명 앞에 'gen'을 붙여서 만들어라
    # save_format = 'jpg', # 저장할 이미지 확장자 명시
    target_size = (150, 150) # CNN 모델 입력 사이즈로 리사이즈 해라
)


# 모델 준비
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 조기종료 등 콜백설정
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
chckpoint_cb = ModelCheckpoint('./catdog_bestmodel.keras',
                               save_best_only=True)
earlystopping_cb = EarlyStopping(
    patience=3, restore_best_weights=True,
)

# 모델 학습 ( Generator를 활용 )
# steps_per_epoch = 200 ==> 총 train 데이터 4000 개 / batch_sizs(20개 )
history = model.fit(train_data_gen,
                    validation_data = test_data_gen,
                    steps_per_epoch = 200, validation_steps = 10,
                    verbose = 1, epochs = 50,
                    callbacks = [chckpoint_cb, earlystopping_cb]
                    )

# 성능 시각화
import matplotlib.pyplot as plt

acc = history.history['accuracy']  # train 데이터의 정확도
val_acc = history.history['val_accuracy'] # test 데이터의 정확도
loss = history.history['loss']  # train 데이터의 loss
val_loss = history.history['val_loss'] # test 데이터의 loss

import numpy as np
epochs = np.arange(len(acc))
plt.figure()
plt.plot(epochs, loss, label = 'Train loss')
plt.plot(epochs, val_loss, label = 'val loss')
plt.legend()
plt.savefig('catdog_model.jpeg')

