
train_dir = '/home/sckit/deeplearning_prj/20260615/cnn_cats_and_dogs_dataset/train'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 증강 유형 생성
train_image_generator = ImageDataGenerator( rescale = 1.0/255.,
                   rotation_range = 20,
                   height_shift_range=0.2 )

# image를 읽어 들이면서 image를 증강 시켜주는 제너레이터 생성
train_data_gen = train_image_generator.flow_from_directory(
    train_dir, # 불러올 이미지 경로
    batch_size = 2,
    shuffle = False,
    save_to_dir = '/home/sckit/deeplearning_prj/20260615/cnn_cats_and_dogs_dataset/temp', # 증강이미지 저장위치
    save_prefix = 'gen', # 증강이미지 파일명 앞에 'gen'을 붙여서 만들어라
    save_format = 'jpg', # 저장할 이미지 확장자 명시
    target_size = (150, 150) # CNN 모델 입력 사이즈로 리사이즈 해라
)

i = 0
for b in train_data_gen:
    i += 1
    if i > 2:
        break