#import cv2
import matplotlib.pyplot as plt
# img = cv2.imread('/home/sckit/deeplearning_prj/20260615/cat0.jpg', cv2.IMREAD_COLOR)
# print(img)

from tensorflow.keras.preprocessing import image

img = image.load_img('/home/sckit/deeplearning_prj/20260615/cat0.jpg')
img = image.img_to_array(img)/ 255.0

print(img.shape)
plt.imshow(img)
plt.savefig('cats.jpeg')
