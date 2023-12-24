import matplotlib.pyplot as plt

from HOG_SVM import HogSVMClassifier
from utility import *

# plot_window_aspect_ratio()

# img = cv.imread("antrenare/barney/0118.jpg")
# plt.imshow(img)
# plt.show()
# img = cv.pyrDown(img)
# plt.imshow(img)
# plt.show()
# img = cv.pyrDown(img)
# plt.imshow(img)
# plt.show()

# img = cv.imread("antrenare/barney/0118.jpg")
# img = img[89:308, 129:360, :]
# img = cv.resize(img, (40, 40))
# plt.imshow(img)
# plt.show()

hog_rider = HogSVMClassifier()
# hog_rider.train()
hog_rider.test()
