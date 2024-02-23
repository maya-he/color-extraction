import cv2
import numpy as np

img = cv2.imread('plate5.png')
height, width, _ = np.shape(img)
# print("rows",height)
# print("cols",width)

res = cv2.resize(img, dsize=(250, 200), interpolation=cv2.INTER_CUBIC)
crop = res[0:80 , 0:250]

data = np.reshape(crop, (80 * 250, 3))
data = np.float32(data)

#clustreing using kmeans
number_clusters = 1     #one dominat color
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #when to stop the algorithm, 10 = max_iter, 1.0 = epsilon
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(data, number_clusters, None, criteria, 10, flags)

def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return bar, (red, green, blue)

bars = []
rgb_values = []

for index, row in enumerate(centers):
    bar, rgb = create_bar(200, 200, row)
    bars.append(bar)
    rgb_values.append(rgb)

img_bar = np.hstack(bars)

font = cv2.FONT_HERSHEY_SIMPLEX

for index, row in enumerate(rgb_values):
    print(f'RGB{row}')
    # color_name = rgb_to_name(row,spec='css3')
    # print(color_name)

cv2.imshow('Image', img)
# cv2.imshow('resized', res)
cv2.imshow('cropped', crop)
cv2.imshow('Dominant colors', img_bar)
cv2.waitKey(0)