import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

# путь для обучающей выборки
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

# Получаем имена обучающих классов и сохраняем их в списке
train_path = args["trainingSet"]
training_names = os.listdir(train_path)

# Получаем все пути к изображениям и сохраняем их в списке
# image_paths и соответствующая метка в image_paths
image_paths = []
image_classes = []
class_id = 0
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imutils.imlist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1

# Создание feature extraction и keypoint detector
fea_det = cv2.xfeatures2d.SIFT_create()
#fea_det = cv2.xfeatures2d.SURF_create()

# Список, где хранятся все дескрипторы
des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    kpts, des = fea_det.detectAndCompute(im, None)
    des_list.append((image_path, des))   
    
# Сложить дескрипторы вертикально в numpy массив
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  

# k-means кластеринг
k = 100
voc, variance = kmeans(descriptors, k, 1) 

# Рассчитаем гистограмму функций
im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# Выполним векторизацию Tf-Idf
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Масштабируем слова
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# Обучаем Linear SVM
clf = LinearSVC()
clf.fit(im_features, np.array(image_classes))

# Сохраняем SVM
joblib.dump((clf, training_names, stdSlr, k, voc), "bof.pkl", compress=3)