import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *

# Загрузить классификатор,имена классов, скейлер, количество кластеров и словрь
clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")

# Получить путь набора тестирования
parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testingSet", help="Path to testing Set")
group.add_argument("-i", "--image", help="Path to image")
parser.add_argument('-v',"--visualize", action='store_true')
args = vars(parser.parse_args())

# Получаем путь к тестовым изображениям и сохраняем их в списке
image_paths = []
if args["testingSet"]:
    test_path = args["testingSet"]
    try:
        testing_names = os.listdir(test_path)
    except OSError:
        print ("No such directory {}\nCheck if the file exists".format(test_path))
        exit()
    for testing_name in testing_names:
        dir = os.path.join(test_path, testing_name)
        class_path = imutils.imlist(dir)
        image_paths+=class_path
else:
    image_paths = [args["image"]]
    
# Создаем объекты извлечения и находим ключевые точки
#fea_det = cv2.xfeatures2d.SIFT_create()
fea_det = cv2.xfeatures2d.SURF_create()

# Список дескрипторов
des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    if im is None:
        print ("No such file {}\nCheck if the file exists".format(image_path))
        exit()
    kpts, des = fea_det.detectAndCompute(im, None)
    des_list.append((image_path, des))   
    
# Сложить дескрипторы вертикально в numpy массив
descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor)) 

# 
test_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1

# Выполняем Tf-Idf векторизацию
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Масштабируем test_features
test_features = stdSlr.transform(test_features)

# Выполняем прогноз
predictions = [classes_names[i] for i in clf.predict(test_features)]


# Показывать результаты, если visualize = true
if args["visualize"]:
    result_file = open('surf.txt', "w", encoding='utf-8')
    for image_path, prediction in zip(image_paths, predictions):
        image = cv2.imread(image_path)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        pt = (0, 3 * image.shape[0] // 4)
        cv2.putText(image, prediction, pt ,cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 255, 0], 2)
        result_file.write(image_path[10:] + '->' + prediction + '\n')
        cv2.imwrite('result/surf/' + image_path[10:-5] + '.png', image)
        cv2.imshow("Image", image)
        cv2.waitKey(100)
    result_file.close()