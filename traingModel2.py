
import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn import metrics

train_data = []
train_lable = []
train_data_anaysis = []


first_lock = False


part = 'orginal images'
pre_prossess_img_parth = 'finalResizeIMG'
#test_parth = 'predict images'

cls = os.listdir( part)
#print(cls)
for clss in cls:
    cls = os.listdir(part+'/'+clss)
    print(clss)
    for jpg_image in cls:

        image_parth = (part + '/' + clss+'/'+jpg_image)
        #print(jpg_image)
        image = cv2.imread(image_parth)

#######################################hog features extractions##############################################3

        resize_img = cv2.resize(image, (300, 300),interpolation = cv2.INTER_NEAREST)
        hog_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
        #fd, hog_img = hog(hog_img, orientations=5, pixels_per_cell=(5, 5),cells_per_block=(2, 2), visualize=True, multichannel=True)
        hog_img = cv2.adaptiveThreshold(hog_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 199, 5)
        hog_img = cv2.adaptiveThreshold(hog_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 199, 5)

        kernel = np.ones((5, 5), np.uint8)
        #hog_img = cv2.erode(hog_img, kernel)
        hog_img = cv2.dilate(hog_img, kernel, iterations=5)
        hog_img = cv2.GaussianBlur(hog_img, (3, 3), 0)

        hog_img = cv2.adaptiveThreshold(hog_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)

        # kernel = np.array([
        #     [-1, 0, 1],
        #     [-2, 0, 2],
        #     [-1, 0, 1]
        # ])
        # hog_img = cv2.filter2D(hog_img, -1, kernel)
        #hog_img = cv2.Canny(image=hog_img, threshold1=100, threshold2=200)

        hog_img = cv2.bitwise_not(hog_img)

        number_of_white_pix = np.sum(hog_img == 255)
        number_of_black_pix = np.sum(hog_img == 0)

        train_data_anaysis.append(number_of_white_pix)
        #print(number_of_white_pix)

        if clss == 'melanoma':
            x = 0
            train_lable.append(x)
        if clss == 'normal skin':
            x = 1
            train_lable.append(x)

############################################################################################
        cv2.imshow('hog',hog_img)
        cv2.imshow('img', resize_img)
        cv2.waitKey(1)

        save_parth = pre_prossess_img_parth+'/'+ clss
        cv2.imwrite(os.path.join(save_parth, jpg_image), resize_img)


plt.plot(train_data_anaysis)
plt.show()

#print(train_lable)

svm2_data = np.array(train_data_anaysis)
svm2_data = svm2_data.reshape((34 ,1))


#used for get training data

from sklearn.model_selection import train_test_split

svm2_data1, X_test, train_lable1, Y_test = train_test_split(svm2_data, train_lable, test_size=0.2, random_state=5)

#classification

clf = svm.SVC()
clf = svm.SVC(gamma=0.01, C=100) #0.0.1
clf.fit(svm2_data1,train_lable1)
y_predict = clf.predict(X_test)


#clf = LinearSVC(max_iter=10000)
#clf.fit(svm2_data1,train_lable1)

#use for test accuracy


print("Accuracy:",metrics.accuracy_score(Y_test, y_predict))

from sklearn.metrics import classification_report

print(classification_report(Y_test, y_predict ))


# save file



################### finish training part##################################







################### predict script ##################################


test_parth = 'predict images'

while True:

    image_name = input("Enter Photo name - :")
    #image_name = 'test'

    test_image_org = cv2.imread(test_parth + '/' + image_name + '.jpg')
    test_image = cv2.resize(test_image_org, (300, 300), interpolation=cv2.INTER_NEAREST)

    
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image = cv2.adaptiveThreshold(test_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)
    test_image = cv2.adaptiveThreshold(test_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)

    kernel = np.ones((5, 5), np.uint8)
    # hog_img = cv2.erode(hog_img, kernel)
    test_image = cv2.dilate(test_image, kernel, iterations=5)
    test_image = cv2.GaussianBlur(test_image, (3, 3), 0)

    test_image = cv2.adaptiveThreshold(test_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
    test_image = cv2.bitwise_not(test_image)

    number_of_white_pix = np.sum(test_image == 255)
    number_of_black_pix = np.sum(test_image == 0)

    predict_data = np.array([number_of_white_pix])
    predict_data = predict_data.reshape((1, -1))

    print(clf.predict(predict_data))

    if clf.predict(predict_data) == [0]:
        rezalt = 'Melanoma'
        print('melanoma')

    if clf.predict(predict_data) == [1]:
        rezalt = 'Normal'
        print('normal')
    prw = cv2.resize(test_image_org, (500, 500),interpolation = cv2.INTER_NEAREST)
    prw = cv2.putText(prw, rezalt, (50,450), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 2, cv2.LINE_AA)
    cv2.imshow('test image', prw)
    cv2.waitKey(1)


################### finish predict part##################################
