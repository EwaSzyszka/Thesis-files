
############################################################################################################
'''
1. Start the program and press 'k' to sample skin color
2. Then, do the gesture
3. Press 'd' if you want to detect your sign
'''
############################################################################################################
import cv2
from pynput.keyboard import Listener, Key
from collections import namedtuple
import numpy as np
#Model making imports
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import keras

#Model loading imports
import json
import numpy as np
from keras import models
from keras.models import model_from_json, load_model

#Data procesing imports
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import random
import pickle

#Visualization imports
import os
import cv2
import glob
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from IPython.display import display


#_______ LOADING THE TRAINED MODEL TO MAKE PREDICTIONS ON UNSEEN DATA ____
model = models.load_model('/Users/ewa_anna_szyszka/Desktop/🎓/PyCON/my_model.h5')
############################################################################################################

hand_hist = None
traverse_point = []
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None


def rescale_frame(frame, wpercent=70, hpercent=70):   #originally it was 130 and 130
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    #_, cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


#This is where the rectangles capturing the color are made
def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame

#histogram for the skin color
def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

#cuts only the hand out of the image
def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

    # thresh = cv2.dilate(thresh, None, iterations=5)

    thresh = cv2.merge((thresh, thresh, thresh))

    return cv2.bitwise_and(frame, thresh)


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None

'''
def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None
'''

def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)


list_of_centroids = []
#we find the convexity defect here which is the farthest point from the centroid
def manage_image_opr(frame, hand_hist):
    hist_mask_image = hist_masking(frame, hand_hist)

    hist_mask_image = cv2.erode(hist_mask_image, None, iterations=2)
    hist_mask_image = cv2.dilate(hist_mask_image, None, iterations=2)

    contour_list = contours(hist_mask_image)
    max_cont = max(contour_list, key=cv2.contourArea)

    cnt_centroid = centroid(max_cont)
    cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)


    ########################creating the list of centroids to be able to do the averages########################################################
    list_of_centroids.append(cnt_centroid)

    #If you want to print the values of the list_of_centroids uncomment the next line
    #print(list_of_centroids)
    pressed_key = cv2.waitKey(1)

    #If 'd' is pressed determine whether it is static/horizontal/vertical movement
    if pressed_key & 0xFF == ord('d'):
        threshold_upper = 0.0
        threshold_middle = 0.0
        threshold_lower = 0.0

        threshold_left = 0.0
        threshold_middle_vertical = 0.0
        threshold_right = 0.0

        ################## Checking to which HORIZONTAL tiers did the majority of the points fell ##################

        for i in range(len(list_of_centroids)):

            #HORIZONTAL DETECTORS optimized for 100 screen size

            if list_of_centroids[i][0] in range(0,1222) and list_of_centroids[i][1] in range(0,250):
                threshold_upper+= 1

            if list_of_centroids[i][0] in range(0,1222) and list_of_centroids[i][1] in range(250,550):
                threshold_middle+= 1

            if list_of_centroids[i][0] in range(0,1222) and list_of_centroids[i][1] in range(550,720):
                threshold_lower+= 1

            #VERTICAL DETECTORS

            if list_of_centroids[i][0] in range(0,400) and list_of_centroids[i][1] in range(0,716):
                threshold_left+= 1

            if list_of_centroids[i][0] in range(400,800) and list_of_centroids[i][1] in range(0,716):
                threshold_middle_vertical+= 1

            if list_of_centroids[i][0] in range(800,1280) and list_of_centroids[i][1] in range(0,716):
                threshold_right+= 1


        #70% threshold is set up
        cap = 0.7

        Upper_horizontal_movement = (threshold_upper/len(list_of_centroids))
        Middle_horizontal_movement = (threshold_middle/len(list_of_centroids))
        Lower_horizontal_movement = (threshold_lower/len(list_of_centroids))
        Left_vertical_movement = (threshold_left/len(list_of_centroids))
        Middle_vertical_movement = (threshold_middle_vertical/len(list_of_centroids))
        Right_vertical_mevement = (threshold_right/len(list_of_centroids))

        areas = (Upper_horizontal_movement,Middle_horizontal_movement,Lower_horizontal_movement,Left_vertical_movement,Middle_vertical_movement,Right_vertical_mevement)
        areas_names = ('Upper_horizontal_movement','Middle_horizontal_movement','Lower_horizontal_movement','Left_vertical_movement','Middle_vertical_movement','Right_vertical_mevement')

        print(areas.index(max(areas)))
        print(areas_names[areas.index(max(areas))])

        '''
        if (threshold_upper/len(list_of_centroids)) > cap:
            print('Upper horizontal movement')
        elif (threshold_middle/len(list_of_centroids)) > cap:
            print('Middle horizontal movement')
        elif (threshold_lower/len(list_of_centroids)) > cap:
            print('Lower horizontal movement')

        elif (threshold_left/len(list_of_centroids)) > cap:
            print('Left vertical movement')
        elif (threshold_middle_vertical/len(list_of_centroids)) > cap:
            print('Middle vertical movement')
        elif (threshold_right/len(list_of_centroids)) > cap:
            print('Right vertical movement')

        else:
            print('total chaos')
        '''


        print(threshold_upper/len(list_of_centroids))
        print(threshold_middle/len(list_of_centroids))
        print(threshold_lower/len(list_of_centroids))
        print((threshold_left/len(list_of_centroids)))
        print((threshold_middle_vertical/len(list_of_centroids)))
        print((threshold_right/len(list_of_centroids)))

        list_of_centroids.clear

    #_______ LOADING THE TRAINED MODEL TO MAKE PREDICTIONS ON UNSEEN DATA ____
    #model = models.load_model('my_model.h5')

    ################################################################################################################################################

    #if cnt_centroid[0] in range(0,900) and cnt_centroid[1] in range(300,400):
    #    print('A')


    #range_tuple = namedtuple('range_tuple', 'low high')
    #t = range_tuple(300, 400)
    #if t.low <= cnt_centroid <= t.high:
    #    print('vert')

    if max_cont is not None:
        hull = cv2.convexHull(max_cont, returnPoints=False)
        defects = cv2.convexityDefects(max_cont, hull)

        #Here I commented out the tiny points following the hand
        '''
        far_point = farthest_point(defects, max_cont, cnt_centroid) #those are the yellow points
        #print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
        cv2.circle(frame, far_point, 5, [0, 0, 255], -1)
        if len(traverse_point) < 20:
            traverse_point.append(far_point)
        else:
            traverse_point.pop(0)
            traverse_point.append(far_point)

        draw_circles(frame, traverse_point)
        '''

def main():
    global hand_hist
    is_hand_hist_created = False
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)

        if pressed_key & 0xFF == ord('z'):
            is_hand_hist_created = True
            hand_hist = hand_histogram(frame)

        if is_hand_hist_created:
            manage_image_opr(frame, hand_hist)

        else:
            frame = draw_rect(frame)

        cv2.imshow("Live Feed", rescale_frame(frame))

        #THIS WA ADDED
        '''
        img_counter = 0

        if pressed_key & 0xFF == ord('d'):
            #______ OBJECT OPENING TRAINED MODEL AND PREDICTING IMAGES FROM CAPTURED DATA_____


            # SPACE pressed
            img_name = "Desktop/Code/ImageRecognition/datacapture/opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

            '''

        #UNTIL HERE ADDED

        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()



if __name__ == '__main__':
    main()
