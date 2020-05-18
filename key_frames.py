# -*- coding: utf-8 -*-
import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()

    #Drawing the rectangle area for detection
    start_point = (5, 5)
    end_point = (220, 220)
    color = (255, 0, 0)
    thickness = 2
    cv2.rectangle(frame, (100,100),(600,600),(255,0,0), 2)

    #Coordinates that you want to save --> (102, 102, 499, 499)


    cv2.imshow("test", frame)

    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed --> SAVE THE ENTIRE IMAGE TO FOLDER
        img_name = "/Users/ewa_anna_szyszka/Desktop/Code/ImageRecognition/datacapture/opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))

        # SPACE pressed  --> SAVE THE BOX REGION TO FOLDER
        img_box = "/Users/ewa_anna_szyszka/Desktop/Code/ImageRecognition/datacapturebox/A/opencv_box__{}.png".format(img_counter)
        r = (102, 102, 499, 499)
        imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        cv2.imwrite(img_box, imCrop)
        print("{} BOX written!".format(img_box))


        img_counter += 1

cam.release()

cv2.destroyAllWindows()
