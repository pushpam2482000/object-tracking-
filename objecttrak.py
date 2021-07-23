from os import rename
import cv2
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
#import tensorflow as tf


#this is a data frame to keep a log of the traffic
log = pd.DataFrame( columns=["sanp number", "day", "month", "date", "time", "year", "dimensions", "comments"])
#this will also keep in the record of cnt with the snaps
log2 = pd.DataFrame( columns=["sanp number", "day", "month", "date", "time", "year", "comments"])
#log= pd.read_csv("log.csv")

def vidcap(source):
    
    cap = cv2.VideoCapture('stock_vid1.mp4')
    cv2.namedWindow('frame')

    while True:

        suc, frame = cap.read()
        cv2.imshow('frame', frame)

        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):

            break
        elif k  == ord('s'): # wait for 's' key to save and exit

            cv2.imwrite('target.png',frame)
            #scv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyWindow('frame')


def draw_a_rectangel(frame):

    # this will draw a rectangel on the frame of interest
    img = cv2.imread("flower.jpg")
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scaled_image = cv2.resize(rgb_img,(0,0), rgb_img, 0.4, 0.4)

    def just_print_for_all(event, x, y, flags, param):

        print("chercher tech is my name")

    # set when to have a call back
    cv2.namedWindow("Title of Popup Window")

    #what to happen on call back
    cv2.setMouseCallback("Title of Popup Window", just_print_for_all)

    #show image to user with titles
    cv2.imshow("Title of Popup Window", scaled_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    pass

def target_ana():

    def crapfunc(c):
        pass

    #here we create some trackebars these bars will help us select our color 
    cv2.namedWindow("trackbars")

    # the createTrakbars funtion of opencv needs to call a function every time it is called so we pass a empty funtion here 
    cv2.createTrackbar("hue min","trackbars",0 , 179 , crapfunc )
    cv2.createTrackbar("hue max","trackbars",179 , 179 , crapfunc )
    cv2.createTrackbar("sat min","trackbars",0 , 255 , crapfunc )
    cv2.createTrackbar("sat max","trackbars",255 , 255 , crapfunc )
    cv2.createTrackbar("val min","trackbars",0 , 255 , crapfunc )
    cv2.createTrackbar("val max","trackbars",255 , 255 , crapfunc )

    #but why do we care about the min and also the max 
    #why not the absolute value 
    #answer is that we want a range of color one specific value wont cut it

    img = cv2.imread("target.png")

    #thsi will change the image to hsv format
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    while True:
    
        #here we get the values of our image hsv from the trackbars we made
        h_min = cv2.getTrackbarPos("hue min" , "trackbars")
        h_max = cv2.getTrackbarPos("hue max" , "trackbars")
        s_min = cv2.getTrackbarPos("sat min" , "trackbars")
        s_max = cv2.getTrackbarPos("sat max" , "trackbars")
        v_min = cv2.getTrackbarPos("val min" , "trackbars")
        v_max = cv2.getTrackbarPos("val max" , "trackbars")

        
        #with the use of numpy lib we make some limits to make the mask 
        lower = np.array([h_min,s_min,v_min])
        upper = np.array([h_max,s_max,v_max])

        #here we make the mask
        mask = cv2.inRange(imghsv,lower,upper)

        #its is very imp to see live results of our manipulation 
        #cv2.imshow("carap" , img)
        #cv2.imshow("carap 2" , imghsv)
        cv2.imshow("MASK" , mask)

        #for some reason waitkey(0) was not working and this is also far better than using just the waitkey()
        #thsi is programed to quit at the prees of letter "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #noe we print the values of our manipulation just in case if we need them in future
    print(h_min,h_max , s_min,s_max , v_min,v_max)

    #this is the img we need with the mask on 
    #we used the bitwise and operation here 
    final = cv2.bitwise_and(img , img , mask = mask)
    cv2.imshow("IMAGE VIEW", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return (lower, upper)

def the_main_preview():

    lower, upper = target_ana()
    cap = cv2.VideoCapture('stock_vid1.mp4')

    while True:

        suc, frame = cap.read()
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv,lower,upper)
        final = cv2.bitwise_and(frame , frame , mask = mask)
        cv2.imshow("FINAL VIEW", final)

        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break


def sssaver(name, target, target_mask, num):

    #this will save the the required snap of the vid file
    cv2.imwrite(name+str(num)+".png", target)
    cv2.imwrite(name+str(num)+"maks"+".png", target_mask)
    
def logger(name, comment, dat):

    #this one make an excel file to save the time and other info of the sanp
    timedet = time.ctime().split()
    log.loc[len(log.index)] = [name, timedet[0], timedet[1], timedet[2], timedet[3], timedet[4], dat, comment]
    log2.loc[len(log.index)] = [name, timedet[0], timedet[1], timedet[2], timedet[3], timedet[4], comment]
    log2.to_csv("log.csv")
    log.to_csv("log_with_cnt_details.csv")


cap1 = cv2.VideoCapture('highway.mp4')

object_detec = cv2.createBackgroundSubtractorMOG2(varThreshold=50)

i = 0

while True:

    _, frame = cap1.read()
    height, wicth, _ = frame.shape
    print(height, wicth)
    #mask = object_detec.apply(frame)

    roi = frame[400:,100:850]
    #roi = frame

    mask_roi = object_detec.apply(roi)

    _, mask_roi = cv2.threshold(mask_roi, 254, 255, cv2.THRESH_BINARY)

    conture, _ = cv2.findContours(mask_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    dat = []

    for cnt in conture:

        area = cv2.contourArea(cnt)

        if area > 200:

            #cv2.drawContours(roi, cnt, -1 , (255,0,0), 3)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x+w, y+h), (255,0,0), 3 )
            dat = cnt

    cv2.imshow('highway', frame)
    cv2.imshow('mask', mask_roi)
    cv2.imshow('reginon of interst', roi)

    k = cv2.waitKey(30) & 0xff

    if k == ord('q'):

        # this will end the program
        break

    if k == ord('s'):

        # this is to save snaps of the .mp4 file
        i+=1
        sssaver("snap", frame, mask_roi, i)
        nothing = "NONE"
        logger("snap"+str(i), nothing, dat)
        continue


cap1.release()
cv2.destroyAllWindows()