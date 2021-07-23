import cv2
import numpy as np



def vidcap():

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


def draw_a_rectangel():
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



vidcap()
the_main_preview()

cv2.destroyAllWindows()

