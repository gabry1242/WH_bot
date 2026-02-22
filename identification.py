import imageio
import pyautogui
import numpy as np
import cv2 as cv
myScreenshot = pyautogui.screenshot(region=(655,150,1250,820))
image_path = 'C:/Users/gabri/Pictures/answers.png'
myScreenshot.save(image_path)

img = cv.imread(image_path)
assert img is not None, "file could not be read, check with os.path.exists()"

#translate the image in greyscale, apply treshold, draw the countours of the message boxes
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 35, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# draw contours on the original image for `CHAIN_APPROX_SIMPLE`
image_copy = img.copy()
cv.drawContours(image_copy, contours, -1, (0, 255, 0), 2, cv.LINE_AA)
cv.imshow("countour", image_copy)
cv.waitKey(0)


#Block that saves all the countour in different images, iterates through all the countours and crop the image based on the bounding rect
i = 0
for c in contours:
    # get the bounding rect
    x, y, w, h = cv.boundingRect(c)
    # to save the images
    cv.imwrite('C:/Users/gabri/Pictures/img_{}.jpg'.format(i), img[y:y+h,x:x+w])
    i += 1
cv.destroyAllWindows()