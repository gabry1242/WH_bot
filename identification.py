import imageio
import pyautogui
import numpy as np
import cv2 as cv
myScreenshot = pyautogui.screenshot(region=(650,150,1270,820))
image_path = 'C:/Users/gabri/Pictures/answers.png'
myScreenshot.save(image_path)

img = cv.imread(image_path)
assert img is not None, "file could not be read, check with os.path.exists()"


cv.imshow("screen",img)
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 35, 255, 0)
cv.imshow('Binary image', thresh)
cv.waitKey(0)
contours1, hierarchy1 = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# draw contours on the original image for `CHAIN_APPROX_SIMPLE`
image_copy1 = img.copy()
cv.drawContours(image_copy1, contours1, -1, (0, 255, 0), 2, cv.LINE_AA)
cv.imshow("countour", image_copy1)
cv.waitKey(0)
cv.destroyAllWindows()