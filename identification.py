import imageio
import pyautogui
import numpy as np
import cv2 as cv
myScreenshot = pyautogui.screenshot()
image_path = 'C:/Users/gabri/Pictures/answers.png'
myScreenshot.save(image_path)

img = cv.imread(image_path)
assert img is not None, "file could not be read, check with os.path.exists()"

px =img[100,100]
print(px)
cv.imshow("screen",img)
cv.waitKey(0)
cv.destroyAllWindows()