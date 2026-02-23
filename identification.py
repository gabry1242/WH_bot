import imageio
import pyautogui
import numpy as np
import cv2 as cv
import pytesseract


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


#I am not interested in my messages so I would likely want to remove those
#look at the pixel color and discard the one with a green color
#Block that saves all the countour in different images, iterates through all the countours and crop the image based on the bounding rect
i = 0
for c in contours:
    # get the bounding rect
    x, y, w, h = cv.boundingRect(c)
    px =img[int(y+3), int(x+w/2)]  #looking at a pixel top middle of the message 
    if px[1] > 60: #if the pixel green channel is higher than 60 (hence pixel is green don't save the message) we just care about other's messages
        pass
    else:
        cv.imwrite('C:/Users/gabri/Pictures/img_{}.jpg'.format(i), img[y:y+h,x:x+w])      # to save the images
        i += 1
cv.destroyAllWindows()

for j in range (i-1, -1, -1):
    result = cv.imread(f'C:/Users/gabri/Pictures/img_{j}.jpg')
    print(result.shape)
    h,w,c = result.shape
    cropped = result[:, :w-37]
    cv.imshow("cropped", cropped)
    res_gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    _, binary_image = cv.threshold(res_gray, 180, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imshow("countour", binary_image)
    cv.waitKey(0)
    text = pytesseract.image_to_string(binary_image, lang='ita')
    print(text)