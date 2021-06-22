import cv2
import matplotlib.pyplot as plt
import numpy as np

#1
I = cv2.imread('anh5.jpg')
cv2.imshow('I', I)
#2
height = I.shape[0]
width = I.shape[1]
Ig = np.zeros((height, width), dtype='uint8')
for i in range(height):
    for j in range(width):
        Ig[i][j] = int(0.39 * I[i][j][2]) + int(0.5 * I[i][j][1]) + int(0.11 * I[i][j][0])
cv2.imshow("Ig", Ig)
print("Mức xám tb ảnh ig: ", np.mean(Ig))
#3
Ihsv=cv2.cvtColor(I,cv2.COLOR_BGR2HSV) # OpenCV: BGR:0->B,1->G,2->R Hoac HSV: 0->H,1->S, 2->V
cv2.imshow(" Hien thi kenh h cua Ihsv", Ihsv[:,:,0])
print("mức sáng trung bình của kênh S", np.mean(Ihsv[:, :, 1]))
cv2.waitKey(0)
# 4
thresh, Ib = cv2.threshold(Ig, 0, 255, cv2.THRESH_OTSU)

# 5
contours, _ = cv2.findContours(Ib, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_area = 0.0
contours_max = []
for contour in contours:
    if cv2.arcLength(contour, True) > max_area/3.0:
        max_area = cv2.arcLength(contour, True)
        contours_max = contour
print("giá trị chu vi là: ", max_area)
cv2.drawContours(I, [contours_max], -1, (0,255,255), 2)
cv2.imshow("Anh sau khi bien doi ", I)

cv2.waitKey(0)
cv2.destroyAllWindows()