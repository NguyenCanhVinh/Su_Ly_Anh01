import cv2
import numpy as np
import matplotlib.pyplot as plt

#1 Hien thi kenh B cau anh I
I=cv2.imread('hat1.PNG')
cv2.imshow("Hien thi kenh B cua anh I",I[:,:,0])
cv2.waitKey(0)

#2.  Chuyển ảnh sang biểu diễn HSV được ma trận Ihsv. Hiển thị kênh H của Ihsv
Ihsv=cv2.cvtColor(I,cv2.COLOR_BGR2HSV) # OpenCV: BGR:0->B,1->G,2->R Hoac HSV: 0->H,1->S, 2->V
cv2.imshow(" Hien thi kenh h cua Ihsv", Ihsv[:,:,0])
print("mức sáng trung bình của kênh S", np.mean(Ihsv[:, :, 1]))
cv2.waitKey(0)

#3
hist = cv2.calcHist(Ihsv[:, :, 1], channels=[2], mask=None, histSize=[256], ranges=[0, 256])
plt.plot(hist)
plt.title('Histogram cua kenh V')
plt.show()
#4
Is = cv2.blur(Ihsv[:, :, 2], (3,3))
cv2.imshow("Is", Is)

y = 9
x = 11
print("Các độ xám của cửa sổ lân cận 5x5 là: ")
for i in range(-2, 3):
    for j in range(-2, 3):
        if y + i >=0 & y + i <= Is.shape[0] - 1 & x + j >= 0 & x + j < Is.shape[1] -1:
            print(Is[y + i][x + j])
#5
thresh, Ib = cv2.threshold(Is, 0, 255, cv2.THRESH_OTSU)
cv2.imshow("Ib", Ib)

contours, con = cv2.findContours(Ib, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv_max = 0.0
contours_max = []
for ct in contours:
    if cv_max <= cv2.arcLength(ct, True):
        cv_max = cv2.arcLength(ct, True)
        contours_max = ct
cv2.drawContours(I, [contours_max], -1, (0, 0, 255), 3)
cv2.imshow("Contours I",I)
cv2.waitKey(0)