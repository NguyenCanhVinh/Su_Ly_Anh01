import matplotlib.pyplot as plt
import numpy as np
import cv2
#1 Hiện thị tỉ lệ giữa giá trị độ cao và độ rộng của ảnh I.
I=cv2.imread('I04.jpg')
cv2.imshow('Cau 1 anh dau vao',I)
print("ti le giua do rong va do cao  cua anh I:", I.shape[0]/I.shape[1])
cv2.waitKey(0)

#2 Hiểu chỉnh lại ảnh I với size mới là độ cao 256, độ rộng 256 được ảnh mới I2. Hiện thị ảnh I2.
I2=cv2.resize(I,(256,256))#do rong,do cao
cv2.imshow('Cau 2 anh co gian',I2)
cv2.waitKey(0)

#3 Chuyển đổi ảnh I sang ảnh HSV được ma trân Ihsv. Hiển thi kênh S của ahr Ihsv.
Ihsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
cv2.imshow("Cau 3 Kenh S cua anh Ihsv", Ihsv[:, :, 1])
cv2.waitKey(0)

#4 Xác định biên theo phương pháp Canny của kênh V của ảnh Ihsv được ảnh nhị phân Ivb
Ivb = cv2.Canny(Ihsv[:, :, 2], 0, 255)
cv2.imshow("Cau 4 Ivb", Ivb)
cv2.waitKey(0)

# 5 Xác định histogram của kênh S của ảnh Ihsv
histogram = cv2.calcHist(Ihsv[:, :, 1], channels=[2], mask=None, histSize=[256], ranges=[0, 256])
plt.plot(histogram)
plt.title("Cau 5 Histogram kênh S")
plt.show()
cv2.waitKey(0)


