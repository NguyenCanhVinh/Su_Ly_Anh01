import cv2
import matplotlib.pyplot as plt
import numpy as np

#1. Hiện thị ảnh I
I=cv2.imread('hat1.PNG')
cv2.imshow("Cau1 Hien thi anh I", I)
cv2.waitKey(0)

#2 Chuyển ảnh sang ảnh HSV sang ma trận Ihsv. Hiện thị kênh H của Ihsv. Xác định giá trị mức sánh lớn nhất của kênh S cảu ảnh Ihsv
Ihsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
cv2.imshow(" Cau 2 Hien thi kênh H của Ihsv.", Ihsv[:, :, 0])
cv2.waitKey(0)
print("Nức xám lớn nhất của kênh S của ảnh Ihsv", np.max(Ihsv[:, :, 1]))

#3 Xác định và vẽ Histogram của kênh V cảu ảnh Ihsv.
histogram = cv2.calcHist(Ihsv[:, :, 2], channels=[2], mask=None, histSize=[256], ranges=[0, 256])
plt.plot(histogram)
plt.title("Histogram kênh V")
plt.show()

#4 Làm trơn ảnh kênh S của Ihsv theo bộ lọc median, kích thước cửa sổ lần lượt là 5X5 được ảnh Is. Hiện thị ảnh Is
Is = cv2.medianBlur(Ihsv[:, :, 1],5)
cv2.imshow("cau 4 Hien thi anh Is ", Is)
cv2.waitKey(0)
#5 Xác định đường contour có chu vi lớn nhất của ảnh Ib. Vẽ đường contour trên ảnh gốc I. Hiện thị ảnh I.
#thresh , Ib = cv2.threshold(Is, 0, 255, cv2.THRESH_OTSU)
#cv2.imshow("Ib", Ib)
#contours, _ = cv2.findContours(Ib, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#max = 0.0
#contours_max = []
#for contour in contours:
#    if cv2.arcLength(contour, True) > max:
#        max = cv2.arcLength(contour, True)
#        contours_max = contour
#cv2.drawContours(I, [contours_max], -1, (0,0,255), 2)
#cv2.imshow("Cau 5 đường contour trên ảnh gốc I", I)
#cv2.waitKey(0)
# 5
thresh, Ib = cv2.threshold(Is, 0, 255, cv2.THRESH_OTSU)
cv2.imshow('Anh nhi phan Ib', Ib)
cv2.waitKey(0)
# 5.2 Tìm các contour của ảnh Ib.
contours, con = cv2.findContours(Ib, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(I, contours, -1, (0, 255, 0), 2) #mã màu xanh
#cv2.imshow('Cac duong contour tren anh goc', I)
#cv2.waitKey(0)
# 5.3 Vẽ đường contour có chu vi  lớn nhất ở câu trên ảnh gốc I. Hiển thị ảnh I.
max = 0.0
contours_max = []
for contour in contours:
    if cv2.arcLength(contour, True) > max:
        max = cv2.arcLength(contour, True)
        contours_max = contour
cv2.drawContours(I, [contours_max], -1, (0,0,255), 2)
cv2.imshow("Cau 5 đường contour trên ảnh gốc I", I)
cv2.waitKey(0)