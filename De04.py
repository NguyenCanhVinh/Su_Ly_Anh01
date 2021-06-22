import cv2
import numpy as np
import matplotlib.pyplot as plt

#1 Hien thi anh I
I=cv2.imread("anh5.jpg")
cv2.imshow("Cau 1. Hien thi anh goc", I)
cv2.waitKey(0)

#Cau2. . Chuyển ảnh mầu I sang ảnh đa cấp xám (grayscale)theo công thức xác định mức độ xám từ tổ hợp các thành phần màu(r,g,b)
# theo tỉ lệ(0.39,0.50,0.11) đc ma trận Ig.Hien thi Ig va Hiển thị giá trị đô cao, độ rộng của ảnh Ig.
h=I.shape[0]#do cao anh
w=I.shape[1]#do rong anh
#Chuyen anh ve muc xam
Ig= np.zeros((h,w),dtype='uint8') #Tạo một ảnh có chiều cao và chiều rộng #cap phat 1 anh gray 8bit
for i in range(h):
  for j in range(w):
    #gray=0.39*r+0.50*g+0.11*b
    d=39*int(I[i][j][2])+ 50*int(I[i][j][1])+ 11*int(I[i][j][0])# OpenCV: BGR:0->B,1->G,2->R
    d=d//100
    Ig[i][j]=d
cv2.imshow('Cau 2. Anh gray theo cong thuc bien doi', Ig)
cv2.waitKey(0)
print("Mức xám lớn nhất: ", np.max(Ig))
#3 Lấy biên của ảnh Ig theo phương pháp Canny được ảnh Ie là ảnh nhị phân nền đen. Kiểm tra pixel có tạo độ y=160, cột x=326 có điểm biên của ảnh Ig theo phép dò biên Canny không?
# 3.1lấy biên của ảnh Ig bằng Candy thành ảnh biên Ie nhịp phân.
Ie = cv2.Canny(Ig, 0, 255)
# cv2.imshow("Ie", Ie)
# 3.2 Kiểm tra pixel có tọa độ dòng y=160, cột x=326 có là điểm biên của ảnh Ig theo phép dò biên Canny không
y = 160
x = 326
if Ie[y][x] == 255:
    print("là điểm biên của phép dò biên Canny")
else:
    print("không là điểm biên của phép dò biên Canny")
#4 Nhị phân hóa anh Ig theo ngưỡng otsu dc ảnh ib.hiển thi Ib.
thresh, Ib = cv2.threshold(Ig, 0, 255, cv2.THRESH_OTSU)
cv2.imshow("Ib", Ib)
cv2.waitKey(0)
# 5.1 Tìm các contour của ảnh Ib.
contours, con = cv2.findContours(Ib, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

max_cv = 0.0
contours_max = []
for ct in contours:
    if max_cv <= cv2.arcLength(ct, True):
        max_cv = cv2.arcLength(ct, True)
        contours_max = ct
cv2.drawContours(I, [contours_max], -1, (0, 255, 255), 2)
cv2.imshow("Anh sau khi ve lan 2", I)

cv2.waitKey(0)
