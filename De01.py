import  cv2
import numpy as np
import matplotlib.pyplot as plt

#Cau1. Hien thi kenh B cua anh I
I=cv2.imread("the_cancuoc_congdan.jpg")
cv2.imshow("Cau1. Hien thi kenh B cua anh I", I[:,:,0])#OpenCV: BGR:0->B,1->G,2->R
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

#Hiển thị giá trị đô cao, độ rộng của ảnh Ig.
(h, w, c) = I.shape  #Height, Width, Channel (gọi tắt HWC).
print("Chieu cao cua anh img la:{}\nChieu rong cua anh la:{}".format(h, w, c))

#Cau3. Chuyển ảnh Ig sang ảnh nhị phân Ib với ngưỡng Otsu. Hiện thị ảnh nhị phân Ib.
nguongotsu, Ib = cv2.threshold(Ig, 0, 255, cv2.THRESH_OTSU)
cv2.imshow('Cau 3 Anh nhi phan Ib', Ib)
cv2.waitKey(0)

#Cau 4 Làm trơn ảnh Ig, theo bộ lọc trung bình cộng và lân cận cửa sổ kích thước 5X5 thu đc Im hiện thị Im.
Im=cv2.blur(Ig,(5,5))
cv2.imshow("Cau4. hiện thị Im.", Im)
cv2.waitKey(0)

#Cau  5. Xác định ảnh Ie của ảnh Ig sử dụng phương pháp độ biên Canny.Hiện thị kết quả Ie
Ie= cv2.Canny(Ig,0,255)
cv2.imshow("Cau5 Su dung pp canny", Ie)
cv2.waitKey(0)

#Cau 6. Xác định các contour cảu ảnh Im và vẽ các contour lên ảnh gốc I ban đầu
contours,hierarchy= cv2.findContours(Ib, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(Ig, contours, -1, (0, 0,255), 2)
cv2.imshow("Cau 6 vẽ các contour lên ảnh gốc I ban đầu", Ig)
cv2.waitKey(0)
#Muon ve dc contour thi phải nhị phân hóa anh gốc
