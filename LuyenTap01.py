import cv2
import numpy as np
import matplotlib.pyplot as plt

#Hiện thị ảnh bằng hàm ỉmread
I=cv2.imread('anh2.jpg')
cv2.imshow("Hien thi anh goc",I)
cv2.waitKey(0) #Muc dich de anh dung lai

#Cau1.  Chuyển ảnh sang biểu diễn HSV được ma trận Ihsv. Hiển thị kênh H của Ihsv
Ihsv=cv2.cvtColor(I,cv2.COLOR_BGR2HSV) # OpenCV: BGR:0->B,1->G,2->R Hoac HSV: 0->H,1->S, 2->V
cv2.imshow("Cau1. Hien thi kenh h cua Ihsv", Ihsv[:,:,0]) # HSV: 0->H,1->S, 2->V =>(hsv_h, hsv_s, hsv_v)
cv2.waitKey(0)

#Cau2. Cân bằng histogram của kênh V của Ihsv. Hiển thị kênh V được cân bằng.
Ihsv[:,:,2]=cv2.equalizeHist(Ihsv[:,:,2]) #Cân bằng histogram là cân bằng lại mức cường độ sáng, tức chỉ là 1 trong 3 chanel của hệ màu HSV.
cv2.imshow("cau 2. Hien thi kenh V can bang", Ihsv[:,:,2])
cv2.waitKey(0)
#histogram là biểu đồ tần xuất được dùng để thống kê số lần xuất hiện các mức sáng trong ảnh.
#Hệ màu HSV bao gồm 3 chanel: H-HUE: giá trị màu,S-SATURATION: độ bảo hòa.,V- VALUE: độ sáng của màu sắc.

#Cau3. Thay đổi kênh V của Ihsv thành kênh V đã cân bằng. Chuyển Ihsv về biểu diễn RGB được ảnh I2. Hiển thị I2.
I2=cv2.cvtColor(Ihsv, cv2.COLOR_HSV2BGR) #Phương thức cv2.cvtColor () được sử dụng để chuyển đổi một hình ảnh từ không gian màu này sang không gian màu khác.
cv2.imshow("Cau3: Bieu dien BGR", I2)
cv2.waitKey(0)
# Sử dụng phương thức cv2.cvtColor () # Sử dụng không gian màu cv2.COLOR_BGR2HSV # mã chuyển đổi
#Chuyển đổi giữa các không gian màu:
#OpenCV hỗ trợ phép chuyển đổi ảnh giữa một số cặp không gian màu bằng hàm cvtColor():cv2.cvtColor(src, code[, dst[, dstCn]])
#src: Đây là hình ảnh có không gian màu được thay đổi.
#code: mã chuyển không gian màu.
#dstCn: số kênh của ảnh đầu ra, mặc định là 0 và sẽ được lấy tự động từ số kênh của ảnh đầu vào.
#dst: Là hình ảnh đầu ra có cùng kích thước và độ sâu với hình ảnh src. Nó là một tham số tùy chọn.

#Cau4.  Ghi ma trận ảnh I2 thành file ảnh anh1.png
cv2.imwrite("anh1.png",I2)
print('Successfully saved')
#Để lưu hình ảnh vào bộ nhớ cục bộ bằng Python, hãy sử dụng hàm cv2.imwrite () trên thư viện OpenCV. Cú pháp của hàm imwrite () là:
# trong đó đường dẫn là đường dẫn hoàn chỉnh của tệp đầu ra mà bạn muốn ghi mảng hình ảnh. cv2.imwrite () trả về một giá trị boolean.
#cv2.imwrite(filename, image)
# trong do: filename: Một chuỗi đại diện cho tên tệp. Tên tệp phải bao gồm định dạng hình ảnh như .jpg, .png, v.v.
# image: It is the image that is to be saved.
# Return Value: It returns true if image is saved successfully.

#Cau5. Giảm size ảnh xuống 1/2. Hiển thị ảnh đã giảm size.
#Cú pháp của hàm thay đổi kích thước trong OpenCV là:hàm resize
#cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
h=I.shape[0]
w=I.shape[1]
I_moi=cv2.resize(I,(w//4,h//4))
cv2.imshow("Cau5 Hiển thị ảnh đã giảm size.",I_moi)
cv2.waitKey(0)

#Cau 6: Làm trơn ảnh kênh H của Ihsv theo bộ lọc trung bình cộng, kích thước cửa sổ lân cận là 7x7 được ảnh Ih. Hiển thị ảnh Ih.
Ih= cv2.blur(Ihsv[:,:,0],(7,7))#Phương thức cv2.blur () được sử dụng để làm mờ hình ảnh bằng cách sử dụng bộ lọc hộp chuẩn hóa.
cv2.imshow('Cau 6. Anh tron kenh H', Ih)
cv2.waitKey(0)
#cv2.blur(src,ksize)
#src: It is the image whose is to be blurred.
#ksize: A tuple representing the blurring kernel size.

#Cau 7: Nhị phân hóa anh Ih theo ngưỡng Otsu được ảnh nhị phân Ib. Hiển thị ảnh Ib.
nguongotsu, Ib = cv2.threshold(Ih, 0, 255, cv2.THRESH_OTSU)
cv2.imshow('Cau 7 Anh nhi phan Ib', Ib)
cv2.waitKey(0)
# Ảnh nhị phân chỉ chứa hai giá trị 0 hoặc 1 (hoặc 0 và 255 tùy theo quy định của cấu trúc ảnh).
#Theo đó, giá trị 0 sẽ là giá trị ứng với những điểm đen trên ảnh và giá trị 1 (hoặc 255) sẽ là giá trị ứng với những điểm trắng.
#Trong OpenCV, việc nhị phân hóa ảnh được thực hiện rất đơn giản bằng cách gọi hàm cvThreshold(const CvArr *src, CvArr

#Cau8 : Tìm các contour của ảnh Ib. Vẽ các đường contour trên ảnh gốc I. Hiển thị ảnh I.
contours, hierarchy = cv2.findContours(Ib, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(I, contours, -1, (0, 0, 255), 2) #mã màu đỏ
cv2.imshow('Cau 8Cac duong contour tren anh goc', I)
cv2.waitKey(0)
#contour đơn giản chỉ là đường cong khép kín và được biểu diễn bằng danh sách các điểm.
#Tập hợp các điểm này có màu sắc hay cường độ sáng như nhau.
#Contours có khá nhiều ứng dụng thực tế trong xử lý ảnh, đặc biệt là khoanh vùng và nhận diện vật thể.
#Trong  mot bưc hinh thì có nhieuf contours
# hàm findContours():Hàm này đưa ra/tìm kiếm danh sách các contours tìm được
#image: ảnh đầu vào (đơn kênh hoặc ảnh nhị phân)
#contours: danh sách contours tìm được
#mode: cách truy vấn các contours, thông thường mình đặt là CV_RETR_TREE
#method: phương thức ước lượng số đỉnh của từng contour.
# Có 2 phương thức thường dùng nhất là CV_CHAIN_APPROX_NONE (liệt kê toàn bộ các đỉnh của đa giác contour) và CV_CHAIN_APPROX_SIMPLE (hạn chế số đỉnh phải lưu trữ).
#-----------------
#Trong đó:

#image: ảnh đầu vào
##contours: danh sách các contour tìm được từ hàm findContours()
#contourIdx: thư tự của contour cần vẽ trong biến contours
#color: màu vẽ

# Cau 9 Vẽ đường contour có diện tích lớn nhất ở câu 8 trên ảnh gốc I. Hiển thị ảnh I.
max_area = 0.0
cnt_max = []
for cnt in contours:
        if max_area < cv2.contourArea(cnt):
            max_area = cv2.contourArea(cnt)
            cnt_max = cnt
print('Dien tich lon nhat cua anh goc I: ', max_area)
cv2.drawContours (I, [cnt_max], -1, (255,0,255),2) #Mã màu hồng
cv2.imshow('Cau 9 anh co dien tich max',I)
cv2.waitKey(0)

#Cau 10: Tìm kiếm biên theo thuật toán Canny của kênh S của ảnh Ihsv được ảnh nhị phân Ie. Hiển thị ảnh Ie.
##Trong OpenCV, để dùng giải thuật Canny, ta đơn giản chỉ cần 1 lệnh cv2.Canny
Ie = cv2.Canny(Ihsv[:, :, 1], 0, 255)
cv2.imshow(' Cau 10 Bien cua kenh S theo thuat toan Canny', Ie)
cv2.waitKey(0)
# HSV: 0->H,1->S, 2->V =>(hsv_h, hsv_s, hsv_v)
#Theo đó, giá trị 0 sẽ là giá trị ứng với những điểm đen trên ảnh và giá trị 1 (hoặc 255) sẽ là giá trị ứng với những điểm trắng.

#Cau11. Kiểm tra pixel có tọa độ dòng y=312, cột x=279 có là điểm biên của kênh S của ảnh Ihsv theo phép dò biên Canny không?
if Ie[312][279] == 255:
    print('Pixel có tọa độ dòng y=312, cột x=279 có là điểm biên của kênh S của ảnh Ihsv theo phép dò biên Canny')
else:
    print('Không phải')

#Cau 12. Hiển thi các giá trị mức xám của kênh S của ảnh Ihsv trong lân cận cửa sổ 5x5 của pixel có tọa độ dòng y=312, cột x=279.
y=312#toa do dong y
x=279 #toa do dong x
Is=Ihsv[:,:,1]## HSV: 0->H,1->S, 2->V
for k in range(-2, 3): #lan can (5, 5)
   for l in range(-1,2):
      if ((y+k) >= 0) & ((y+k)<=h-1) & ((x+l) >= 0) & ((x+l)<=w-1):
          print(Is[y+k,x+l])

#Cau 13.Chuyển ảnh I sang ảnh grayscale theo công thức biến đổi bộ mầu (r,g,b) về mức xám=0.39*r+0.50*g+0.11*b, được ảnh Ig. Hiển thị ảnh Ig.

#Đổi màu BGR sang gray(ảnh nhị phân)
#Ig=cv2.cvtColor(I,cv2.COLOR_BGR2BGRA)#Chuyển đổi giữa các không gian màu:
#OpenCV hỗ trợ phép chuyển đổi ảnh giữa một số cặp không gian màu bằng hàm cvtColor():cv2.cvtColor(src, code[, dst[, dstCn]])
#code: mã chuyển không gian màu.
#dstCn: số kênh của ảnh đầu ra, mặc định là 0 và sẽ được lấy tự động từ số kênh của ảnh đầu vào.
h=I.shape[0]#do cao anh
w=I.shape[1]#do rong anh
#Chuyen anh ve muc xam
Ig= np.zeros((h,w),dtype='uint8') #Tạo một ảnh có chiều cao và chiều rộng
for i in range(h):
  for j in range(w):
    #gray=0.39*r+0.50*g+0.11*b
    d=39*int(I[i][j][2])+ 50*int(I[i][j][1])+ 11*int(I[i][j][0])# OpenCV: BGR:0->B,1->G,2->R
    d=d//100
    Ig[i][j]=d
cv2.imshow('Cau 13. Anh gray theo cong thuc bien doi', Ig)
cv2.waitKey(0)

#Cau 14. Tìm kiếm biên theo thuật toán Canny của Ig được ảnh nhị phân Ie2. Hiển thị ảnh Ie2.
Ie2 = cv2.Canny(Ig, 0, 255) #tuong tu cau 10.
cv2.imshow('Cau 14 .Bien cua Ig theo thuat toan Canny', Ie2)
cv2.waitKey(0)

#Cau15 Hiển thị giá trị đô cao, độ rộng của ảnh Ig.
(h, w, c) = I.shape  #Height, Width, Channel (gọi tắt HWC).
print("Chieu cao cua anh img la:{}\nChieu rong cua anh la:{}".format(h, w, c))
#Lệnh img.shape để lấy ra kích thước của mảng này với h, w, d lần lượt là chiều cao, chiều rộng, độ sâu của bước ảnh.
# Với ảnh có màu thì độ sau thường là 3, ảnh đen trắng là 1.

#Cau 16: Xác định ma trận gradient theo hướng x của ảnh Ig với phương pháp Sobel được ảnh IgradientX. Hiển thi IgradientX.
sobelx = cv2.Sobel(Ig, cv2.CV_64F, 1, 0, ksize=5)
plt.subplot(2, 2, 1), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.show()

#Cau 17 Xác định ma trận gradient theo hướng y của ảnh Ig với phương pháp Sobel được ảnh IgradientY. Hiển thi IgradientY.
sobely = cv2.Sobel(Ig, cv2.CV_64F, 0, 1, ksize=5)
plt.subplot(2, 2, 2), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

#Cau18: Xác định ma trận gradient của ảnh Ig theo cả 2 hướng y và hướng x với phương pháp Sobel được ảnh Igradient=(IgradientX+IgradientY)//2. Hiển thi Igradient.
wIg = Ig.shape[1]
hIg = Ig.shape[0]
Igradient = np.zeros((h, w), dtype= 'float64')
for i in range (hIg):
    for j in range (wIg):
        t1 = sobelx[i][j]*sobelx[i][j]
        t2 = sobely[i][j]*sobely[i][j]
Igradient[i][j]= np.sqrt(t1+t2)
plt.subplot(2, 2, 3), plt.imshow(Igradient, cmap='gray')
plt.title('Igradient'), plt.xticks([]), plt.yticks([])
plt.show()

#Cau 19 Xác định ảnh biên của ảnh Ig sử dụng phương pháp dò biên Sobel và ma trận gradient Igradient với ngưỡng quyết định điểm biên nguong_bien=30, được ảnh nhị phân Ie3. Hiển thị ảnh Ie3.
nguong = 30
Ie3 = np.zeros((h, w), dtype='uint8')
for i in range (h):
    for j in range (w):
       if Ig[i][j] >= nguong:
          Ie3[i][j]=255
       else:
           I[i][j]=0
cv2.imshow("Cau 20. bien cua anh Ig", Ie3)
cv2.waitKey(0)

 #Cau 21 Kiểm tra pixel có tọa độ dòng y=312, cột x=279 có là điểm biên của ảnh Ig theo phép dò biên Sobel trên không?
if Ig[312][279] == 255:
    print('Pixel có tọa độ dòng y=312, cột x=279 có là điểm biên của  ảnh Ig theo phép dò biên Sobel')
else:
    print('Không phải')

#Cau 22 Giãn mức xám của kênh V của ảnh Ihsv lên khoảng tối đa [0,255] được ảnh Iv2. Hiển thị ảnh Iv2.
Iv = Ihsv[:, :, 2]
def gianmucxamtuyentinh(Iv):
    a=np.min(Iv)
    b=np.max(Iv)
    aLUT=np.zeros(256,dtype='uint8')
    for g in range(0,255+1):
        aLUT[g]=(255*(g-a))//(b-a)
    for i in range(Iv.shape[0]):
        for j in range(Iv.shape[1]):
            Iv[i,j]=aLUT[Iv[i,j]]
    return Iv
Iv2 = gianmucxamtuyentinh(Iv)
cv2.imshow('Cau 22.Gian muc tuyen tinh kenh V', Iv2)
cv2.waitKey(0)

#Cau 23  Giãn mức xám của kênh S của ảnh Ihsv lên khoảng tối đa [0,255] được ảnh Is2. Hiển thị ảnh Is2.
Is1 = Ihsv[:, :, 1]
def gianmucxamtuyentinh(Is1):
    a=np.min(Is1)
    b=np.max(Is1)
    aLUT=np.zeros(256,dtype='uint8')
    for g in range(0,255+1):
        aLUT[g]=(255*(g-a))//(b-a)
    for i in range(Is1.shape[0]):
        for j in range(Is1.shape[1]):
            Is1[i,j]=aLUT[Is1[i,j]]
    return Is1
Is2 = gianmucxamtuyentinh(Is1)
cv2.imshow('Cau 23. Gian muc tuyen tinh kenh S', Is2)
cv2.waitKey(0)
