import  cv2
import numpy as np
import matplotlib.pyplot as plt

#1
I=cv2.imread('watch.jpg')
cv2.imshow("I", I[:, :, 0])
cv2.waitKey(0)

#2
Ihsv=cv2.cvtColor(I,cv2.COLOR_BGR2HSV) # OpenCV: BGR:0->B,1->G,2->R Hoac HSV: 0->H,1->S, 2->V
cv2.imshow(" Hien thi kenh h cua Ihsv", Ihsv[:,:,0])
print("mức sáng trung bình của kênh V", np.mean(Ihsv[:, :, 0]))
cv2.waitKey(0)
#3
Is = cv2.blur(Ihsv[:, :, 1], (3, 3))
cv2.imshow('Lam tron kenh S', Is)
cv2.waitKey(0)
#4
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
#5
def gianmucxam(Igray):
    max_Igray = np.max(Igray)
    min_Igray = np.min(Igray)

    I_gray = np.zeros(256, dtype='uint8')
    # giãn mức maxxxx
    for i in range(0, 256):
        I_gray[i] = (255 * (i - min_Igray)) // (max_Igray - min_Igray)
    for u in range(0, Igray.shape[0]):
        for v in range(0, Igray.shape[1]):
            Igray[u][v] = I_gray[Igray[u][v]]
    return Igray


Ihsv[:, :, 2] = gianmucxam(Ihsv[:, :, 2])
# cv2.imshow("gian muc xam kenh V", Ihsv[:, :, 2])
I = cv2.cvtColor(Ihsv, cv2.COLOR_HSV2RGB)
cv2.imshow("bien doi Ihsv ve RGB", I)
cv2.waitKey(0)