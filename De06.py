import cv2
import numpy as np

# 1
I = cv2.imread("anh5.jpg")
cv2.imshow("I", I)
cv2.waitKey(0)
# 2
Ihsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
print("Muc sang lon nhat cua kenh S:", np.max(Ihsv[:, :, 1]))
cv2.imshow("Ihsv kenh H", Ihsv[:, :, 0])
cv2.waitKey(0)
# 3
Is = cv2.blur(Ihsv[:, :, 1], (7, 7))
cv2.imshow("Is lam tron", Is)
cv2.waitKey(0)
# 4
thresh, Ib = cv2.threshold(Is, 0, 255, cv2.THRESH_OTSU)
cv2.imshow("Ib", Ib)
cv2.waitKey(0)

contours, con = cv2.findContours(255-Ib, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
max = 0.0
contours_max = []
for contour in contours:
    if cv2.arcLength(contour, True) > max:
        max = cv2.arcLength(contour, True)
        contours_max = contour
print("chu vi max:", max)
cv2.drawContours(I, [contours_max], -1, (0,0,255), 2)
cv2.imshow("Cau 4 đường contour trên ảnh gốc I", I)
cv2.waitKey(0)
# 5
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
I = cv2.cvtColor(Ihsv, cv2.COLOR_HSV2BGR)
# cv2.imshow("gian muc xam kenh V", Ihsv[:, :, 2])
cv2.imshow("bien doi Ihsv ve RGB", I)


cv2.waitKey(0)