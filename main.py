import cv2

img = cv2.imread('../memeSolver/images/cat.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
faces = cv2.CascadeClassifier('catfaces.xml')
results = faces.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)
for (x, y, w, h) in results:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)


cv2.imshow('cat', img)
cv2.waitKey(0)
