#%%
import cv2
import pytesseract
###############################
#%%
frameWidth = 640
frameHeight = 480
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
NumPlateCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
#NumPlateCascade = cv2.CascadeClassifier(r'C:\Users\Abdul Basit Aftab\Desktop\LicensePlateDetectinhaarcascade_russian_plate_number.xml')
minArea = 500
color = (0,255,255)

#%%
##################################
cap = cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,150)
while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberplates = NumPlateCascade.detectMultiScale(img, 1.1, 4)  # the scond one is the
    for (x, y, w, h) in numberplates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img,"Number Plate Detected",(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
            ImgPlate = img[y:y+h,x:x+w]
            text = pytesseract.image_to_string(ImgPlate, lang='eng', boxes=False,
                                               config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            print(text)
            cv2.imshow('Number Plate', ImgPlate)
    cv2.imshow('Result',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()