import cv2
import glob
import numpy as np

def main():
    img_list = glob.glob('./kojiharu_raw/*')
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    output_path = './kojiharu_face'
    index = 0
    for img_name in img_list:
        img = cv2.imread(img_name)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x,y,w,h) in faces:
                face = img[y:y+h, x:x+w]
                cv2.imwrite('./kojiharu_face/%d.jpg'%(index), face)
                index += 1
        except:
            print("invalid image")
        print('img processed')

if __name__ == '__main__':
    main()

