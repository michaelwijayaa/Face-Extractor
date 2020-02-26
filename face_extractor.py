#imports
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import numpy as np

img = cv2.imread('ex3.jpg')
cv2.imshow('image',img)
cv2.waitKey(0)

faces, confidences = cv.detect_face(img)

id = 0

# apply face detection
face, conf = cv.detect_face(img)

padding = 20

# loop through detected faces
for f in face:

    (startX,startY) = max(0, f[0]-padding), max(0, f[1]-padding)
    (endX,endY) = min(img.shape[1]-1, f[2]+padding), min(img.shape[0]-1, f[3]+padding)
    
    # draw rectangle over face
    # cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)

    face_crop = np.copy(img[startY:endY, startX:endX])

    # apply gender detection
    (label, confidence) = cv.detect_gender(face_crop)

    # print(confidence)
    # print(label)

    conf = np.argmax(confidence)
    label = label[conf]

    label = "{}: {:.2f}%".format(label, confidence[conf] * 100)
    print(label)
    Y = startY - 10 if startY - 10 > 10 else startY + 10

    # cv2.putText(img, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
    #             0.7, (0, 0, 0), 2)
    save_crop = np.copy(img[startY-20:endY+padding, startX-20:endX+padding])
    namefile = label[0:4]
    percent = label[6:11]
    if namefile == 'fema':
        namefile = label[0:6]
        percent = label[8:13]
    
    if namefile == 'male':
        cv2.imwrite('../faceextractor/male/'+namefile+'_'+percent+'_{}.png'.format(id), save_crop)
    else:
        cv2.imwrite('../faceextractor/female/'+namefile+'_'+percent+'_{}.png'.format(id), save_crop)
    id = id + 1

# cv2.imshow('image',img)
# cv2.waitKey(0)