from yoksul.social_distancing_config import NMS_THRESH
from yoksul.social_distancing_config import MIN_CONF
import numpy as np
import cv2

def detect_people (frame,net,ln,personIdx=0):
    #framlerin boyutlarını aldık ve listeyi başlatmak için sonunda sonuca kaydet
    (H,W) = frame.shape[:2]
    results=[]
    #giriş framiyle bir blob oluşturduk ve yolo ile opencv ile nesne tespiti yapmak için.
    blob = cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True , crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    #tespit edilen  kutular,kütle merkezi,nesne tespiti güvenirliği.
    boxes = []
    centroids = []
    confidences = []

    #layeroutpular için döngüler

    for output in layerOutputs:
        for detection in output:
            #sınıf kimliğini ve güveni cıkarttık ve mevcut nesne algılama sayısını.
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            #mevcut tespitin kişi oldugunu ve min güvenliğin sağlaması
            if classID == personIdx and confidence > MIN_CONF:
                #sınırlayıcı kutu koordinatlarını hesaplıyoruz ve ardından sınırlayıcı kutunun merkezini (yaniağırlık merkezini)
                box = detection[0:4] * np.array([W,H,W,H])
                (centerX,centerY,width,height)= box.astype("int")
                # sınırlayıcı kutu kordinatlarını kullanarak nesnenin sol üst kordinatlarını türettik.
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                #listemizi güncelledik (kutuları,agırlık merkezlerini ve güvenirliğimizi
                boxes.append([x,y,int(width), int(height)])
                centroids.append((centerX,centerY))
                confidences.append(float(confidence))


    #zayıf ve üst üste sınırlayıcı kutular için maksimum olmayan bastırma uyguladık (opencv içerisinde )
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    #en az bir algılamanın oldugundan emın olmak için
    if len(idxs) > 0 :
        #tuttugumuz diziler üzerinde döngü
        for i in idxs.flatten ():
            #sınırlayıcı kutunun kordinatları
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            #sonuç listemizi kişiden kaynaklı güncelledik
            #tahmin olasılığı, sınırlayıcı kutu kordinatları ve agırlık merkezi olarak.
            r = (confidences[i], (x,y,x+w,y+h),centroids[i])
            results.append(r)
    #sonucların listesini döndürdük
    return results




