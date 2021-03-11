from yoksul import social_distancing_config as config
from yoksul.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import cv2
import os
import argparse
import imutils

#bağımsız değişlenleri yapılandırıp çözümledik
#uygulamamızı herangi isteğe bağlı videolarda görüntülemek için
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", type=str, default="", help="(isteğe bağlı) giriş video dosyasının yolu")

ap.add_argument("-o", "--output", type=str, default="", help="(istege baglı) çıktı video dosyasının yolu")

ap.add_argument("-d", "--display", type=int, default=1, help="çıktı cercevesinin görüntülenip görüntülenmeyeceği")

args= vars(ap.parse_args())

#yolo sınıfından coco etiketlerimizi ekledik
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#yolo yollarımızı tanımladık
weightPath = os.path.sep.join([config.MODEL_PATH,"yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])


#yolo yollarını kullanarak modeli belleğe bağlayacaz
#coco veri setini(80 sınıf) yolo nesne dedektörümüze yüklüyoruz

print("Yolo diskten yükleniyor... ")
#opencvnin DNN modülünü kullanarak yoloyu yükledik
net = cv2.dnn.readNetFromDarknet(configPath,weightPath)

#gpu yu kullanacaksak
#if config.USE_GPU:
    #CUDA yı tercih edilen hedef doğrultusunda ayarlıyoruz
    #print("[BİLGİ] Tercih edilen arka hedefi CUDA'ya ayarla..")
    #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#yolodan cıktı katmanlarının isimlerini toplar bu bize sonuçlarımıza işlemek için yardımcı olacak
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#video akışımızı ya da kameramızı başlatırız
print("Video akışına erişiliyor...")
vpath=""
a = input("Hazır video kullanmak istiyor musunuz(e/h): ")
if a.lower() == "e":
    vpath=input("Dosya yolunu girin: ")
    print("Video hazırlanıyor...")

#vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
vs = cv2.VideoCapture(vpath if vpath else 0)
writer = None

#video üzeindeki kareler üzerinde döngü başlattık
while True:
    (grabbed,frame) = vs.read()
    if not grabbed:
        break

    #frami yeniden boyutlandırdık sonra içindeki insanları (yalnızca insanları) tespit ettik
    frame = imutils.resize(frame,width=500)
    results = detect_people(frame,net,ln, personIdx=LABELS.index("person"))
    # min sosyalliği ihlal edenlerin kümesi
    violate = set()

    #karedeki insanların arasındaki mesafeleri kontrol etme
    #framde 2 kişi varsa
    if len(results) >= 2 :
        #tüm ağırlık merkezi çiftleri arasında ki öklid mesafesını hesapladık
        centroids = np.array([r[2] for r in results])
        D= dist.cdist(centroids,centroids,metric="euclidean")
        #mesafe matrisinin üst üçgeni üzerinden döngü yaptık
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # bu hesapladığımız mesafe matrisi belirtilen min uzaklığının kontrol
                if D[i,j] < config.MIN_DISTANCE:
                    #mesafeyi ihlal edenler listeye eklenir
                    violate.add(i)
                    violate.add(j)


    for (i,(prob,bbox,centroid)) in enumerate(results):

        (startX, startY,endX,endY) = bbox
        (cX,cY) =centroid
        color = (0,255,0)

        if i in violate:
            color=(0,0,255)


        cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
        cv2.circle(frame,(cX,cY),5,color,1)

    text = "SOSYAL MESAFE IHLALLERI: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25,(frame.shape[1], frame.shape[0]), True)

    if writer is not None:
        writer.write(frame)

