# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	#çerçevenin boyutlarını alın ve ondan bir blob oluşturun

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# blob'u ağ üzerinden geçirin ve yüz algılamalarını elde edin
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# yüz listemizi, ilgili konumlarını ve yüz maskesi ağımızdan tahminlerin listesini başlatın
	faces = []
	locs = []
	preds = []

	# algılamalar üzerinde döngü
	for i in range(0, detections.shape[2]):
		# algılama ile ilişkili güveni yani olasılığı ayıklayın
		confidence = detections[0, 0, i, 2]

		# güvenin minimum güvenden daha büyük olmasını sağlayarak zayıf algılamaları filtreleyin
		if confidence > 0.5:
			#nesne için sınırlayıcı kutunun (x, y) koordinatlarını hesaplayın
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			#sınırlayıcı kutuların çerçevenin boyutlarına uyduğundan emin olun
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# yüz ROI'Yİ ayıklayın, BGR'DEN RGB kanal siparişine dönüştürün, 224x224'e yeniden boyutlandırın ve ön işlem yapın
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# yüz ve sınırlayıcı kutuları ilgili listelerine ekleyin
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# sadece en az bir yüz tespit edilirse bir tahmin yapın
	if len(faces) > 0:

		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# yüz konumlarının ve bunlara karşılık gelen konumların 2-tuple'ını döndürün
	return (locs, preds)

# serileştirilmiş yüz dedektörü modelimizi diskten yükleyin
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# yüz maskesi dedektör modelini diskten yükleyin
maskNet = load_model("mask_detector.model")

# video akışını başlat
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# video akışından kareler üzerinde döngü
while True:
	# çerçeveyi video akışından alın ve maksimum 400 piksel genişliğe sahip olacak şekilde yeniden boyutlandırın
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# çerçevedeki yüzleri algılayın ve yüz maskesi takıp takmadıklarını belirleyin

	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# algılanan yüz konumları ve bunlara karşılık gelen konumlar üzerinde döngü

	for (box, pred) in zip(locs, preds):
		# sınırlayıcı kutuyu ve tahminleri paketinden çıkarın
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# Çizmek için kullanacağımız sınıf etiketini ve rengini belirleyin
		# sınırlayıcı kutu ve metin

		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# olasılığı etikete dahil et
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# çıktıda etiketi ve sınırlayıcı kutu dikdörtgenini görüntüleyin
		# çerçeve
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# çıktı çerçevesini göster
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# "q" tuşuna basıldıysa döngüden çıkın
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()