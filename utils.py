import torch
import torch.nn as nn 
import cv2
import PIL.Image as Image
import numpy as np

def accuracy(output, labels):
  _, preds = torch.max(output, dim=1)
  return torch.tensor(torch.sum(preds == labels).item() / len(preds) ) *100


def calc_loss(prediction ,actual):
  loss = nn.CrossEntropyLoss()

  return loss(prediction, actual)


def extract_faces(image):
  img=np.array(image)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
  face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

  for (x,y,w,h) in face:

    face_img= img[y:y+h,x:x+w]

  face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
  face_img=Image.fromarray(face_img)

  return face_img


