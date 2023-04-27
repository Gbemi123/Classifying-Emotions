import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from utils import *
from config import *
from model import *
import matplotlib.pyplot as plt



def train_model(model, optim, train_data, epochs):
  '''
  model: the model to be trained
  optim: The optimizer chosen in this case Adam
  train_data: training data
  '''

  #training sequence
  epoch_loss = []
  epoch_accuracy =[]

  for epoch in range(epochs):

    for img, label  in train_data:
      #transforming images to tensor and resizing 
      # img =prepare_imgs(img) images already transformed to tensor
      img =img.to(device)
      label  = label.to(device)

      #resetting gradients
      optim.zero_grad()

      #running img through model
      output = model(img)
        

      #calculate loss
      #cross entropy loss a reducing loss is good as the closer to zero the more accurate
      _loss = calc_loss(output, label)
      loss =_loss.item()

      #Calclulate accuracy
    
      acc = accuracy(output, label)
      

      #loss step backwards 
      _loss.backward()
      optim.step()

    # results 
    epoch_loss.append(_loss.detach().cpu().numpy())
    epoch_accuracy.append(acc.detach().cpu().numpy())
    print( 'Epoch: {}'.format(epoch+1), 'loss: {:.4f}'.format(loss), 'accuracy: {:.2f}'.format(acc) )

  return epoch_loss, epoch_accuracy



# training the model 

# loss, acc= train_model(model, optim, train_loader, epochs)


#torch.save model
# torch.save(model, 'Faces_model.pth')

class_names = ['anger','contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'suprise']
def predict_faces(img, model):
  # define image
  x =img.unsqueeze(0).to(device)

  preds =model(x)
 
  _, label_idx = torch.max(preds, dim =1)
  

  #get class label
  return class_names[label_idx[0]]

rows = 2
cols = 5
axes = []
fig =plt.figure(figsize=(14,6))

for i in range(rows*cols):
 
  img,label = test_set[i]
  axes.append(fig.add_subplot(rows, cols, i+1))
  sub_title = ('Label: {}, Pred: {}'.format(class_names[label], predict_faces(img, model)))
  axes[-1].set_title(sub_title, fontsize=12)
  plt.imshow(img.clip(0,1).permute(1,2,0), cmap='gray')
  plt.axis('off')
  
  
fig.tight_layout()
plt.show()
