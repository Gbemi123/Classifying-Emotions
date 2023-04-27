
import streamlit as st
import torch.nn as nn
from PIL import Image, ImageOps
import torch 
import torchvision
from config import *
from utils import extract_faces


st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation= True)

def load_model():

  model = torch.load('Faces_model.pth', map_location=torch.device('cpu'))

  model=model.to('cpu')

  return model

device= torch.device('cpu')

with st.spinner('Model is being loaded..'):

  model = load_model()
  model.to(device);
    

st.write("""

        # Emotions Classification

""")
file = st.file_uploader('Please upload an image of a face', type=['jpg', 'png'])

def predict_expression (img, model):

    
    # define image
    prep = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.CenterCrop((300,300)),
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),  # Trnsfor to tensor
        torchvision.transforms.Normalize((0.5) ,(0.5)) # Normalise to get data in range
    ])
  
    img = prep(img).to(device)



    # define image
    x =img.unsqueeze(0)

    preds =model(x)
    
    _, label_idx = torch.max(preds, dim =1)
    

    #get class label
    return label_idx[0]


if file is None:
   st.text('please upload an image')

else:
   
   image=Image.open(file)
   st.image(image, use_column_width= True)


   prediction = predict_expression(extract_faces(image), model)

   class_names = ['angry','contemptful', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'suprised']

   string = 'You look... ' + class_names[prediction]
   st.success(string)






