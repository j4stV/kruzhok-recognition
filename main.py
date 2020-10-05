import torch
import os
import cv2

print('Preparing the model. Please wait')
PATH = 'model.pt'
model = torch.load(PATH)
model.eval()
path_to_image = input('Path to your image:')

image = cv2.imread(path_to_image)
image /= 255
result = model(image)[0]
if result['score'] >= 0.6:
    print('kruzhok')
