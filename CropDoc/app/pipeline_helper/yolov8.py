# from ultralytics import YOLO
# model = YOLO('yolov8n-cls.pt')
# model.train(data='classify', epochs=10, imgsz=64, batch=512)

import sys
from dataset import CropCCMTDataset

from ultralytics import YOLO, checks, hub
checks()

hub.login('fc32f0556a2215712769d62a84dcc06cbb5635077b')

dataset_dir = '../../../../yolov5/datasets/classify'


# y_model = 'https://hub.ultralytics.com/models/uu2o2jCUUozCt7grEZ7l'
# y_model = 'yolov8n-cls.pt'
y_model = 'runs/classify/train22/weights/best.pt'
y_config = 'yolov8.yaml'

def train():
    model = YOLO(y_model)
    
    results = model.train(epochs=10, imgsz=64, batch=512)

def classify():
    # model = YOLO(y_config)
    model = YOLO(y_model)
    # model.train(data='classify', epochs=10, imgsz=64, batch=512)   
    model.train(
        data=dataset_dir, 
        epochs=10, 
        imgsz=160,
        batch=128
        )
    
def predict():
    model = YOLO(y_model)
    model.predict(
        source='test/', 
        save=True, 
        imgsz=160, 
        batch=128
        )
    
def validate():
	model = YOLO(y_model)
	model.val(
	data=dataset_dir,
	save=True,
	batch=128
	)    

if __name__ == '__main__':
    func = sys.argv[1]
    if func == 'train':
        print('Launching train!')
        train()
    elif func == 'classify':
        print('Launching classify!')
        classify()
    elif func == 'predict':
        print('Launching predict!')
        predict()        
    elif func == 'validate':
        print('Launching validate!')
        validate() 

