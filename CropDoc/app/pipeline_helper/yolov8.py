# from ultralytics import YOLO
# model = YOLO('yolov8n-cls.pt')
# model.train(data='classify', epochs=10, imgsz=64, batch=512)

import sys
# from dataset import CropCCMTDataset

from ultralytics import YOLO, checks, hub
checks()

hub.login('fc32f0556a2215712769d62a84dcc06cbb5635077b')

dataset_dir = '../../../../yolov5/datasets/classify'

model = {
    'hub': 'https://hub.ultralytics.com/models/uu2o2jCUUozCt7grEZ7l',
    'local': 'runs/classify/train22/weights/best.pt',
}

config = {
    'custom': 'yolov8.yaml'
}
# y_model = 'https://hub.ultralytics.com/models/uu2o2jCUUozCt7grEZ7l' # last hub model trained
# y_model = 'yolov8n-cls.pt'  # pre-trained yolov8 model
# y_model = 'runs/classify/train22/weights/best.pt'  # last local trained model based on yolov8 pretrained model
# y_config = 'yolov8.yaml'  # yolov8 config

y_model = model['local']

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
	batch=128,
    imgsz=160,
    split='test'
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

