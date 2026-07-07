from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt') 

    # 2. Mulai proses training
    model.train(
        data='Multimedia_S6/dataset/data.yaml', 
        epochs=20,                
        imgsz=640,                
        device='cpu',
        workers=4                 
    )
if __name__ == '__main__':
    main()