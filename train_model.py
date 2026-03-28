from ultralytics import YOLO

def main():
    print("Starting YOLOv8n training for Algerian License Plates...")
    
    # Load a pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # Start training
    # using 15 epochs for quick training, could increase based on performance
    results = model.train(
        data=r"c:\Users\Mohamed\Desktop\projects\one\yolo_dataset\dataset.yaml", 
        epochs=20, 
        imgsz=640,
        batch=16,
        name="license_plate_det3",
        workers=2  # Prevents excessive multiprocessing issues on Windows
    )
    
    print("Training finished!")

if __name__ == '__main__':
    main()
