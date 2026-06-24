from ultralytics import YOLO

# 1. Load a pre-trained YOLOv8 classification model
model = YOLO("yolov8n-cls.pt")

# 2. Start training on the Masked + CLAHE Dataset
results = model.train(
    # Path to your Method 2 dataset
    data=r"D:\Y4 FYP file\Dataset\FYP_Data\Method1_Raw",
    epochs=100,
    imgsz=224,
    project="Banana_Comparison_50&100epoch",
    name="Model1_Raw_50epoch"   # Different name for this run
)

