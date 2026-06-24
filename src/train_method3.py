from ultralytics import YOLO

# 1. Load a pre-trained YOLOv8 classification model
model = YOLO("yolov8n-cls.pt")

# 2. Start training on the Isolated + CLAHE Dataset
results = model.train(
    # Path to your Method 3 dataset
    data=r"D:\Y4 FYP file\Dataset\FYP_Data_Final\Method3_Isolated_CLAHE_New2",
    epochs=100,
    imgsz=224,
    project="Banana_Comparison_50&100epoch",
    name="Model3_Isolated_CLAHE_New2_100epoch"  # Different name for this run
)
