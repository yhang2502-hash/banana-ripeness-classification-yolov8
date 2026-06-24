from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

results = model.train(
    data=r"D:\Y4 FYP file\Dataset\FYP_Data_Final\Method2_Masked_CLAHE_Fair",
    epochs=250,
    imgsz=224,
    project="Model_Banana_Method_250epoch",
    name="Model2_Masked_CLAHE_Fair_250epoch",
    patience=0
)
