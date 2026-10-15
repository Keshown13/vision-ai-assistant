from ultralytics import YOLO

# 加载YOLOv8最小的预训练模型，训练速度最快
model = YOLO('yolov8n.pt')

# 执行训练：用官方自带的小数据集，训练3轮，批次设小保证流畅
if __name__ == '__main__':
    results = model.train(
        data='coco8.yaml',
        epochs=3,
        batch=4,
        imgsz=640
    )
    print("✅ YOLO训练完成！结果在桌面 runs/detect/train 文件夹里")
