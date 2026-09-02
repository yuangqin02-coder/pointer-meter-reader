from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("../yolo最新权重文件/best.pt")

# 预测 
results = model.predict(
    source="/home/qya/YOLO_Projects/Instrument/data/111",  # 图片/文件夹/视频路径 - 请改为你自己的图片文件夹
    imgsz=640,          # 图片尺寸
    conf=0.55,          # 置信度阈值
    save=True,           # 保存结果到 runs/detect/predict 目录
    save_txt=True,      # 保存保存预测框到labels里面.txt文件
    project="/home/qya/YOLO_Projects/Instrument/data/111_out"  # 结果根目录 - 请改为你自己的输出文件夹
)

print("预测完成，结果已保存。")








