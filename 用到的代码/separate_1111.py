import os
import cv2
 
 #裁剪图片
 
def main():
    # 图片路径
    img_path = '/home/qya/YOLO_Projects/Instrument/data/111_out/predict'
    # .txt标签文件路径，也就是图像的边框
    label_path = '/home/qya/YOLO_Projects/Instrument/data/111_out/predict/labels' 
    # 保存路径，裁剪之后的结果保存到哪个位置
    save_path = '/home/qya/YOLO_Projects/Instrument/data/111_sep'
    os.makedirs(save_path, exist_ok=True)  # 确保输出目录存在  
 
    img_total = []
    label_total = []
    imgfile = os.listdir(img_path)
    labelfile = os.listdir(label_path)
 
    for filename in imgfile:
        name, type = os.path.splitext(filename)
        if type in ['.jpg', '.png']:
            img_total.append(name)
    for filename in labelfile:
        name, type = os.path.splitext(filename)
        if type == '.txt':
            label_total.append(name)
 
 
 
    for _img in img_total:
        if _img in label_total:
            # 获取图片的名字.jpg
            filename_img = _img + '.jpg'
            # 获取图片的路径，具体到图片名字.jpg
            path = os.path.join(img_path, filename_img)
            # 读取图片，结果为三维数组
            img = cv2.imread(path)  
            filename_label = _img + '.txt'
            w = img.shape[1]  # 图片宽度(像素)
            h = img.shape[0]  # 图片高度(像素)
            n = 1
            # 打开文件，编码格式'utf-8','r+'读写
            with open(os.path.join(label_path, filename_label), "r", encoding='utf-8', errors="ignore") as f:
                for line in f:
                    msg = line.split(" ")  # 根据空格切割字符串，最后得到的是一个list
                    x1 = int((float(msg[1]) - float(msg[3]) / 2) * w)  # x_center - width/2
                    y1 = int((float(msg[2]) - float(msg[4]) / 2) * h)  # y_center - height/2
                    x2 = int((float(msg[1]) + float(msg[3]) / 2) * w)  # x_center + width/2
                    y2 = int((float(msg[2]) + float(msg[4]) / 2) * h)  # y_center + height/2
                    filename_last = _img + "_" + str(n) + ".jpg"
                    print(f"✓ 已保存: {filename_last}")
                    img_roi = img[y1:y2, x1:x2] # 剪裁，roi:region of interest
                    cv2.imwrite(os.path.join(save_path, filename_last), img_roi)
                    n = n + 1
        else:
            continue
 
if __name__ == '__main__':
    main()