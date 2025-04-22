import facedetection
import cv2
import os

os.environ["DISPLAY"] = ":0.0"

model_path = "model/yunet_n_320_320.nb"
picture_path = "image/sample.jpg"
# picture_path = "image/musk.jpg"
output_path = ".result.jpg"


# 检测图片
yolo = facedetection.FACEDETECTION(model_path)
print(f"model: {model_path}")
boxes = yolo.detect(picture_path)
print(f"boxes: {boxes.__len__()}")
for i in boxes:
    print(
        "{:f} ({:4d},{:4d}) ({:4d},{:4d}) {:s}".format(
            i.reliability,
            i.left_x,
            i.left_y,
            i.right_x,
            i.right_y,
        )
    )

# 到图上画框
img = cv2.imread(picture_path)
for box in boxes:

    label = str(box.reliability)
    (label_width, label_height), bottom = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1,
    )
    cv2.rectangle(
        img,
        (box.left_x, box.left_y),
        (box.right_x, box.right_y),
        (255, 255, 0),
        2,
    )
    cv2.rectangle(
        img,
        (box.left_x, box.left_y - label_height * 2),
        (box.left_x + label_width, box.left_y),
        (255, 255, 255),
        -1,
    )
    cv2.putText(
        img,
        label,
        (box.left_x, box.left_y - label_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )
cv2.imwrite(output_path, img)

# 创建一个窗口并设置为全屏模式
# cv2.namedWindow("result", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.imshow("result", img)
cv2.waitKey(1)
input("按回车退出")
