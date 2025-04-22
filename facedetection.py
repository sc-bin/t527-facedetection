import os
import numpy as np
import cv2
from typing import List
import awnn_t527

import time
import threading

tmp_time = 0

image_size = 320


def get_time_ms():
    return int(round(time.time() * 1000))


def get_ms_last():
    # 返回自从上次调用本函数后过去了多久
    global tmp_time
    ret = get_time_ms() - tmp_time
    tmp_time = get_time_ms()
    return ret


anchors = {
    8: [[10, 13], [16, 30], [33, 23]],  # P3/8
    16: [[30, 61], [62, 45], [59, 119]],  # P4/16
    32: [[116, 90], [156, 198], [373, 326]],  # P5/32
}


class YOLO_RESULT:
    left_x: int
    left_y: int
    right_x: int
    right_y: int
    class_index: int
    reliability: float


def desigmoid(x):
    return -np.log(1.0 / x - 1.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class FACEDETECTION(YOLO_RESULT):
    data: bytearray
    has_result = False
    results: List[YOLO_RESULT] = []
    thread_async_running = False

    def __init__(self, path: str):
        """
        初始化
        @path: 模型路径
        """
        self.npu = awnn_t527.awnn(path)

    def thread_async_detect(self):
        time_detect_start = get_time_ms()
        get_ms_last()
        if not self.npu.is_async_running():
            self.npu.run_async(self.img2byte(self.img_image_size))

        while self.npu.is_async_running():
            time.sleep(0.001)
        print(f"\t{get_ms_last()}ms  推理")

        self.results = self.post_process()
        self.thread_async_running = False
        print(f"\t{get_ms_last()}ms  后处理")
        self.has_result = True
        time_detect = get_time_ms() - time_detect_start
        fps = int(1000 / time_detect)
        print(f"\t共计 {time_detect}ms  fps:{fps}\n")

    def pre_process(self, img):
        """图像前处理"""
        # 判断picture的数据类型是str吗
        if isinstance(img, str):
            if not os.path.isfile(img):
                raise FileNotFoundError("文件不存在")
            return self.pre_process(cv2.imread(img))
        if len(img.shape) == 2 or img.shape[2] == 1:
            self.img_raw = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            self.img_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 将图片缩放到image_size*image_size
        self.img_image_size = self.letterbox_image(
            self.img_raw, (image_size, image_size)
        )

    def detect(self, img) -> List["YOLO_RESULT"]:
        """
        检测图片，阻塞直到检测完成，返回检测结果
        @path: 图片路径
        """
        self.detect_async(img)
        while not self.has_result:
            time.sleep(0.001)
        return self.results

    def detect_async(self, img):
        """
        检测图片，立即返回，不阻塞
        @path: 图片路径
        """
        if not self.thread_async_running:
            self.thread_async_running = True
            time_progress_start = get_time_ms()

            self.pre_process(img)
            thread = threading.Thread(target=self.thread_async_detect)
            thread.start()
            time_progress_end = get_time_ms()
            print(f"\t{time_progress_end-time_progress_start}ms 前处理")

    def get_result(self):
        self.has_result = False
        return self.results

    def post_process(self):
        tensor = {
            "cls_8": np.reshape(
                self.npu.output_buffer.get(0, 1 * 1600 * 1), (1, 1600, 1)
            ),
            "cls_16": np.reshape(
                self.npu.output_buffer.get(1, 1 * 400 * 1), (1, 400, 1)
            ),
            "cls_32": np.reshape(
                self.npu.output_buffer.get(2, 1 * 100 * 1), (1, 100, 1)
            ),
            "obj_8": np.reshape(
                self.npu.output_buffer.get(3, 1 * 1600 * 1), (1, 1600, 1)
            ),
            "obj_16": np.reshape(
                self.npu.output_buffer.get(4, 1 * 400 * 1), (1, 400, 1)
            ),
            "obj_32": np.reshape(
                self.npu.output_buffer.get(5, 1 * 100 * 1), (1, 100, 1)
            ),
            "bbox_8": np.reshape(
                self.npu.output_buffer.get(6, 1 * 1600 * 4), (1, 1600, 4)
            ),
            "bbox_16": np.reshape(
                self.npu.output_buffer.get(7, 1 * 400 * 4), (1, 400, 4)
            ),
            "bbox_32": np.reshape(
                self.npu.output_buffer.get(8, 1 * 100 * 4), (1, 100, 4)
            ),
            "kps_8": np.reshape(
                self.npu.output_buffer.get(9, 1 * 1600 * 10), (1, 1600, 10)
            ),
            "kps_16": np.reshape(
                self.npu.output_buffer.get(10, 1 * 400 * 10), (1, 400, 10)
            ),
            "kps_32": np.reshape(
                self.npu.output_buffer.get(11, 1 * 100 * 10), (1, 100, 10)
            ),
        }
        boxes = self.transfer(tensor)
        print(f"transfer boexs={boxes.__len__()}")
        results = self.nms_sorted_bboxes(boxes)

        # 缩放坐标到与原始图像一致
        # original_height, original_width, _ = self.img_raw.shape
        # results = self.scale_coords(original_height, original_width, results)

        return results

    def letterbox_image(self, img, size):
        """Resize image with unchanged aspect ratio using padding"""
        ih, iw, _ = img.shape
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
        new_img = np.full((h, w, 3), 128, dtype=np.uint8)  # 填充颜色为128
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        new_img[dy : dy + nh, dx : dx + nw, :] = img
        return new_img

    def img2byte(self, img):
        img_nchw = np.transpose(img, (2, 0, 1))
        img_nchw_uint8 = img_nchw.astype(np.uint8)
        return bytearray(img_nchw_uint8.tobytes())

    def transfer(self, tensor: np.ndarray, conf_threshold=0.1) -> List["YOLO_RESULT"]:
        """在模型返回的张量数据中寻找可信度达到指定值的检测框"""
        detections = []
        self.npu.save_tensor(".test")
        for scale in [8, 16, 32]:
            cls_key = f"cls_{scale}"
            obj_key = f"obj_{scale}"
            bbox_key = f"bbox_{scale}"

            # 获取当前尺度的类别预测、目标存在性预测和边界框预测
            cls_scores = tensor[cls_key][0, :, 0] 
            obj_scores = tensor[obj_key][0, :, 0]
            bbox_preds = tensor[bbox_key][0, :, :]

            # 筛选出置信度高于阈值的预测
            indices = np.where(obj_scores > conf_threshold)[0]
            print(f"高置信度的有 {len(indices)} 个框")
            for idx in indices:
                obj_score = obj_scores[idx]
                cls_score = cls_scores[idx]
                bbox_pred = bbox_preds[idx, :]

                # 假设人脸类别的索引为1
                if cls_score > conf_threshold:
                    x, y, w, h = bbox_pred
                    re=YOLO_RESULT()
                    re.left_x = int(x - w / 2)
                    re.left_x = int(x + w / 2)
                    re.right_x = int(x + w / 2)
                    re.right_y = int(y + h / 2)
                    re.reliability = obj_score * cls_score
                    # 将检测结果添加到列表中
                    detections.append(re)
            for i in detections:
                print(
                    "{:f} ({:4d},{:4d}) ({:4d},{:4d}) ".format(
                        i[4],
                        int(i[0]),
                        int(i[1]),
                        int(i[2]),
                        int(i[3]),
                    )
                )
            return detections
    def nms_sorted_bboxes(
        self, boxes: List["YOLO_RESULT"], nms_threshold: float = 0.45
    ) -> List["YOLO_RESULT"]:
        if not boxes:
            return []

        # Sort boxes by reliability in descending order
        boxes.sort(key=lambda box: box.reliability, reverse=True)

        picked = []
        areas = [
            (box.right_x - box.left_x) * (box.right_y - box.left_y) for box in boxes
        ]  # Calculate areas of all boxes

        for i in range(len(boxes)):
            if areas[i] == 0:
                continue

            picked.append(boxes[i])

            for j in range(i + 1, len(boxes)):
                if areas[j] == 0:
                    continue

                # Calculate intersection area
                x1 = max(boxes[i].left_x, boxes[j].left_x)
                y1 = max(boxes[i].left_y, boxes[j].left_y)
                x2 = min(boxes[i].right_x, boxes[j].right_x)
                y2 = min(boxes[i].right_y, boxes[j].right_y)

                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                inter_area = w * h

                # Calculate union area
                union_area = areas[i] + areas[j] - inter_area

                # Calculate IoU
                iou = inter_area / union_area

                # Suppress the box if IoU is greater than the threshold
                if iou > nms_threshold:
                    areas[j] = 0

        return picked

    def scale_coords(self, original_height, original_width, boxes):
        """将边界框坐标从image_sizex*image_size转换为原始图像坐标"""
        scale = min(image_size / original_width, image_size / original_height)
        pad_x = (image_size - original_width * scale) / 2
        pad_y = (image_size - original_height * scale) / 2

        for box in boxes:
            box.left_x = int((box.left_x - pad_x) / scale)
            box.left_y = int((box.left_y - pad_y) / scale)
            box.right_x = int((box.right_x - pad_x) / scale)
            box.right_y = int((box.right_y - pad_y) / scale)

        return boxes
