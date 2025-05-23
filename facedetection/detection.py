import os
import numpy as np
import cv2
from typing import List
import awnn_t527

import time
import threading

image_size = 320


__tmp_time = 0


def get_ms_last():
    # 返回自从上次调用本函数后过去了多少ms
    global __tmp_time
    ret = int(time.time() * 1000 - __tmp_time)
    __tmp_time = int(time.time() * 1000)
    return ret


anchors = {
    8: [[10, 13], [16, 30], [33, 23]],  # P3/8
    16: [[30, 61], [62, 45], [59, 119]],  # P4/16
    32: [[116, 90], [156, 198], [373, 326]],  # P5/32
}


class FACE_BOX:
    left_x: int
    left_y: int
    right_x: int
    right_y: int
    class_index: int  # 类别索引
    reliability: float  # 置信度


def desigmoid(x):
    return -np.log(1.0 / x - 1.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class FACEDETECTION:
    has_result = False
    is_running = False
    results: List[FACE_BOX] = []
    npu: awnn_t527.awnn

    def __init__(self, path: str):
        """
        初始化
        @path: 模型路径
        """
        self.npu = awnn_t527.awnn(path)

    def thread_async_detect(self, img, reliability_threshold):
        time_detect_start = int(time.time() * 1000)

        if not self.npu.is_async_running():
            get_ms_last()
            data = self.pre_process(img)
            print(f"\t{get_ms_last()}ms 前处理")
            self.npu.run_async(data)

        while self.npu.is_async_running():
            time.sleep(0.001)
            print(f"{time.time()}", end="\r")
        print(f"                           ", end="\r")
        print(f"\t{get_ms_last()}ms  推理")
        self.npu.save_tensor(".test")
        self.results = self.post_process(reliability_threshold)
        print(f"\t{get_ms_last()}ms  后处理")

        self.has_result = True
        self.is_running = False
        time_detect = int(time.time() * 1000 - time_detect_start)
        fps = int(1000 / time_detect)
        print(f"\t共计 {time_detect}ms  fps:{fps}\n")

    def detect(self, img, reliability_threshold=0.5) -> List["FACE_BOX"]:
        """
        检测图片，阻塞直到检测完成，返回检测结果
        @path: 图片路径
        """
        self.detect_async(img, reliability_threshold)
        while not self.has_result:
            time.sleep(0.001)
        return self.results

    def detect_async(self, img, reliability_threshold=0.5):
        """
        检测图片，立即返回，不阻塞
        @path: 图片路径
        """
        if not self.is_running:
            self.is_running = True

            thread = threading.Thread(
                target=self.thread_async_detect,
                args=(
                    img,
                    reliability_threshold,
                ),
            )
            thread.start()

    def get_result(self):
        self.has_result = False
        return self.results

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

        # 缩放图片
        ih, iw, _ = self.img_raw.shape
        w, h = image_size, image_size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        self.img_resize = np.full((h, w, 3), 128, dtype=np.float32)  # 填充颜色为128
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        self.img_resize[dy : dy + nh, dx : dx + nw, :] = cv2.resize(
            self.img_raw, (nw, nh), interpolation=cv2.INTER_CUBIC
        )
        self.mean_bgr = np.array((123, 117,104 ), dtype=np.float32) 
        # self.mean_bgr = (104, 117, 123)
        self.img_resize -= self.mean_bgr
        self.img_resize = np.ascontiguousarray(self.img_resize)
        # 转换数据格式
        img_nchw = np.transpose(self.img_resize, (2, 0, 1))
        img_nchw_uint8 = img_nchw.astype(np.float32)
        return bytearray(img_nchw_uint8.tobytes())

    def post_process(self, reliability_threshold):

        tensor_position = np.reshape(
            self.npu.output_buffer.get(0, 1 * 4200 * 4), (1, 4200, 4)
        )
        tensor_reliability = np.reshape(
            self.npu.output_buffer.get(1, 1 * 4200 * 2), (1, 4200, 2)
        )
        tensor_keypoint = np.reshape(
            self.npu.output_buffer.get(2, 1 * 4200 * 10), (1, 4200, 10)
        )
        t_real = tensor_reliability[..., 0]
        print(f"reliability_threshold={reliability_threshold}")
        print(f"t_real={t_real[0]}")
        mask = t_real > reliability_threshold
        print(f"{mask.shape}")
        indices = np.argwhere(mask)
        print(f"indices={indices.__len__()}")
        box_x = tensor_position[..., 0][mask]
        box_y = tensor_position[..., 1][mask]
        box_x2 = tensor_position[..., 2][mask]
        box_y2 = tensor_position[..., 3][mask]
        real = tensor_reliability[..., 0][mask]
        boxes = []
        for lx, ly, rx, ry, r in zip(box_x, box_y, box_x2, box_y2, real):
            res = FACE_BOX()
            res.left_x = int(lx)
            res.left_y = int(ly)
            res.right_x = int(rx)
            res.right_y = int(ry)
            res.reliability = r
            boxes.append(res)
        print(f"后处理 boxes:{boxes.__len__()}")
        return []
        return boxes
        # 缩放坐标到与原始图像一致
        original_height, original_width, _ = self.img_raw.shape
        scale = min(image_size / original_width, image_size / original_height)
        pad_x = (image_size - original_width * scale) / 2
        pad_y = (image_size - original_height * scale) / 2

        for box in boxes:
            box.left_x = int((box.left_x - pad_x) / scale)
            box.left_y = int((box.left_y - pad_y) / scale)
            box.right_x = int((box.right_x - pad_x) / scale)
            box.right_y = int((box.right_y - pad_y) / scale)

        return boxes

    def tensor_parse(
        self, tensor: np.ndarray, stride: int, reliability_threshold
    ) -> List["FACE_BOX"]:
        """在模型返回的张量数据中寻找可信度达到指定值的检测框"""
        ret = []
        tensor = np.reshape(
            tensor, (3, int(image_size / stride), int(image_size / stride), 85)
        )

        de_releability_threshold = desigmoid(reliability_threshold)

        sigmoid_output = tensor[..., 4]
        mask = sigmoid_output > de_releability_threshold
        indices = np.argwhere(mask)
        if len(indices) == 0:
            return ret
        sigmoid_output_masked = sigmoid_output[mask]
        classes = tensor[..., 5:][mask]
        box_x = (sigmoid(tensor[..., 0][mask]) * 2 - 0.5 + indices[:, 2]) * stride
        box_y = (sigmoid(tensor[..., 1][mask]) * 2 - 0.5 + indices[:, 1]) * stride
        box_w = (sigmoid(tensor[..., 2][mask]) * 2) ** 2
        box_h = (sigmoid(tensor[..., 3][mask]) * 2) ** 2

        anchor_indices = indices[:, 0]
        anchors_selected = np.array([anchors[stride][i] for i in anchor_indices])
        box_w *= anchors_selected[:, 0]
        box_h *= anchors_selected[:, 1]

        x_min = box_x - box_w / 2
        y_min = box_y - box_h / 2
        x_max = box_x + box_w / 2
        y_max = box_y + box_h / 2

        class_indices = np.argmax(classes, axis=1)
        max_classes = np.take_along_axis(
            classes, class_indices[:, np.newaxis], axis=1
        ).flatten()
        reliability = sigmoid(max_classes) * sigmoid(sigmoid_output_masked)

        # 过滤掉置信度低于阈值的边界框
        valid_indices = reliability > 0.5
        x_min, y_min, x_max, y_max, class_indices, reliability = (
            x_min[valid_indices],
            y_min[valid_indices],
            x_max[valid_indices],
            y_max[valid_indices],
            class_indices[valid_indices],
            reliability[valid_indices],
        )

        # 裁剪坐标以确保它们在图像的有效范围内
        image_width = image_size
        image_height = image_size

        x_min = np.clip(x_min, 0, image_width).astype(int)
        y_min = np.clip(y_min, 0, image_height).astype(int)
        x_max = np.clip(x_max, 0, image_width).astype(int)
        y_max = np.clip(y_max, 0, image_height).astype(int)

        for lx, ly, rx, ry, ci, rel in zip(
            x_min, y_min, x_max, y_max, class_indices, reliability
        ):
            re = FACE_BOX()
            re.left_x = lx
            re.left_y = ly
            re.right_x = rx
            re.right_y = ry
            re.reliability = rel
            re.class_index = ci
            ret.append(re)

        return ret

    def nms_sorted_bboxes(
        self, boxes: List["FACE_BOX"], nms_threshold: float = 0.45
    ) -> List["FACE_BOX"]:
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
