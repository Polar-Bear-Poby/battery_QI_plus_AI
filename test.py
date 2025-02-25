import argparse
import os
import time, calendar
import numpy as np 
import tqdm
import torch
import cv2
import json
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from skimage.transform import resize
from dataloaders.datasets import simple
from modeling.deeplab import *
from dataloaders.utils import get_pascal_labels
from utils.metrics import Evaluator
from logger import LoggerHelper
import itertools

# 전역 변수 추가 (테스트용 확인 여부)
checked_pred = False
class Tester(object):
    def __init__(self, args, base_dir):
        if not os.path.isfile(args.model):
            raise RuntimeError("No checkpoint found at '{}'".format(args.model))
        
        self.args = args
        self.color_map = get_pascal_labels()

        if base_dir and len(base_dir):
            self.test_set = simple.SimpleSegmentation(args, base_dir=base_dir, split='test')
        else:
            self.test_set = simple.SimpleSegmentation(args, split='test')
        simple.SimpleSegmentation.NUM_CLASSES = args.num_classes
        self.num_classes = simple.SimpleSegmentation.NUM_CLASSES

        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.test_loader = DataLoader(self.test_set, shuffle=True, **kwargs)

        # Define model
        self.model = DeepLab(num_classes=self.num_classes,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=False,
                        freeze_bn=False)
        
        # Using cuda
        if args.gpu_ids and torch.cuda.is_available():
            if len(args.gpu_ids) > 1:
                self.device = torch.device("cuda")
                self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            else: 
                self.device = torch.device("cuda:{}".format(args.gpu_ids[0]))
        else:
            self.device = torch.device('cpu')

        checkpoint = torch.load(args.model, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.evaluator = Evaluator(self.num_classes)
        self.fwIoUs = []
        self.mIoUs = []
        self.elapsed_list = []  

    def save_pr_json(self, output_json_path, gt_json_path, pollution_contours, damaged_contours, battery_outline_contours):
        """
        모델 예측을 반영한 JSON 저장 (올바른 JSON 형식 유지, 픽셀 좌표 리스트 평탄화 적용)
        """
        # GT JSON 파일 로드
        with open(gt_json_path, "r", encoding="utf-8") as f:
            gt_json_data = json.load(f)

        # 기존 GT 데이터 복사
        output_json_data = gt_json_data.copy()

        # `defects` 리스트가 없으면 빈 리스트로 초기화
        output_json_data.setdefault("defects", [])

        # `defects`에 pollution 및 damaged 추가 (2D 리스트 → 1D 리스트 변환 적용)
        output_json_data["defects"].extend([
            {
                "id": 2,
                "name": "Pollution",
                "points": list(itertools.chain.from_iterable(instance["contour"]))
            }
            for instance in pollution_contours
        ] + [
            {
                "id": 1,
                "name": "Damaged",
                "points": list(itertools.chain.from_iterable(instance["contour"]))
            }
            for instance in damaged_contours
        ])

        # ✅ `swelling` 안에 `battery_outline` 저장 (defects에 포함하지 않음)
        output_json_data.setdefault("swelling", {})

        output_json_data["swelling"]["battery_outline"] = list(
            itertools.chain.from_iterable(itertools.chain.from_iterable(instance["contour"] for instance in battery_outline_contours))
        )  # ✅ 2D → 1D 변환 적용

        # `is_normal` 업데이트 (defects가 하나라도 존재하면 False, 없으면 True)
        output_json_data["image_info"]["is_normal"] = len(output_json_data["defects"]) == 0

        # 수정된 JSON 저장
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_json_data, f, indent=4, ensure_ascii=False)



    def add_pr_outline(self, image, pollution_contours, damaged_contours, battery_outline_contours):
        """모델 예측된 외곽선을 원본 이미지에 겹치는 함수"""
        # ⚠️ PIL.Image → NumPy 배열로 변환 (OpenCV에서 사용 가능하도록)
        image = np.array(image)

        # 색상 정의
        colors = {
            "battery_outline": (255, 255, 0),  # 노란색
            "damaged": (255, 165, 0),         # 주황색
            "pollution": (0, 0, 255)          # 파란색
        }

        # 외곽선 그리기 (최적화: zip 활용)
        for contours, color in zip(
            [battery_outline_contours, damaged_contours, pollution_contours],
            [colors["battery_outline"], colors["damaged"], colors["pollution"]]
        ):
            for contour in contours:
                cv2.drawContours(image, [np.array(contour["contour"]).reshape(-1, 1, 2)], -1, color, 5)

        return Image.fromarray(image)


    def save(self, array, id, type, resize=None):
        r = array.copy()
        g = array.copy()
        b = array.copy()

        for i in range(self.num_classes):
            r[array == i] = self.color_map[i][0]
            g[array == i] = self.color_map[i][1]
            b[array == i] = self.color_map[i][2]

        rgb = np.dstack((r, g, b))

        img = Image.fromarray(rgb.astype('uint8'))
        if resize and (img.size[0] != resize[0] or img.size[1] != resize[1]):
            img = img.resize(resize, Image.NEAREST)

        filename = str(id) + '_' + type + '.png'
        img.save(os.path.join(self.args.save_dir, filename))

        return filename

    def extract_contour_coordinates(self, mask, class_name):
        """특정 클래스의 마스크에서 개별 객체의 외곽선 좌표를 추출 (최적화 버전)"""
        binary_mask = (mask > 0).astype(np.uint8)  # 변환 연산 최소화
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return [{
            "class": class_name,
            "contour": contour.reshape(-1, 2).tolist()
        } for contour in contours]

    def add_mask_outline(self, image, mask, color, thickness=5):
        """원본 이미지에 외곽선만 겹쳐서 표시"""
        image = image.copy()
        draw = ImageDraw.Draw(image)

        h, w = mask.shape
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                # 현재 픽셀이 배경이 아니고, 주변에 다른 클래스(즉, 경계선)가 있으면 선을 그림
                if mask[y, x] > 0:
                    if (mask[y, x] != mask[y-1, x] or mask[y, x] != mask[y+1, x] or
                        mask[y, x] != mask[y, x-1] or mask[y, x] != mask[y, x+1]):
                        draw.line([(x-thickness//2, y-thickness//2), (x+thickness//2, y+thickness//2)], fill=color, width=thickness)

        return image

    def test(self):
        self.model.eval()
        self.model.to(device=self.device)

        LoggerHelper.getLogger().info("********************************************************************************")
        LoggerHelper.getLogger().info(f"Evaluation Started. (timestamp: {calendar.timegm(time.gmtime())})")
        LoggerHelper.getLogger().info(str(self.args).replace("Namespace(", "Argument("))
        LoggerHelper.getLogger().info("********************************************************************************")

        total = len(self.test_loader)
        for i, sample in tqdm.tqdm(enumerate(self.test_loader), total=total):
            image, target, imagename, imagesize = sample['image'], sample['label'], sample['imagename'], sample['imagesize']
            target = target.numpy()
            if torch.cuda.is_available():
                image = image.cuda()

            started = time.time()
            pred = None
            with torch.no_grad():
                if self.args.crop == "slide":
                    crop_size = self.args.crop_size
                    _, height, width = target.shape
                    pred = np.zeros((height, width), "uint8")
                    for x in range(0 + self.args.offset_width, width - self.args.offset_width, crop_size):
                        for y in range(0 + self.args.offset_height, height - self.args.offset_height, crop_size):
                            box = (x, y,
                                    x + crop_size if x + crop_size < width else width,
                                    y + crop_size if y + crop_size < height else height)
                            output = self.model(image[:, :, y:box[3], x:box[2]])
                            pred[y:box[3], x:box[2]] = np.argmax(output[0].detach().cpu().clone().numpy(), axis=0)
                else:
                    output = self.model(image)
                    pred = np.argmax(output[0].detach().cpu().clone().numpy(), axis=0)

            elapsed = time.time() - started
            self.evaluator.add_batch(target[0], pred)
            fwIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            mIoU = self.evaluator.Mean_Intersection_over_Union()

            LoggerHelper.getLogger().info(f"[{i+1}/{total}] \"{os.path.basename(imagename[0])}\" predicted. mIoU: {mIoU:.4f}, fwIoU: {fwIoU:.4f}, Elapsed: {elapsed:.4f}s")

            if self.args.save_dir and len(self.args.save_dir):
                if not os.path.exists(self.args.save_dir):
                    os.makedirs(self.args.save_dir)
                savename = os.path.splitext(os.path.basename(imagename[0]))[0]
                pr_filename = self.save(pred, savename, "pr", (int(imagesize[0]), int(imagesize[1])) if self.args.crop == "none" else None)
                gt_filename = self.save(target[0], savename, "gt", (int(imagesize[0]), int(imagesize[1])) if self.args.crop == "none" else None)
                np.savetxt(os.path.join(self.args.save_dir, f"{savename}_cm.txt"), self.evaluator.confusion_matrix)

            self.evaluator.reset()
            self.fwIoUs.append(fwIoU)
            self.mIoUs.append(mIoU)
            self.elapsed_list.append(elapsed)

            # 🔹 **GT JSON 파일 경로 설정**
            gt_json_path = os.path.join(self.args.base_dir, f"masks/test/{savename}.json")
            pr_json_path = os.path.join(self.args.save_dir, f"{savename}_pr.json")

            # 🔹 **세그멘테이션에서 개별 객체 외곽선 추출**
            pollution_contours = self.extract_contour_coordinates((pred == 1).astype(np.uint8), "Pollution")
            damaged_contours = self.extract_contour_coordinates((pred == 2).astype(np.uint8), "Damaged")
            battery_outline_contours = self.extract_contour_coordinates((pred == 3).astype(np.uint8), "Battery_Outline")  # 추가

            # 🔹 **JSON 저장 (배터리 외곽선도 포함)**
            self.save_pr_json(pr_json_path, gt_json_path, pollution_contours, damaged_contours, battery_outline_contours)

            # 🔹 **PR 외곽선을 원본 이미지에 추가하여 저장**
            pr_outline_image = self.add_pr_outline(
                Image.open(imagename[0]).convert("RGB"),
                pollution_contours,
                damaged_contours,
                battery_outline_contours
            )
            pr_outline_image.save(os.path.join(self.args.save_dir, f"{savename}_PR_외곽선_원본이미지.png"))

        LoggerHelper.getLogger().info("********************************************************************************")
        LoggerHelper.getLogger().info(f"Evaluation Summary (timestamp: {calendar.timegm(time.gmtime())})")
        LoggerHelper.getLogger().info(f"Number of Datasets: {total}")
        LoggerHelper.getLogger().info(f"Mean IoU: {np.mean(self.mIoUs):.4f}")
        LoggerHelper.getLogger().info(f"Frequency Weighted IoU: {np.mean(self.fwIoUs):.4f}")
        LoggerHelper.getLogger().info(f"Prediction Elapsed: {np.sum(self.elapsed_list):.4f}s")
        LoggerHelper.getLogger().info(f"Average FPS: {total/np.sum(self.elapsed_list):.4f}")
        LoggerHelper.getLogger().info("********************************************************************************")


def main():
    parser = argparse.ArgumentParser(description='DeepLab v3+ Evaluation')
    
    # Essential params
    parser.add_argument('--num-classes', type=int, 
                        default=4,
                        # default=3,
                        help='number of classes (default: 4)')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--backbone', default='drn', 
                        help='backbone name (default: drn)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='output stride')
    parser.add_argument('--model', type=str, 
                        default='weights/aihub2023_battery_rgb_c4_b640.pt',
                        # default='weights/easternsky_c3_b650.pt',
                        help='model path to be loaded')
    parser.add_argument('--save_dir', type=str, default='logs/result',
                        help='directory to save prediction')
    parser.add_argument('--base_dir', type=str, 
                        default='testset/RGB/20231023',
                        # default='testset/easternsky/20231013',
                        help='directory to evaluate dataset')

    # Params to use DataLoader
    parser.add_argument('--workers', type=int, default=1,
                        metavar='N', help='dataloader threads')

    parser.add_argument('--crop-size', type=int, 
                        default=640,
                        # default=650,
                        help='crop image size')
    parser.add_argument('--crop', type=str, default='slide',
                        help='crop type: slide, random, none')
    parser.add_argument('--offset_width', type=int, default=0,
                        help='offset width for inference (if crop: slide)')
    parser.add_argument('--offset_height', type=int, default=0,
                        help='offset height for inference (if crop: slide)')

    args = parser.parse_args()
    args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]

    tester = Tester(args, base_dir=args.base_dir)
    tester.test()


if __name__ == "__main__":
    main()