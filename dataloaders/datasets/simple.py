from __future__ import print_function, division
import os
from PIL import Image, ImageDraw
import numpy as np
import json
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr


# added by loveiori: this class makes PascalVoc dataset simpler
class SimpleSegmentation(Dataset):
    NUM_CLASSES = None

    def __init__(self, args, base_dir=Path.db_root_dir('simple'), split='train'):
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'frames')
        self._cat_dir = os.path.join(self._base_dir, 'masks')
        SimpleSegmentation.NUM_CLASSES = args.num_classes

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args
        self.im_ids = []
        im_ids2 = []
        self.images = []
        self.categories = []

        for splt in self.split:
            frame_lines = os.listdir(os.path.join(self._image_dir, splt))
            frame_lines.sort()
            mask_lines = os.listdir(os.path.join(self._cat_dir, splt))
            mask_lines.sort()

            for line in frame_lines:
                _image = os.path.join(self._image_dir, splt, line)
                if os.path.isdir(_image):
                    continue
                assert os.path.isfile(_image)
                self.im_ids.append(line.rsplit('.', 1)[0])
                self.images.append(_image)

            for line in mask_lines:
                _cat = os.path.join(self._cat_dir, splt, line)
                if os.path.isdir(_cat):
                    continue
                assert os.path.isfile(_cat)
                im_ids2.append(line.rsplit('.', 1)[0])
                self.categories.append(_cat)

        assert len(self.images) == len(self.categories)
        assert bool(set(self.im_ids).intersection(im_ids2))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target, _imagename, _imagesize = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target, 'imagename': _imagename, 'imagesize': _imagesize}
        return self.transform_none(sample)

    def _make_img_gt_point_pair(self, index):
        """이미지와 Ground Truth 세그멘테이션 마스크를 로드"""
        _img = Image.open(self.images[index]).convert('RGB')
        mask_path = self.categories[index]

        if mask_path.endswith('.json'):
            _target = self.load_json_mask(mask_path, _img.size)  # JSON을 마스크로 변환
        else:
            _target = Image.open(mask_path)

        return _img, _target, self.images[index], _img.size

    def load_json_mask(self, json_path, image_size):
        """JSON 파일을 읽어 Ground Truth 마스크로 변환"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mask = Image.new("L", image_size, 0)  # 0: 배경 (Grayscale)
        draw = ImageDraw.Draw(mask)

        class_map = {
            "battery_outline": 3,  # 배터리 외곽선 (초록)
            "Damaged": 2,         # 손상 (빨강)
            "Pollution": 1        # 오염 (파랑)
        }

        # 배터리 외곽선
        if "battery_outline" in data.get("swelling", {}):
            points = data["swelling"]["battery_outline"]
            if len(points) % 2 == 0:
                polygon = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
                draw.polygon(polygon, fill=class_map["battery_outline"])

        # 손상 & 오염
        if "defects" in data:
            for defect in data["defects"]:
                defect_type = defect["name"]
                if defect_type in class_map:
                    points = defect["points"]
                    if len(points) % 2 == 0:
                        polygon = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
                        draw.polygon(polygon, fill=class_map[defect_type])

        return mask


    def transform_none(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def __str__(self):
        return 'SimpleSegmentation(split=' + str(self.split) + ')'


# 💡 데이터셋 로드 테스트 코드 (주석 처리)
# if __name__ == '__main__':
#     from dataloaders.utils import decode_segmap
#     from torch.utils.data import DataLoader
#     import matplotlib.pyplot as plt
#     import argparse

#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     args.base_size = 513
#     args.crop_size = 513

#     args.num_classes = 7

#     voc_train = SimpleSegmentation(args, split='train')

#     dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

#     try:
#         for ii, sample in enumerate(dataloader):
#             for jj in range(sample["image"].size()[0]):
#                 img = sample['image'].numpy()
#                 gt = sample['label'].numpy()
#                 tmp = np.array(gt[jj]).astype(np.uint8)
#                 segmap = decode_segmap(tmp, dataset='simple', n_classes=9)
#                 img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
#                 img_tmp *= (0.229, 0.224, 0.225)
#                 img_tmp += (0.485, 0.456, 0.406)
#                 img_tmp *= 255.0
#                 img_tmp = img_tmp.astype(np.uint8)
#                 plt.figure()
#                 plt.title('display')
#                 plt.subplot(211)
#                 plt.imshow(img_tmp)
#                 plt.subplot(212)
#                 plt.imshow(segmap)

#                 # Added to figure out
#                 plt.waitforbuttonpress(.5)

#             if ii == 1:
#                 break
#     except Exception as e:
#         print(e)
#         pass

#     plt.show(block=True)
