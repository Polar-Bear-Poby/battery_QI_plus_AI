from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr


# added by loveiori: this class makes PascalVoc dataset simpler
class SimpleSegmentation(Dataset):
    NUM_CLASSES = None

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('simple'),
                 split='train',
                 ):
        """
        :param base_dir: path to base directory
        :param split: train/val/test
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'test')
        self._cat_dir = os.path.join(self._base_dir, 'test')
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

            for ii, line in enumerate(frame_lines):
                _image = os.path.join(self._image_dir, splt, line)
                if (os.path.isdir(_image)):
                    continue
                assert os.path.isfile(_image)
                self.im_ids.append(line.rsplit('.', 1)[0])
                self.images.append(_image)

            for ii, line in enumerate(mask_lines):
                _cat = os.path.join(self._cat_dir, splt, line)
                if (os.path.isdir(_cat)):
                    continue
                assert os.path.isfile(_cat)
                im_ids2.append(line.rsplit('.', 1)[0])
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))
        assert (bool(set(self.im_ids).intersection(im_ids2)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target, _imagename, _imagesize = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target, 'imagename': _imagename, 'imagesize': _imagesize}

        for split in self.split:
            if split == "test":  # 테스트 모드일 때만 실행
                if self.args.crop == "slide":  # crop slide 모드일 때만 실행
                    return self.transform_none(sample)

        # 테스트 모드가 아닐 경우는 실행되지 않음

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target, self.images[index], _img.size

    def transform_none(self, sample):
        """ 테스트 모드에서 crop slide일 때만 적용되는 변환 """
        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'SimpleSegmentation(split=' + str(self.split) + ')'

# =============== [아래 코드는 실행되지 않음] ===============
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
