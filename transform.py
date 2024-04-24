# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import random

import SimpleITK as sitk
import numpy as np
from monai import transforms
from torch.utils.data import Dataset
from tqdm import tqdm


def cross_valid(data_dir, k=0):
    all_dirs = os.listdir(data_dir)
    all_paths = [os.path.join(data_dir, d) for d in all_dirs]
    random.seed(39)
    random.shuffle(all_paths)

    train_files = []
    val_files = []
    test_files = []
    assert 0 <= k <= 5
    k_s = []
    for i in range(7):
        k_s.append(all_paths[i * 208:min((i + 1) * 208, 1251)])
    for i in range(6):
        if i == k:
            test_files = k_s[i]
        elif i == (k + 5) % 6:
            val_files = k_s[i]
        else:
            train_files.extend(k_s[i])
    train_files.extend(k_s[6][:2])
    test_files.append(k_s[6][2])
    train_val_files = train_files[:208]

    print(f"train is {len(train_files)}, val is {len(val_files)}")

    return train_files, val_files, train_val_files, test_files


class PretrainDataset1st(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)

    def read_data(self, data_path):

        file_identifizer = data_path.split("/")[-1].split("-")[-2]
        file_idx = data_path.split("/")[-1].split("-")[-1]
        image_paths = [
            os.path.join(data_path, f"BraTS-GLI-{file_identifizer}-{file_idx}-t1n.nii.gz"),
            os.path.join(data_path, f"BraTS-GLI-{file_identifizer}-{file_idx}-t1c.nii.gz"),
            os.path.join(data_path, f"BraTS-GLI-{file_identifizer}-{file_idx}-t2w.nii.gz"),
            os.path.join(data_path, f"BraTS-GLI-{file_identifizer}-{file_idx}-t2f.nii.gz")
        ]

        seg_path = os.path.join(data_path, f"BraTS-GLI-{file_identifizer}-{file_idx}-seg.nii.gz")

        image_data = [sitk.GetArrayFromImage(sitk.ReadImage(p)) for p in image_paths]
        seg_data = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))

        image_data = np.array(image_data).astype(np.float32)
        seg_data = np.expand_dims(np.array(seg_data).astype(np.int32), axis=0)
        # seg NO.3->4
        np.place(seg_data, seg_data == 3, 4)
        # image_data = np.pad(image_data, [(0,0), (0,1), (0,0), (0,0)], "constant")
        # seg_data = np.pad(seg_data, [(0,0), (0,1), (0,0), (0,0)], "constant")
        return {
            "image": image_data,
            "label": seg_data
        }

    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else:
            try:
                image = self.read_data(self.datalist[i])
            except:
                print('getitem error')
                if i != len(self.datalist) - 1:
                    return self.__getitem__(i + 1)
                else:
                    return self.__getitem__(i - 1)
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.datalist)


class PretrainDataset2nd(Dataset):
    def __init__(self, datalist, coarse_dir, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.coarse_dir = coarse_dir
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)

    def read_data(self, data_path):

        file_identifizer = data_path.split("/")[-1].split("-")[-2]
        file_idx = data_path.split("/")[-1].split("-")[-1]
        image_paths = [
            os.path.join(data_path, f"BraTS-GLI-{file_identifizer}-{file_idx}-t1n.nii.gz"),
            os.path.join(data_path, f"BraTS-GLI-{file_identifizer}-{file_idx}-t1c.nii.gz"),
            os.path.join(data_path, f"BraTS-GLI-{file_identifizer}-{file_idx}-t2w.nii.gz"),
            os.path.join(data_path, f"BraTS-GLI-{file_identifizer}-{file_idx}-t2f.nii.gz")
        ]
        coarse_path = os.path.join(self.coarse_dir, f"BraTS-GLI-seg-{file_identifizer}-{file_idx}.nii.gz")

        seg_path = os.path.join(data_path, f"BraTS-GLI-{file_identifizer}-{file_idx}-seg.nii.gz")

        image_data = [sitk.GetArrayFromImage(sitk.ReadImage(p)) for p in image_paths]
        coarse_data = sitk.GetArrayFromImage(sitk.ReadImage(coarse_path))
        seg_data = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))

        image_data = np.array(image_data).astype(np.float32)
        coarse_data = np.array(coarse_data).astype(np.float32)
        if coarse_data.ndim == 3:
            coarse_data = np.expand_dims(coarse_data, axis=0)
            # seg NO.3->4
            np.place(coarse_data, coarse_data == 3, 4)
        seg_data = np.expand_dims(np.array(seg_data).astype(np.int32), axis=0)
        # seg NO.3->4
        np.place(seg_data, seg_data == 3, 4)
        # image_data = np.pad(image_data, [(0,0), (0,1), (0,0), (0,0)], "constant")
        # seg_data = np.pad(seg_data, [(0,0), (0,1), (0,0), (0,0)], "constant")
        return {
            "image": image_data,
            "label": seg_data,
            "coarse": coarse_data,
            "PATH": f"BraTS-GLI-{file_identifizer}-{file_idx}",
        }

    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else:
            try:
                image = self.read_data(self.datalist[i])
            except AssertionError as e:
                # print(f"No ET error id={path}")
                if i != len(self.datalist) - 1:
                    return self.__getitem__(i + 1)
                else:
                    print("BACK GET ITEM")
                    return self.__getitem__(i - 1)
            except:
                print('getitem error')
                if i != len(self.datalist) - 1:
                    return self.__getitem__(i + 1)
                else:
                    print("BACK GET ITEM")
                    return self.__getitem__(i - 1)
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.datalist)


# class PretrainDataset3rd(Dataset):
#     def __init__(self, datalist, coarse_dir, transform=None, cache=False, et_exclude=False) -> None:
#         super().__init__()
#         self.transform = transform
#         self.datalist = datalist
#         self.coarse_dir = coarse_dir
#         self.cache = cache
#         self.et_exclude = et_exclude
#         if cache:
#             self.cache_data = []
#             for i in tqdm(range(len(datalist)), total=len(datalist)):
#                 d = self.read_data(datalist[i])
#                 self.cache_data.append(d)
#
#     def read_data(self, data_path):
#
#         file_identifizer = data_path.split("/")[-1].split("-")[-2]
#         file_idx = data_path.split("/")[-1].split("-")[-1]
#         image_paths = [
#             os.path.join(data_path, f"BraTS-GLI-{file_identifizer}-{file_idx}-t1n.nii.gz"),
#             os.path.join(data_path, f"BraTS-GLI-{file_identifizer}-{file_idx}-t1c.nii.gz"),
#             os.path.join(data_path, f"BraTS-GLI-{file_identifizer}-{file_idx}-t2w.nii.gz"),
#             os.path.join(data_path, f"BraTS-GLI-{file_identifizer}-{file_idx}-t2f.nii.gz")
#         ]
#         coarse_path = os.path.join(self.coarse_dir, f"BraTS-GLI-seg-{file_identifizer}-{file_idx}.nii.gz")
#
#         seg_path = os.path.join(data_path, f"BraTS-GLI-{file_identifizer}-{file_idx}-seg.nii.gz")
#
#         image_data = [sitk.GetArrayFromImage(sitk.ReadImage(p)) for p in image_paths]
#         coarse_data = sitk.GetArrayFromImage(sitk.ReadImage(coarse_path))
#         seg_data = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
#
#         image_data = np.array(image_data).astype(np.float32)
#         coarse_data = np.array(coarse_data).astype(np.float32)
#         if coarse_data.ndim == 3:
#             coarse_data = np.expand_dims(coarse_data, axis=0)
#             # seg NO.3->4
#             np.place(coarse_data, coarse_data == 3, 4)
#         seg_data = np.expand_dims(np.array(seg_data).astype(np.int32), axis=0)
#         # seg NO.3->4
#         np.place(seg_data, seg_data == 3, 4)
#
#         cropping_data = (seg_data == 4) | (coarse_data == 4)
#         # image_data = np.pad(image_data, [(0,0), (0,1), (0,0), (0,0)], "constant")
#         # seg_data = np.pad(seg_data, [(0,0), (0,1), (0,0), (0,0)], "constant")
#         return {
#             "image": image_data,
#             "label": seg_data,
#             "coarse": coarse_data,
#             "cropping": cropping_data.astype(int),
#             "PATH": f"BraTS-GLI-{file_identifizer}-{file_idx}",
#         }
#
#     def __getitem__(self, i):
#         # path = ""
#         if self.cache:
#             image = self.cache_data[i]
#         else:
#             try:
#                 image = self.read_data(self.datalist[i])
#                 assert not (self.et_exclude and image["cropping"].sum() == 0)
#             except AssertionError as e:
#                 # print(f"No ET error id={path}")
#                 if i != len(self.datalist) - 1:
#                     return self.__getitem__(i + 1)
#                 else:
#                     print("BACK GET ITEM")
#                     return self.__getitem__(i - 1)
#             except:
#                 print('getitem error')
#                 if i != len(self.datalist) - 1:
#                     return self.__getitem__(i + 1)
#                 else:
#                     print("BACK GET ITEM")
#                     return self.__getitem__(i - 1)
#         if self.transform is not None:
#             image = self.transform(image)
#
#         return image
#
#     def __len__(self):
#         return len(self.datalist)


def get_loader_brats1st(data_dir, fold=0):
    train_files, val_files, train_val_files, test_files = cross_valid(data_dir, k=fold)

    train_transform = transforms.Compose(
        [
            transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),

            transforms.RandScaleIntensityd(keys="image", factors=0.2, prob=1.0),
            # transforms.SavitzkyGolaySmoothd(keys="image", window_length=3, order=1),

            transforms.Padd(keys=["image", "label"], padder=transforms.Pad([(0, 0), (0, 1), (0, 0), (0, 0)])),
            transforms.Resized(keys=["image", "label"], spatial_size=(78, 120, 120), mode="trilinear"),

            transforms.RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.8, max_zoom=1.0, mode="trilinear"),
            transforms.RandRotated(keys=["image", "label"], prob=0.5, range_x=0.1,
                                   mode="bilinear", padding_mode="zeros"),

            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),

            transforms.ToTensord(keys=["image", "label"], ),
        ]
    )
    val_loss_transform = transforms.Compose(
        [
            transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
            transforms.Padd(keys=["image", "label"], padder=transforms.Pad([(0, 0), (0, 1), (0, 0), (0, 0)])),
            # transforms.SavitzkyGolaySmoothd(keys="image", window_length=3, order=1),
            transforms.Resized(keys=["image", "label"], spatial_size=(78, 120, 120), mode="trilinear"),

            # transforms.ThresholdIntensityd(keys="image", threshold=0.01, above=True, cval=-3000),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # transforms.ThresholdIntensityd(keys="image", threshold=0.0, above=True, cval=0),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
            transforms.Padd(keys=["image", "label"], padder=transforms.Pad([(0, 0), (0, 1), (0, 0), (0, 0)])),
            # transforms.SavitzkyGolaySmoothd(keys="image", window_length=3, order=1),
            transforms.Resized(keys=["image", "label"], spatial_size=(78, 120, 120), mode="trilinear"),

            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    train_ds = PretrainDataset1st(train_files, transform=train_transform)
    val_loss_ds = PretrainDataset1st(val_files, transform=val_loss_transform)
    train_val_ds = PretrainDataset1st(train_val_files, transform=val_transform)
    val_ds = PretrainDataset1st(val_files, transform=val_transform)
    # test_ds = PretrainDataset(test_files, transform=val_transform)

    loader = [train_ds, val_loss_ds, train_val_ds, val_ds]

    return loader


def get_loader_brats2nd(data_dir, coarse_dir, fold=0):
    train_files, val_files, train_val_files, test_files = cross_valid(data_dir, k=fold)

    train_transform = transforms.Compose(
        [
            transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
            transforms.Resized(keys=["coarse"], spatial_size=(156, 240, 240), mode="trilinear"),
            transforms.CropForegroundd(keys=["image", "label", "coarse"], source_key="image"),
            transforms.RandSpatialCropd(keys=["image", "label", "coarse"], roi_size=[96, 96, 96],
                                        random_size=False),
            transforms.RandFlipd(keys=["image", "label", "coarse"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label", "coarse"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label", "coarse"], prob=0.5, spatial_axis=2),

            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label", "coarse", ]),
        ]
    )
    val_loss_transform = transforms.Compose(
        [
            transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
            transforms.Resized(keys=["coarse"], spatial_size=(156, 240, 240), mode="trilinear"),
            transforms.CropForegroundd(keys=["image", "label", "coarse"], source_key="image"),

            transforms.RandSpatialCropd(keys=["image", "label", "coarse"], roi_size=[96, 96, 96],
                                        random_size=False),

            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

            transforms.ToTensord(keys=["image", "label", "coarse"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
            transforms.Resized(keys=["coarse"], spatial_size=(156, 240, 240), mode="trilinear"),
            transforms.SpatialCropd(keys=["coarse"], roi_start=(0, 0, 0), roi_end=(155, 240, 240)),
            transforms.CropForegroundd(keys=["image", "label", "coarse"], source_key="image"),

            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label", "coarse"]),
        ]
    )

    train_ds = PretrainDataset2nd(train_files, coarse_dir=coarse_dir, transform=train_transform)
    val_loss_ds = PretrainDataset2nd(val_files, coarse_dir=coarse_dir, transform=val_loss_transform)
    train_val_ds = PretrainDataset2nd(train_val_files, coarse_dir=coarse_dir, transform=val_transform)
    val_ds = PretrainDataset2nd(val_files, coarse_dir=coarse_dir, transform=val_transform)
    # test_ds = PretrainDataset(test_files, transform=val_transform)

    loader = [train_ds, val_loss_ds, train_val_ds, val_ds]

    return loader

# def get_loader_brats3rd(data_dir, coarse_dir, batch_size=1, fold=0, num_workers=8, input_size=48):
#
#     all_dirs = os.listdir(data_dir)
#     all_paths = [os.path.join(data_dir, d) for d in all_dirs]
#     random.seed(39)
#     random.shuffle(all_paths)
#     size = len(all_paths)
#     print(size)
#     # train_size = int(0.8 * size)
#     # val_size = size - train_size
#
#     train_files, val_files, test_files = cross_valid(all_paths, k=4)
#
#     # train_files = all_paths[:train_size]
#     # train_val_files = train_files[:val_size]
#     # val_files = all_paths[train_size:train_size+val_size]
#     # test_files = all_paths[train_size+val_size:]
#     print(f"train is {len(train_files)}, val is {len(val_files)}")
#
#     train_transform = transforms.Compose(
#         [
#             transforms.CropForegroundd(keys=["image", "label", "coarse", "cropping"], source_key="image"),
#             transforms.RandCropByLabelClassesd(keys=["image", "label", "coarse", "cropping"], label_key="cropping",
#                                                spatial_size=[input_size, input_size, input_size],
#                                                ratios=[0, 1], num_classes=2),
#             # transforms.RandSpatialCropd(keys=["image", "label", "coarse"], roi_size=[input_size, input_size, input_size],
#             #                             random_size=False),
#             transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label", "coarse"]),
#
#             transforms.RandFlipd(keys=["image", "label", "coarse"], prob=0.5, spatial_axis=0),
#             transforms.RandFlipd(keys=["image", "label", "coarse"], prob=0.5, spatial_axis=1),
#             transforms.RandFlipd(keys=["image", "label", "coarse"], prob=0.5, spatial_axis=2),
#
#             transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#
#             transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
#             transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
#             transforms.ToTensord(keys=["image", "label", "coarse"]),
#         ]
#     )
#     val_loss_transform = transforms.Compose(
#         [
#             transforms.CropForegroundd(keys=["image", "label", "coarse", "cropping"], source_key="image"),
#             transforms.RandCropByLabelClassesd(keys=["image", "label", "coarse", "cropping"], label_key="cropping",
#                                                spatial_size=[input_size, input_size, input_size],
#                                                ratios=[0, 1], num_classes=2),
#             # transforms.RandSpatialCropd(keys=["image", "label", "coarse"], roi_size=[input_size, input_size, input_size],
#             #                             random_size=False),
#             transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label", "coarse"]),
#
#             transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#
#             transforms.ToTensord(keys=["image", "label", "coarse"]),
#         ]
#     )
#     val_transform = transforms.Compose(
#         [
#             transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label", "coarse"]),
#             transforms.CropForegroundd(keys=["image", "label", "coarse"], source_key="image"),
#             transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#             transforms.ToTensord(keys=["image", "label", "coarse"]),
#         ]
#     )
#
#     train_ds = PretrainDataset3rd(train_files, coarse_dir=coarse_dir, transform=train_transform, et_exclude=True)
#     val_loss_ds = PretrainDataset3rd(val_files, coarse_dir=coarse_dir, transform=val_loss_transform, et_exclude=True)
#     train_val_ds = PretrainDataset3rd(train_val_files, coarse_dir=coarse_dir, transform=val_transform)
#     val_ds = PretrainDataset3rd(val_files, coarse_dir=coarse_dir, transform=val_transform)
#     # test_ds = PretrainDataset(test_files, transform=val_transform)
#
#     loader = [train_ds, val_loss_ds, train_val_ds, val_ds]
#
#     return loader
