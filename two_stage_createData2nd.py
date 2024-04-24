import numpy as np
from transform import cross_valid
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.trainer import Trainer
from monai.utils import set_determinism
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
import os
import cv2
from torch.utils.data import Dataset
import SimpleITK as sitk
from monai import transforms
import random

from args import Args

set_determinism(42)
model_name = f'Two-Stage-2nd-{Args.FOLD}-~~:~~:~~'
param = 'model/best_model_0.~~~~.pt'
logdir = './logs_brats/' + model_name + '/' + param
coarse_dir = f"two_stage_1st_datas/Two-Stage-1st-{Args.FOLD}-~~:~~:~~"

max_epoch = 300
batch_size = 4
val_every = 10
device = "cuda:0"


class preDataset(Dataset):
    def __init__(self, datalist, coarse_dir, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.coarse_dir = coarse_dir

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

        image_data = [sitk.GetArrayFromImage(sitk.ReadImage(p)) for p in image_paths]
        coarse_data = sitk.GetArrayFromImage(sitk.ReadImage(coarse_path))

        image_data = np.array(image_data).astype(np.float32)
        coarse_data = np.array(coarse_data).astype(np.float32)
        tmp_data = np.zeros_like(image_data)
        return [image_data, coarse_data, tmp_data, f"BraTS-GLI-seg-{file_identifizer}-{file_idx}.nii.gz"]

    def __getitem__(self, i):
        img, coarse_img, null_label, path = self.read_data(self.datalist[i])
        image = {"image": img, "coarse": coarse_img, "label": null_label}
        if self.transform is not None:
            image = self.transform(image)
        image["path"] = path
        return image

    def __len__(self):
        return len(self.datalist)


def save_mri(mri, data_kind='tmp', data_idx=0):
    if Args.DEBUG_FLAG:
        return
    os.makedirs(os.path.join('test_datas', model_name, data_kind, str(data_idx)), exist_ok=True)
    m_min = mri.min()
    m_max = mri.max()
    for i, mri_s in enumerate(mri):
        zero_check = np.where(mri_s == 0, 0, 1)
        img = (mri_s - m_min) / (m_max - m_min) * 255
        img *= zero_check
        detects = int(mri_s.sum())
        cv2.imwrite(
            os.path.join('test_datas', model_name, data_kind, str(data_idx), str(i) + '_' + str(detects) + '.png'), img)


def compute_uncer(pred_out):
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.001] = 0.001
    uncer_out = - pred_out * torch.log(pred_out)
    return uncer_out


class DiffUNet(nn.Module):
    ddim_steps = 50

    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, Args.NUM_MODALITY + Args.NUM_TARGETS, Args.NUM_TARGETS,
                                            [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(3, Args.NUM_MODALITY + Args.NUM_TARGETS * 2, Args.NUM_TARGETS,
                                 [64, 64, 128, 256, 512, 64],
                                 act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         loss_type=LossType.MSE,
                                         )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [self.ddim_steps]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE,
                                                )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            # embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embedding=embedding)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            uncer_step = 5
            sample_outputs = []
            for i in range(uncer_step):
                sample_outputs.append(
                    self.sample_diffusion.ddim_sample_loop(self.model, (1, Args.NUM_TARGETS, 96, 96, 96),
                                                           model_kwargs={"image": image, "embeddings": embeddings}))

            sample_return = torch.zeros((1, Args.NUM_TARGETS, 96, 96, 96))

            for index in range(self.ddim_steps):
                uncer_out = 0
                for i in range(uncer_step):
                    uncer_out += sample_outputs[i]["all_model_outputs"][index]
                uncer_out = uncer_out / uncer_step
                uncer = compute_uncer(uncer_out).cpu()

                w = torch.exp(torch.sigmoid(torch.tensor((index + 1) / self.ddim_steps)) * (1 - uncer))

                for i in range(uncer_step):
                    sample_return += w * sample_outputs[i]["all_samples"][index].cpu()

            return sample_return


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[96, 96, 96],
                                                 # mode="gaussian",
                                                 sw_batch_size=1,
                                                 overlap=0.5)
        self.model = DiffUNet()

    def get_input(self, batch):
        image = batch["image"]
        label = batch["path"]
        coarse = batch["coarse"]

        return image, coarse, label

    def validation_step(self, batch, th=0.5, idx=0):
        image, coarse, label = self.get_input(batch)

        image = torch.cat([image, coarse], dim=1)
        output = self.window_infer(image, self.model, pred_type="ddim_sample")

        output = torch.sigmoid(output)
        # output = output.cpu().numpy()
        output = (output > 0.5).float().cpu().numpy()

        # output = (output > th).float().cpu().numpy().astype(bool)
        WT = output[0, 1]  # 1,2,3
        TC = output[0, 0]  # 1,3
        ET = output[0, 2]  # 3
        mask = np.zeros_like(WT).astype(int)
        np.putmask(mask, WT, 2)
        np.putmask(mask, TC, 1)
        np.putmask(mask, ET, 3)
        crop_start = batch["foreground_start_coord"].cpu().numpy()[0]
        crop_end = batch["foreground_end_coord"].cpu().numpy()[0]
        crop_end[0] = 155 - crop_end[0]
        crop_end[1] = 240 - crop_end[1]
        crop_end[2] = 240 - crop_end[2]
        pad_width = [p for p in zip(crop_start, crop_end)]
        full_mask = np.pad(mask, pad_width, "constant")
        mask_img = sitk.GetImageFromArray(full_mask)
        os.makedirs(os.path.join('two_stage_1st_datas', model_name, 'suf'), exist_ok=True)
        sitk.WriteImage(mask_img, fileName=os.path.join('two_stage_1st_datas', model_name, 'suf', label[0]))
        return [0, 0, 0]


if __name__ == "__main__":
    trainer = BraTSTrainer(env_type="pytorch",
                           max_epochs=max_epoch,
                           batch_size=batch_size,
                           device=device,
                           val_every=val_every,
                           num_gpus=1,
                           master_port=17751,
                           training_script=__file__)

    trainer.load_state_dict(logdir)

    train_files, val_files, test_files = cross_valid(Args.DATA_DIR, k=Args.FOLD)
    test_transform = transforms.Compose(
        [
            transforms.Resized(keys=["coarse"], spatial_size=(156, 240, 240), mode="trilinear"),
            transforms.SpatialCropd(keys=["coarse"], roi_start=(0, 0, 0), roi_end=(155, 240, 240)),
            transforms.CropForegroundd(keys=["image", "label", "coarse"], source_key="image"),

            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label", "coarse"]),
        ]
    )
    ds = preDataset(test_files, coarse_dir=coarse_dir, transform=test_transform)
    trainer.validation_single_gpu(val_dataset=ds, th=0.5)
