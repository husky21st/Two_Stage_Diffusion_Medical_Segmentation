import time
from transform import get_loader_brats1st
import torch
import torch.nn as nn
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
from monai.losses.dice import DiceLoss
from monai.losses.tversky import TverskyLoss
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
import os
import wandb
import datetime

from args import Args

set_determinism(42)
dt = str(datetime.datetime.now())
logfile = f"Two-Stage-1st-{str(Args.FOLD)}-{dt[-15:-7]}"
logdir = f"./logs_brats/{logfile}/" if not Args.DEBUG_FLAG else f"./logs_brats/debug-{dt[:-7]}/"

model_save_path = os.path.join(logdir, "model")

warm_up = 30
LR = 1e-4
WD = 1e-5
al = 0.3
be = 0.7

x4_features = [32, 32, 64, 128, 256, 32]
x2_features = [64, 64, 128, 256, 512, 64]
normal_features = [64, 64, 128, 256, 512, 64]


class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, Args.NUM_MODALITY, Args.NUM_TARGETS, normal_features)

        self.model = BasicUNetDe(3, Args.NUM_MODALITY + Args.NUM_TARGETS, Args.NUM_TARGETS, normal_features,
                                 act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         loss_type=LossType.MSE,
                                         )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE,
                                                )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, Args.NUM_TARGETS, 78, 120, 120),
                                                                model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="two_stage_train1st.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)
        self.model = DiffUNet()

        self.best_dice = {"mean": 0.0, "wt": 0.0, "tc": 0.0, "et": 0.0}
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=WD)
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                       warmup_epochs=warm_up,
                                                       max_epochs=max_epochs)

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)
        self.tversky_loss = TverskyLoss(sigmoid=True, alpha=al, beta=be)

    def training_step(self, batch):
        image, label = self.get_input(batch)
        x_start = torch.where(label > 0, 1.0, 0.0)

        x_start = (x_start) * 2.0 - 1.0
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")

        loss_dice = self.dice_loss(pred_xstart, label)
        loss_tversky = self.tversky_loss(pred_xstart, label)
        loss_bce = self.bce(pred_xstart, label)

        pred_xstart = torch.sigmoid(pred_xstart)
        loss_mse = self.mse(pred_xstart, label)

        loss = loss_dice + loss_bce + loss_mse

        self.log("train_loss", loss, step=self.global_step)

        return {"all": loss, "dice": loss_dice, "tversky": loss_tversky, "bce": loss_bce, "mse": loss_mse}

    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]

        label = label.float()
        return image, label

    def validation_step(self, batch, th=0.5, idx=0):
        image, label = self.get_input(batch)

        output = self.model(image, pred_type="ddim_sample")

        output = torch.sigmoid(output)
        output = output.cpu().numpy()
        target = label.cpu().numpy()
        output = (output > 0.5)
        target = (target > 0)

        o = output[:, 0]
        t = target[:, 0]
        tc = dice(o, t)

        o = output[:, 2]
        t = target[:, 2]
        et = dice(o, t)

        o = output[:, 1]
        t = target[:, 1]
        wt = dice(o, t)

        return [wt, tc, et]

    def validation_end(self, mean_val_outputs, wand, train=False):
        wt, tc, et = mean_val_outputs
        mean_dice = (wt + tc + et) / 3
        if train:
            if wand is not None:
                wand.log({"train_wt": float(wt), "train_tc": float(tc), "train_et": float(et),
                          "train_mean_dice": float(mean_dice)},
                         step=self.epoch)
        else:
            self.log("wt", wt, step=self.epoch)
            self.log("tc", tc, step=self.epoch)
            self.log("et", et, step=self.epoch)

            self.log("mean_dice", mean_dice, step=self.epoch)
            if wand is not None:
                wand.log(
                    {"val_wt": float(wt), "val_tc": float(tc), "val_et": float(et), "val_mean_dice": float(mean_dice)},
                    step=self.epoch)

            if mean_dice > self.best_dice["mean"]:
                self.best_dice["mean"] = mean_dice
                save_new_model_and_delete_last(self.model,
                                               os.path.join(model_save_path,
                                                            f"best_model_{mean_dice:.4f}.pt"),
                                               delete_symbol="best_model")

            save_new_model_and_delete_last(self.model,
                                           os.path.join(model_save_path,
                                                        f"final_model_{mean_dice:.4f}.pt"),
                                           delete_symbol="final_model")

            print(f"wt is {wt}, tc is {tc}, et is {et}, mean_dice is {mean_dice}")


if __name__ == "__main__":

    env = "DDP" if not Args.DEBUG_FLAG else "pytorch"  # or env = "pytorch" if you only have one gpu.

    Max_epoch = 500
    Batch_size = 6
    Val_every = 50
    Num_gpus = 2 if not Args.DEBUG_FLAG else 1
    Device = "cuda:0"

    trainer = BraTSTrainer(env_type=env,
                           max_epochs=Max_epoch,
                           batch_size=Batch_size,
                           device=Device,
                           logdir=logdir,
                           val_every=Val_every,
                           num_gpus=Num_gpus,
                           master_port=17751,
                           training_script=__file__)
    if env == "DDP" and trainer.local_rank == 0 and not Args.DEBUG_FLAG and Args.WANDB_FLAG:
        run = wandb.init(
            # Set the project where this run will be logged
            project="Two-Stage",
            # Track hyperparameters and run metadata
            config={
                "data_dir": Args.DATA_DIR, "logdir": logdir, "model_save_path": model_save_path, "env": env,
                "max_epoch": Max_epoch, "batch_size": Batch_size, "num_gpus": Num_gpus, "lr": LR,
                "wd": WD, "warm_up": warm_up, "alpha": al, "beta": be,
            },
        )
    else:
        run = None
    time.sleep(3)
    train_ds, val_loss_ds, train_val_ds, val_ds = get_loader_brats1st(data_dir=Args.DATA_DIR, fold=Args.FOLD)
    trainer.train(train_dataset=train_ds, val_loss_dataset=val_loss_ds, train_val_dataset=train_val_ds,
                  val_dataset=val_ds, wand=run)
