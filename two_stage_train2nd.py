import time

from transform import get_loader_brats2nd
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
from monai.losses.dice import DiceLoss
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler

set_determinism(42)
import os
import wandb
import datetime

from args import Args

dt = str(datetime.datetime.now())
logfile = f"Two-Stage-2nd-{Args.FOLD}-{dt[-15:-7]}"
coarse_dir = f"./two_stage_1st_datas/Two-Stage-1st-{Args.FOLD}-~~:~~:~~"
logdir = f"./logs_brats/{logfile}/" if not Args.DEBUG_FLAG else f"./logs_brats/debug-{dt[:-7]}/"

model_save_path = os.path.join(logdir, "model")

LR = 1e-4
WD = 1e-5

class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, Args.NUM_MODALITY + Args.NUM_TARGETS, Args.NUM_TARGETS,
                                            [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(3, Args.NUM_MODALITY + Args.NUM_TARGETS * 2, Args.NUM_TARGETS, [64, 64, 128, 256, 512, 64],
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

            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, Args.NUM_TARGETS, 96, 96, 96),
                                                                model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="two_stage_train2nd.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[96, 96, 96],
                                                 # mode="gaussian",
                                                 sw_batch_size=1,
                                                 overlap=0.25)
        self.model = DiffUNet()

        self.best_dice = {"mean": 0.0, "wt": 0.0, "tc": 0.0, "et": 0.0}
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=WD)
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                       warmup_epochs=30,
                                                       max_epochs=max_epochs)

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

    def training_step(self, batch):
        image, label, coarse, PATH = self.get_input(batch)
        x_start = label

        x_start = (x_start) * 2 - 1
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")

        image = torch.cat([image, coarse], dim=1)
        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")

        loss_dice = self.dice_loss(pred_xstart, label)
        loss_bce = self.bce(pred_xstart, label)

        pred_xstart = torch.sigmoid(pred_xstart)
        loss_mse = self.mse(pred_xstart, label)

        loss = loss_dice + loss_bce + loss_mse

        self.log("train_loss", loss, step=self.global_step)

        return {"all": loss, "dice": loss_dice, "tversky": 0.0, "bce": loss_bce, "mse": loss_mse}

    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]
        coarse = batch["coarse"]
        PATH = batch["PATH"][0]

        label = label.float()

        return image, label, coarse, PATH

    def validation_step(self, batch, th=0.5, idx=0):
        image, label, coarse, PATH = self.get_input(batch)

        image = torch.cat([image, coarse], dim=1)
        output = self.window_infer(image, self.model, pred_type="ddim_sample")

        output = torch.sigmoid(output)

        output = (output > 0.5).float().cpu().numpy()

        target = label.cpu().numpy()

        o = output[:, 1]
        t = target[:, 1]  # ce
        wt = dice(o, t)
        # core
        o = output[:, 0]
        t = target[:, 0]
        tc = dice(o, t)
        # active
        o = output[:, 2]
        t = target[:, 2]
        et = dice(o, t)

        mean_val = (wt + tc + et) / 3.0
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
                "coarse_dir": coarse_dir,
                "max_epoch": Max_epoch, "batch_size": Batch_size, "num_gpus": Num_gpus, "lr": LR,
                "wd": WD,
            },
        )
    else:
        run = None
    time.sleep(2)
    train_ds, val_loss_ds, train_val_ds, val_ds = get_loader_brats2nd(data_dir=Args.DATA_DIR, coarse_dir=coarse_dir, fold=0)
    trainer.train(train_dataset=train_ds, val_loss_dataset=val_loss_ds, train_val_dataset=train_val_ds,
                  val_dataset=val_ds, wand=run)
