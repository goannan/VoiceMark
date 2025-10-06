from pathlib import Path
import re
import os
import itertools
from typing import Tuple

from beartype import beartype

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torchgen.gen_aoti_c_shim import base_type_to_c_type

from models import WMEmbedder
from models import WMDetector
# TODO
from myDataset import get_dataloader, audioDataset
from optimizer import get_optimizer
from torch.utils import tensorboard
from loss import *
import json
from STmodels.model import SpeechTokenizer
import time
from tqdm import tqdm
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs, DataLoaderConfiguration
from augmentation import vc_simulated_augmentation
import datetime
# helpers

def exists(val):
    return val is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def checkpoint_num_steps(checkpoint_path):
    """Returns the number of steps trained from a checkpoint based on the filename.

    Filename format assumed to be something like "/path/to/soundstorm.20000.pt" which is
    for 20k train steps. Returns 20000 in that case.
    """
    results = re.findall(r'\d+', str(checkpoint_path))

    if len(results) == 0:
        return 0

    return int(results[-1])

class WMTrainer(nn.Module):
    @beartype
    def __init__(
        self,
        generator: SpeechTokenizer,
        discriminators: dict,
        cfg,
        accelerate_kwargs: dict = dict(),
    ):
        super().__init__()
        torch.manual_seed(cfg.get('seed'))

        self.results_folder = Path(cfg.get('results_folder'))
        self.results_folder.mkdir(parents=True, exist_ok=True)
        with open(self.results_folder / 'config.json', 'w+') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)

        ddp_kwargs = DistributedDataParallelKwargs()
        dataloader_config = DataLoaderConfiguration(split_batches=cfg.get("split_batches", False))
        self.accelerator = Accelerator(
            dataloader_config=dataloader_config,
            kwargs_handlers=[ddp_kwargs],
            **accelerate_kwargs
        )

        if self.is_main:
            run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join(self.results_folder, 'logs', run_name)
            self.writer = tensorboard.SummaryWriter(log_dir)
        self.generator = generator
        self.discriminators = discriminators
        self.batch_size = cfg.get("batch_size")
        self.epochs = cfg.get("epochs")
        self.lr = cfg.get("learning_rate")
        self.num_warmup_steps = cfg.get("num_warmup_steps")
        self.steps = torch.Tensor([0])
        self.best_dev_loss = float('inf')
        self.sample_rate = cfg.get('sample_rate')
        self.showpiece_num = cfg.get('showpiece_num', 8)
        self.save_model_steps = cfg.get('save_model_steps')
        self.vad_loss_lambda = cfg.get('vad_loss_lambda')
        self.cos_loss_lambda = cfg.get('cos_loss_lambda')
        self.mel_loss_lambda = cfg.get('mel_loss_lambda')
        self.adv_loss_lambda = cfg.get('adv_loss_lambda')
        self.dec_loss_lambda = cfg.get('dec_loss_lambda')
        self.multi_scale_mel_loss_lambdas = cfg.get('multi_scale_mel_loss_lambdas')
        self.multi_scale_mel_loss_kwargs_list = []
        mult = 1
        for i in range(len(self.multi_scale_mel_loss_lambdas)):
            self.multi_scale_mel_loss_kwargs_list.append({'n_fft': cfg.get('n_fft') // mult,
                                                           'num_mels': cfg.get('num_mels'), 
                                                           'sample_rate': self.sample_rate,
                                                           'hop_size': cfg.get('hop_size') // mult, 
                                                           'win_size': cfg.get('win_size') // mult, 
                                                           'fmin': cfg.get('fmin'),
                                                           'fmax': cfg.get('fmax')})
            mult = mult * 2
        self.mel_kwargs = {'n_fft': cfg.get('n_fft'), 
                           'num_mels': cfg.get('num_mels'), 
                           'sample_rate': self.sample_rate,
                           'hop_size': cfg.get('hop_size'), 
                           'win_size': cfg.get('win_size'), 
                           'fmin': cfg.get('fmin'),
                           'fmax': cfg.get('fmax')}

        self.msg_processor = WMEmbedder(
            nbits=16,
            input_dim=1024,
            nchunk_size=4,
        )
        self.detector = WMDetector(
            1024,
            16,
            nchunk_size=4,
        ).to(self.device)

        # dataset
        with open(cfg.get("train_files"), 'r') as f:
            train_files = f.readlines()
        with open(cfg.get("valid_files"), 'r') as f:
            valid_files = f.readlines()

        self.ds = audioDataset(file_list=train_files,
                               segment_size=cfg.get("segment_size"),
                               downsample_rate=generator.downsample_rate,
                               sample_rate=self.sample_rate)
        self.valid_ds = audioDataset(file_list=valid_files,
                                     segment_size=cfg.get("segment_size"),
                                     downsample_rate=generator.downsample_rate,
                                     sample_rate=self.sample_rate,
                                     valid=True)

        self.dl = get_dataloader(self.ds, batch_size=self.batch_size, shuffle=True,
                                 drop_last=cfg.get("drop_last", True), num_workers=cfg.get("num_workers"))
        self.valid_dl = get_dataloader(self.valid_ds, batch_size=1, shuffle=False, drop_last=False, num_workers=1)

        # optimizers
        self.optim_generator = get_optimizer(generator.parameters(), lr=self.lr, wd=cfg.get("wd"), betas=cfg.get("betas"))
        self.optim_discriminators = get_optimizer(itertools.chain(*[d.parameters() for d in discriminators.values()]),
                                                  lr=self.lr, wd=cfg.get("wd"), betas=cfg.get("betas"))

        # scheduler
        num_train_steps = self.epochs * len(self.ds) // self.batch_size
        self.scheduler_generator = CosineAnnealingLR(self.optim_generator, T_max=num_train_steps)
        self.scheduler_discriminator = CosineAnnealingLR(self.optim_discriminators, T_max=num_train_steps)

        self.generator, self.optim_generator, self.optim_discriminators, self.scheduler_generator, self.scheduler_discriminator, self.dl, self.valid_dl = self.accelerator.prepare(
            self.generator, self.optim_generator, self.optim_discriminators, self.scheduler_generator, self.scheduler_discriminator, self.dl, self.valid_dl
        )

        self.discriminators = {k: self.accelerator.prepare(v) for k, v in self.discriminators.items()}

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def device(self):
        return self.accelerator.device

    def log(self, values: dict, step, type=None, **kwargs):
        if type == 'figure':
            for k, v in values.items():
                self.writer.add_figure(k, v, global_step=step)
        elif type == 'audio':
            for k, v in values.items():
                self.writer.add_audio(k, v, global_step=step, **kwargs)
        else:
            for k, v in values.items():
                self.writer.add_scalar(k, v, global_step=step)

    def save(self, path, best_dev_loss):
        if best_dev_loss < self.best_dev_loss:
            self.best_dev_loss = best_dev_loss
            torch.save(self.accelerator.get_state_dict(self.generator),
                       f'{self.results_folder}/embedder_best.pt')

        pkg = dict(
            embedder=self.accelerator.get_state_dict(self.generator),
            detectors={k: self.accelerator.get_state_dict(v) for k, v in self.discriminators.items()},
            optim_embedder=self.optim_generator.state_dict(),
            optim_detectors=self.optim_discriminators.state_dict(),
            scheduler_embedder=self.scheduler_generator.state_dict(),
            scheduler_detectors=self.scheduler_discriminator.state_dict(),
            best_dev_loss=self.best_dev_loss
        )
        torch.save(pkg, path)

    def validate(self):
        print(f'\rValidate Epoch start...')
        self.generator.eval()
        for d in self.discriminators.values():
            d.eval()

        total_mel_loss = 0.0
        total_cos_loss = 0.0
        total_adv_loss = 0.0
        total_dec_loss = 0.0
        total_vad_loss = 0.0
        num = 0

        with torch.inference_mode():
            for i, batch in enumerate(tqdm(self.valid_dl, desc="Validation", ncols=100)):
                # if i >= 10:
                #     break
                ori_audio = torch.stack(batch).to(self.device)

                # read watermark
                with open("wmpool.txt", 'r') as wmp:
                    watermark = wmp.readline()
                watermark = torch.tensor(eval(watermark), dtype=torch.float32).to(self.device)

                batch_size = ori_audio.size(0)
                message = watermark.unsqueeze(0).repeat(batch_size, 1)

                # ------------------- Generator forward -------------------

                _, wm_audio, acoustic, acoustic_wm = self.generator(
                    ori_audio, message=message, msg_processor=self.msg_processor
                )

                min_len = min(ori_audio.shape[-1], wm_audio.shape[-1])
                ori_audio = ori_audio[..., :min_len]
                wm_audio = wm_audio[..., :min_len]

                # ------------------- Audio losses -------------------
                loss_mel = sum(map(lambda mel_k:mel_k[0] * mel_loss(ori_audio, wm_audio, **mel_k[1]), zip(self.multi_scale_mel_loss_lambdas, self.multi_scale_mel_loss_kwargs_list))) * self.mel_loss_lambda
                min_len = min(acoustic_wm.shape[-1], acoustic.shape[-1])
                acoustic_wm_aligned = acoustic_wm[..., :min_len]
                acoustic_aligned = acoustic[..., :min_len]
                loss_cos = cos_loss(acoustic_wm_aligned, acoustic_aligned) * self.cos_loss_lambda

                # ------------------- Adversarial loss -------------------
                loss_adv = 0.0
                # Forward through all training discriminators (lightweight)
                for d in self.discriminators.values():
                    _, fake_preds_list, _, _ = d(ori_audio, wm_audio)  # Not detached, keep computation graph
                    # Merge all fake_pred and then compute loss
                    fake_preds_tensor = torch.cat([fp.flatten() for fp in fake_preds_list])
                    loss_adv += adversarial_loss_g(fake_preds_tensor)

                    augmented_audio, vad_labels = vc_simulated_augmentation(wm_audio, sample_rate=self.sample_rate,
                                                                             orig_audio=ori_audio)

                    # ---------------- Train Decoding loss -----------------
                    logits, chuck_logits = self.detect_watermark(augmented_audio, return_logits=True)
                    loss_dec = decoding_loss(chuck_logits, bits_to_chunks(message)) * self.dec_loss_lambda

                    # ---------------- Train vad loss --------------------------
                    min_lens = min(logits.shape[-1], vad_labels.shape[-1])
                    loss_vad = vad_based_loss(logits[..., :min_lens], vad_labels[..., :min_lens],
                                              from_logits=True) * self.vad_loss_lambda

                # ------------------- Accumulate losses -------------------
                total_mel_loss += loss_mel.item()
                total_cos_loss += loss_cos.item()
                total_adv_loss += loss_adv.item()
                total_dec_loss += loss_dec.item()
                total_vad_loss += loss_vad.item()
                num += ori_audio.size(0)

                # ------------------- Logging first few audios -------------------
                if i < self.showpiece_num and self.is_main:
                    self.log({f'groundtruth/x_{self.steps.item()}_{i}': ori_audio[0].cpu().numpy()}, type='audio',
                            sample_rate=self.sample_rate, step=int(self.steps.item()))
                    x_spec = mel_spectrogram(ori_audio.squeeze(1), **self.mel_kwargs)
                    self.log({f'groundtruth/x_spec_{self.steps.item()}_{i}': plot_spectrogram(x_spec[0].cpu().numpy())}, type='figure',
                            step=int(self.steps.item()))
                    
                    self.log({f'generate/x_hat_{self.steps.item()}_{i}': wm_audio[0].cpu().numpy()}, type='audio',
                            sample_rate=self.sample_rate, step=int(self.steps.item()))
                    x_hat_spec = mel_spectrogram(wm_audio.squeeze(1), **self.mel_kwargs)
                    self.log({f'generate/x_hat_spec_{self.steps.item()}_{i}': plot_spectrogram(x_hat_spec[0].cpu().numpy())}, type='figure',
                            step=int(self.steps.item()))

        # ------------------- Compute average losses -------------------
        avg_mel_loss = total_mel_loss / num
        avg_cos_loss = total_cos_loss / num
        avg_adv_loss = total_adv_loss / num
        avg_dec_loss = total_dec_loss / num
        avg_vad_loss = total_vad_loss / num

        # ------------------- Logging per batch -------------------
        if self.is_main:
            self.log({
                f"val/mel_loss": avg_mel_loss,
                f"val/cos_loss": avg_cos_loss,
                f"val/adv_loss": avg_adv_loss,
                f"val/decoding_loss": avg_dec_loss,
                f"val/vad_loss": avg_vad_loss,
                f"val/total_loss": avg_mel_loss + avg_cos_loss + avg_adv_loss + avg_dec_loss + avg_vad_loss
            }, step=self.steps.item())

        return avg_mel_loss, avg_cos_loss, avg_adv_loss, avg_dec_loss, avg_vad_loss

    def train(self):
        # print(torch.cuda.is_available())  # True
        # print(torch.cuda.current_device())  # 0
        # print(torch.cuda.get_device_name(0))
        print(f'Training start...')
        self.generator.train()
        for d in self.discriminators.values():
            d.train()

        steps = int(self.steps.item())
        for epoch in range(self.epochs):
            # waveform = [batch, 1, T]
            for i, audio in enumerate(tqdm(self.dl, desc=f"Epoch {epoch}", ncols=100)):
                # start_batch = time.time()  # Record the start time of the batch
                # ------------------- Prepare batch -------------------
                ori_audio = torch.stack(audio).to(self.device)
                with open("wmpool.txt", 'r') as wmp:
                    watermark = eval(wmp.readline())
                # watermark = 16 bits binary
                watermark = torch.tensor(watermark, dtype=torch.int64).to(self.device)
                batch_size = ori_audio.size(0)
                # message = [batch, 16]
                message = watermark.unsqueeze(0).repeat(batch_size, 1)

                # ------------------- Embedder forward -------------------
                # t0 = time.time()
                self.optim_generator.zero_grad()
                _, wm_audio, acoustic, acoustic_wm = self.generator(
                    ori_audio, message=message, msg_processor=self.msg_processor.to(self.device)
                )
                # t_embedder = time.time() - t0

                min_len = min(ori_audio.shape[-1], wm_audio.shape[-1])
                ori_audio = ori_audio[..., :min_len]
                wm_audio = wm_audio[..., :min_len]

                # ------------------- Train Discriminators -------------------
                # t0 = time.time()
                self.optim_discriminators.zero_grad()
                loss_D = 0.0
                for d in self.discriminators.values():
                    real_preds_list, fake_preds_list, _, _ = d(ori_audio, wm_audio.detach())
                    for real_pred, fake_pred in zip(real_preds_list, fake_preds_list):
                        loss_D += adversarial_loss_d(real_pred, fake_pred)
                self.accelerator.backward(loss_D)

                # gradient clipping for Discriminators
                # max_grad_norm = 10.0
                # for d in self.discriminators.values():
                #     torch.nn.utils.clip_grad_norm_(d.parameters(), max_norm=max_grad_norm)
                self.optim_discriminators.step()
                # t_discriminator = time.time() - t0

                # ------------------- Train mel loss -------------------
                # t0 = time.time()
                loss_mel = sum(map(lambda mel_k:mel_k[0] * mel_loss(ori_audio, wm_audio, **mel_k[1]), zip(self.multi_scale_mel_loss_lambdas, self.multi_scale_mel_loss_kwargs_list))) * self.mel_loss_lambda
                # t_mel = time.time() - t0

                # ------------------ Train cos loss ---------------------
                # t0 = time.time()
                min_len = min(acoustic_wm.shape[-1], acoustic.shape[-1])
                acoustic_wm_aligned = acoustic_wm[..., :min_len]
                acoustic_aligned = acoustic[..., :min_len]
                loss_cos = cos_loss(acoustic_wm_aligned, acoustic_aligned) * self.cos_loss_lambda
                # t_cos = time.time() - t0

                # ------------------- Train Generator -------------------
                # t0 = time.time()
                self.optim_generator.zero_grad()
                loss_adv = 0.0
                # Forward through all training discriminators (lightweight)
                for d in self.discriminators.values():
                    _, fake_preds_list, _, _ = d(ori_audio, wm_audio)  # Not detached, keep computation graph
                    # Merge all fake_pred and then compute loss
                    fake_preds_tensor = torch.cat([fp.flatten() for fp in fake_preds_list])
                    loss_adv += adversarial_loss_g(fake_preds_tensor)

                loss_adv = loss_adv * self.adv_loss_lambda
                # t_generator = time.time() - t0

                # before training decoding loss and vad loss, watermarked audio should go through distortion layer.
                # add vc simulation augmentation methods
                augmented_audio, vad_labels = vc_simulated_augmentation(wm_audio, sample_rate=self.sample_rate, orig_audio=ori_audio)

                # ---------------- Train Decoding loss -----------------
                # t0 = time.time()
                logits, chuck_logits = self.detect_watermark(augmented_audio, return_logits=True)
                loss_dec = decoding_loss(chuck_logits, bits_to_chunks(message)) * self.dec_loss_lambda
                # t_decoding = time.time() - t0

                # ---------------- Train vad loss --------------------------
                # t0 = time.time()
                min_lens = min(logits.shape[-1], vad_labels.shape[-1])
                loss_vad = vad_based_loss(logits[..., :min_lens], vad_labels[..., :min_lens], from_logits=True) * self.vad_loss_lambda
                # t_vad = time.time() - t0

                # ---------------- accumulate total loss ---------------------
                total_loss = loss_mel + loss_cos + loss_adv + loss_dec + loss_vad

                # Use accelerator backward
                self.accelerator.backward(total_loss)

                # ---------------- gradient clipping for Generator ---------------------
                # max_grad_norm = 10.0  # 
                # torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=max_grad_norm)

                total_norm = 0
                for p in self.generator.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                if self.is_main and steps % 100 == 0:
                    print(f"[Step {steps}] Grad Norm = {total_norm:.4f}")
                    self.log({"train/grad_norm": total_norm}, step=steps)

                self.optim_generator.step()

                # ------------------- Scheduler step -------------------
                # t0 = time.time()
                self.scheduler_generator.step()
                self.scheduler_discriminator.step()
                # t_scheduler = time.time() - t0

                for param_group in self.optim_generator.param_groups:
                    lr_g = param_group["lr"]
                for param_group in self.optim_discriminators.param_groups:
                    lr_d = param_group["lr"]

                self.log({"lr/generator": lr_g, "lr/discriminator": lr_d}, step=steps)

                # ------------------- Logging -------------------
                if self.is_main:
                    self.log({
                        "train/mel_loss": loss_mel.item(),
                        "train/vad_loss": loss_vad.item(),
                        "train/cos_loss": loss_cos.item(),
                        "train/decoding_loss": loss_dec.item(),
                        "train/d_loss": loss_D.item(),
                        "train/adv_loss": loss_adv.item(),
                        "train/total_loss": total_loss.item()
                    }, step=steps)

                    # Print time spent on each step
                    # print(f"Batch {i}: embedder={t_embedder:.2f}s, discriminator={t_discriminator:.2f}s, "
                    #       f"compute_losses={t_compute_losses:.2f}s, generator={t_generator:.2f}s, t_decoding={t_decoding:.2f}s, t_vad={t_vad:.2f}s, "
                    #       f"scheduler={t_scheduler:.2f}s, total={time.time() - start_batch:.2f}s")

                    # ------------------- Validation & Save -------------------
                    if steps % self.save_model_steps == 0 and steps != 0:
                    # if steps % 10 == 0 and steps != 0:
                        val_mel, val_cos, val_adv, val_dec, val_vad = self.validate()
                        self.save(
                            self.results_folder / f'WatermarkTrainer_{steps:08d}.pt',
                            val_mel + val_cos + val_adv + val_dec + val_vad
                        )
                        # Switch back to train mode
                        self.generator.train()
                        for d in self.discriminators.values():
                            d.train()
                        print(
                            f'Steps={steps}: '
                            f'Mel={val_mel:.4f}, Cos={val_cos:.4f}, Adv={val_adv:.4f}, Dec={val_dec:.4f}, Vad={val_vad:.4f} saved.\r'
                        )

                    steps += 1
                    self.steps = torch.tensor([steps], device=self.device)

    def continue_train(self):
        self.load()
        self.train()

    def detect_watermark(
        self, x: torch.Tensor, return_logits=False
    ):
        embedding = self.generator.forward_feature(x).to(self.device)
        if return_logits:
            return self.detector(embedding)
        return self.detector.detect_watermark(embedding)