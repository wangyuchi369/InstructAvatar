import argparse, os, sys, datetime, glob, logging
import numpy as np
import time
import pytz
from packaging import version
from omegaconf import OmegaConf
from functools import partial
from PIL import Image

import torch
import torchvision
from torch.utils.data import IterableDataset, DataLoader, Dataset, Subset

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from ldm.util import instantiate_from_config

import warnings
from lightning.pytorch.utilities.warnings import PossibleUserWarning
from lightning.pytorch.utilities.rank_zero import LightningDeprecationWarning
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
pl._logger.setLevel(logging.INFO)
# torch.cuda.set_per_process_memory_fraction(0.4, 0)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = LightningArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir. Compared to resume_from_checkpoint, resume additionally includes configs and logdir",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from a lightning checkpoint",
    )
    parser.add_argument(
        "--not_resume_from_checkpoint",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="force to not resume_from_checkpoint, restore from the model itself",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--only-val",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="only do validation",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--not_compile",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="force to not compile with PyTorch 2.0",
    )
    parser.add_argument(
        "-e",
        "--extra",
        type=str,
        default="",
        help="extra config for the base",
    )
    return parser


def nondefault_trainer_args(opt):
    default_parser = LightningArgumentParser()
    default_parser.add_lightning_class_args(Trainer, "default_trainer")
    default_args = default_parser.parse_args([])
    return sorted(k for k in vars(default_args.default_trainer) if getattr(opt.trainer, k) != getattr(default_args.default_trainer, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def is_iterable_dataset(dataset):
    return True if isinstance(dataset, IterableDataset) else False


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if is_iterable_dataset(dataset):
        split_size = dataset.num_shards // worker_info.num_workers
        # reset num_shards to the true number to retain reliable length information
        if hasattr(dataset, 'valid_ids'):
            dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, batch_size_val=None, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, use_worker_init_fn=False, shuffle_val_dataloader=False, shuffle_test_loader=False,
                 train_collect_fn=None, val_collect_fn=None, train_pin_memory=False, train_prefetch_factor=2):
        super().__init__()
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else min(batch_size * 2, os.cpu_count())
        self.use_worker_init_fn = use_worker_init_fn
        self.train_collect_fn = None
        self.val_collect_fn = None
        self.train_pin_memory = train_pin_memory
        self.train_prefetch_factor = train_prefetch_factor
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
            if train_collect_fn is not None:
                rank_zero_info(f"[DataModuleFromConfig] Use train_collect_fn: {train_collect_fn}")
                self.train_collect_fn = instantiate_from_config(train_collect_fn)
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
            if val_collect_fn is not None:
                rank_zero_info(f"[DataModuleFromConfig] Use val_collect_fn: {val_collect_fn}")
                self.val_collect_fn = instantiate_from_config(val_collect_fn)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_train_dataset = is_iterable_dataset(self.datasets['train'])
        if is_iterable_train_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_train_dataset else True,
                          collate_fn=self.train_collect_fn, worker_init_fn=init_fn,
                          pin_memory=self.train_pin_memory, prefetch_factor=self.train_prefetch_factor)

    def _val_dataloader(self, shuffle=False):
        if is_iterable_dataset(self.datasets['validation']) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size if self.batch_size_val is None else self.batch_size_val,
                          num_workers=self.num_workers,
                          shuffle=shuffle,
                          collate_fn=self.val_collect_fn,
                          worker_init_fn=init_fn)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = is_iterable_dataset(self.datasets['test'])
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if is_iterable_dataset(self.datasets['predict']) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("[main] Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            rank_zero_info(f"[main] Creating logdir: {self.logdir}")
            os.makedirs(self.logdir, exist_ok=True)
            rank_zero_info(f"[main] Creating ckptdir: {self.ckptdir}")
            os.makedirs(self.ckptdir, exist_ok=True)
            rank_zero_info(f"[main] Creating cfgdir: {self.cfgdir}")
            os.makedirs(self.cfgdir, exist_ok=True)
            rank_zero_info("[main] Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            rank_zero_info("[main] Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))
        # else:
        #     # ModelCheckpoint callback created log directory --- remove it
        #     if not self.resume and os.path.exists(self.logdir):
        #         dst, name = os.path.split(self.logdir)
        #         dst = os.path.join(dst, "child_runs", name)
        #         os.makedirs(os.path.split(dst)[0], exist_ok=True)
        #         try:
        #             os.rename(self.logdir, dst)
        #         except FileNotFoundError:
        #             pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, saving_img_batch_interval=20):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency  # when logging image, saving interval of batches
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TensorBoardLogger: self._tensorboard,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.saving_img_batch_interval = saving_img_batch_interval

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k][:, -1, ...])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            pl_module.logger.log_image(
                f"{split}/{k}", grid,
                step=pl_module.global_step)

    @rank_zero_only
    def _tensorboard(self, pl_module, images, batch_idx, split):
        for k in images:
            # inputs: (1, 16, 3, 256, 256)
            # reconstruction: (4, 3, 256, 256)
            # diffusion_row: (3, 4130, 1550)
            # samples: (4, 3, 256, 256)
            # progressive_row: (3, 4130, 1550)
            if len(images[k].shape) == 5:
                to_grid = images[k][0, ...]
            else:
                to_grid = images[k]
            grid = torchvision.utils.make_grid(to_grid, nrow=4)  # stay if len(shape) == 3
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            # https://www.pytorchlightning.ai/blog/tensorboard-with-pytorch-lightning
            pl_module.logger.experiment.add_image(
                f"{split}/{k}", grid,
                global_step=pl_module.global_step,
                dataformats="CHW")

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            # inputs: (1, 16, 3, 256, 256)
            # reconstruction: (4, 3, 256, 256)
            # diffusion_row: (3, 4130, 1550)
            # samples: (4, 3, 256, 256)
            # progressive_row: (3, 4130, 1550)
            if len(images[k].shape) == 5:
                to_grid = images[k][0, ...]
            else:
                to_grid = images[k]
            grid = torchvision.utils.make_grid(to_grid, nrow=4)  # stay if len(shape) == 3
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)  # (h, w, 3)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if ('val' in split) or self.check_frequency(check_idx):  # batch_idx % self.batch_freq == 0
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                # print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        # no outputs for on_train_batch_start
        if hasattr(pl_module, "log_images") and callable(pl_module.log_images):
            if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
                self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        # no outputs for on_validation_batch_start
        if hasattr(pl_module, "log_images") and callable(pl_module.log_images):
            if (not self.disabled) and (batch_idx % self.saving_img_batch_interval == 0):
                self.log_img(pl_module, batch, batch_idx, split="val")


class LatentLogger(Callback):
    def __init__(self, batch_frequency, max_latents=4, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_latents_kwargs=None, batch_frequency_val=20):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency  # when logging image, saving interval of batches
        self.max_latents = max_latents
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_latents_kwargs = log_latents_kwargs if log_latents_kwargs else {}
        self.log_first_step = log_first_step
        self.batch_frequency_val = batch_frequency_val

    @rank_zero_only
    def log_local(self, save_dir, split, latents,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "latents", split)
        os.makedirs(root, exist_ok=True)

        for ind, latents_filename in enumerate(latents['latent_filename']):
            for k in latents:
                if k == 'latent_filename':
                    continue
                # latents[k]: (bs, T, 512)
                filename = "{}_k-{}_gs-{:06}_e-{:06}_b-{:06}.npy".format(
                    latents_filename,
                    k,
                    global_step,
                    current_epoch,
                    batch_idx)
                path = os.path.join(root, filename)
                np.save(path, latents[k][ind].numpy())
                # if k == 'latent_rec':
                #     # decode to latent_infer_render100
                #     diffae_decode(path)

    def log_latent(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if ('val' in split) or self.check_frequency(check_idx):  # batch_idx % self.batch_freq == 0

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                latents = pl_module.log_latents(batch, split=split, **self.log_latents_kwargs)

            for k in latents:
                if isinstance(latents[k], list):
                    N = min(len(latents[k]), self.max_latents)
                    latents[k] = latents[k][:N]
                    continue
                # latents[k]: (bs, T, 512)
                N = min(latents[k].shape[0], self.max_latents)
                latents[k] = latents[k][:N]
                if isinstance(latents[k], torch.Tensor):
                    latents[k] = latents[k].detach().cpu()

            self.log_local(pl_module.logger.save_dir, split, latents,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                # print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        # no outputs for on_train_batch_start
        if hasattr(pl_module, "log_latents") and callable(pl_module.log_latents):
            if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
                self.log_latent(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        # no outputs for on_validation_batch_start
        if hasattr(pl_module, "log_latents") and callable(pl_module.log_latents):
            if (not self.disabled) and (batch_idx % self.batch_frequency_val == 0):
                self.log_latent(pl_module, batch, batch_idx, split="val")


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.strategy.reduce(max_memory)
            epoch_time = trainer.strategy.reduce(epoch_time)

            if trainer.current_epoch % 20 == 0:
                rank_zero_info(f"")
                rank_zero_info(f"[main] Average Epoch time: {epoch_time:.2f} seconds")
                rank_zero_info(f"[main] Average Peak memory {max_memory:.2f}MiB")
                rank_zero_info(f"[main] Beijing Time {datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai'))}")
        except AttributeError:
            pass


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.
    now = datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser.add_lightning_class_args(Trainer, "trainer")
    opt = parser.parse_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    resume_from_checkpoint = ""
    # resume_from_checkpoint only resume model weights and states (num epochs)
    # resume additionally includes configs and logdir
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        
        # if find previous logdir, then reuse last.ckpt of it
        if os.path.exists(opt.logdir):
            for sub_dir in sorted(os.listdir(opt.logdir)):
                if sub_dir.endswith(name + opt.postfix):
                    ckpt = os.path.join(opt.logdir, sub_dir, "checkpoints", "last.ckpt")
                    if os.path.isfile(ckpt):
                        resume_from_checkpoint = ckpt
                        rank_zero_info(f"[main] Find previous log dir and checkpoint: {ckpt}")
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    # resume from checkpoint, if not specified, then use the last.ckpt obtained before
    if opt.resume_from_checkpoint != "":
        if not os.path.exists(opt.resume_from_checkpoint):
            rank_zero_info("[main] Cannot find {}".format(opt.resume_from_checkpoint))
        else:
            resume_from_checkpoint = opt.resume_from_checkpoint
    resume_from_checkpoint = resume_from_checkpoint if os.path.exists(resume_from_checkpoint) else None
    if opt.not_resume_from_checkpoint: resume_from_checkpoint = None
    rank_zero_info("[main] Will resume_from_checkpoint: {}".format(resume_from_checkpoint))

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        opt.extra = opt.extra.split(",") if opt.extra != "" else []
        opt.extra = [x.strip() for x in opt.extra]
        cli = OmegaConf.from_dotlist(opt.extra)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default profiler
        trainer_config["profiler"] = None if opt.only_val else "simple"
        # default to ddp
        if "strategy" not in trainer_config:
            # trainer_config["strategy"] = "ddp_find_unused_parameters_false"
            trainer_config["strategy"] = "ddp"
        # obtain other trainer args from cli, e.g., gpus/devices, num_nodes, etc.
        for k in nondefault_trainer_args(opt):
            # check_val_every_n_epoch default to 1 in lightning
            trainer_config[k] = getattr(opt.trainer, k)
        # original use gpus instead of devices
        if not "devices" in trainer_config:
            del trainer_config["strategy"]
            rank_zero_info(f"[main] Running on CPU")
            cpu = True
        else:
            trainer_config["accelerator"] = 'gpu'
            gpuinfo = trainer_config["devices"]
            rank_zero_info(f"[main] Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        # https://github.com/Lightning-AI/lightning/issues/15248
        wandb_mode = "disabled" if not opt.debug else "offline"
        default_logger_cfgs = {
            "wandb": {
                "target": "lightning.pytorch.loggers.WandbLogger",
                "params": {
                    "mode": wandb_mode,
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": False,  # opt.debug
                    "version": nowname,
                }
            },
            "tensorboard": {
                "target": "lightning.pytorch.loggers.TensorBoardLogger",
                "params": {
                    "save_dir": logdir,
                    "name": 'tensorboard',
                    "version": nowname,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["tensorboard"]
        # default_logger_cfg = default_logger_cfgs["wandb"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "lightning.pytorch.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            rank_zero_info(f"[main] Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 2

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        rank_zero_info(f"[main] Merged modelckpt callback: {modelckpt_cfg}")

        default_progressbar_cfg = {
            "target": "lightning.pytorch.callbacks.TQDMProgressBar",
            "params": {
                "refresh_rate": 1 if opt.only_val else 10,
                "process_position": 0,
            }
        }
        if "progressbar" in lightning_config:
            progressbar_cfg = lightning_config.progressbar
        else:
            progressbar_cfg =  OmegaConf.create()
        progressbar_cfg = OmegaConf.merge(default_progressbar_cfg, progressbar_cfg)
        rank_zero_info(f"[main] Merged progressbar callback: {progressbar_cfg}")

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 200,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "latent_logger": {
                "target": "main.LatentLogger",
                "params": {
                    "batch_frequency": 200,
                    "max_latents": 4,
                    "increase_log_steps": False
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    "log_momentum": False
                }
            },
            # "cuda_callback": {
            #     "target": "main.CUDACallback"
            # },
        }
        default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})
        default_callbacks_cfg.update({'progressbar_callback': progressbar_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        # add precision plugins, only trigger if precision is set to 16
        if not cpu and hasattr(trainer_opt, "precision") and ('16' in str(trainer_opt.precision)):
            default_gradscaler_cfg = {
                    "target": "torch.cuda.amp.GradScaler",
                    "params": {
                        "enabled": False,
                    }
            }

            if "gradscaler" in lightning_config:
                gradscaler_cfg = lightning_config.gradscaler
            else:
                gradscaler_cfg = OmegaConf.create()

            gradscaler_cfg = OmegaConf.merge(default_gradscaler_cfg, gradscaler_cfg)
            rank_zero_info(f"[main] Merged gradscaler: {gradscaler_cfg}")
            gradscaler = instantiate_from_config(gradscaler_cfg)

            # default: https://github.com/Lightning-AI/lightning/blob/1.9.5/src/pytorch_lightning/trainer/connectors/accelerator_connector.py#L720
            # https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_gpt_pretraining.py#L61
            default_plugins_cfg = {
                "precision_plugin": {
                    "target": "lightning.pytorch.plugins.precision.MixedPrecisionPlugin",
                    "params": {
                        "precision": trainer_opt.precision,
                        "device": 'cuda' if hasattr(trainer_opt, 'devices') else 'cpu',
                        "scaler": gradscaler,
                    }
                },
            }
            rank_zero_info(f"[main] Merged plugins: {default_plugins_cfg}")
            trainer_kwargs["plugins"] = [instantiate_from_config(default_plugins_cfg[k]) for k in default_plugins_cfg]

        trainer = Trainer(**vars(trainer_opt), **trainer_kwargs)
        trainer.logdir = logdir

        # data
        if opt.only_val:
            # remove train dataset for accelerated validation
            del config.data.params['train']
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        rank_zero_info("[main] Data:")
        for k in data.datasets:
            rank_zero_info(f"[main] {k}: {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        opt.scale_lr = config.model.scale_learning_rate if hasattr(config.model, 'scale_learning_rate') else opt.scale_lr
        if 'num_nodes' in lightning_config.trainer:
            num_nodes = lightning_config.trainer.num_nodes
        else:
            num_nodes = 1
        if not cpu:
            if ',' in lightning_config.trainer.devices:
                # --trainer.devices 0,
                ngpu = len(lightning_config.trainer.devices.strip(",").split(','))
            else:
                # --trainer.devices 1
                ngpu = int(lightning_config.trainer.devices)
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        rank_zero_info(f"[main] Given accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        # set learning_rate in model
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * num_nodes * bs * base_lr
            rank_zero_info("[main] Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_nodes) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, num_nodes, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            rank_zero_info("[main] ++++ NOT USING LR SCALING ++++")
            rank_zero_info(f"[main] Setting learning rate to {model.learning_rate:.2e}")
        # set data_root in model
        model.train_data_root = data.datasets['train'].data_root if ('train' in data.datasets) and (hasattr(data.datasets['train'], "data_root")) else None
        model.val_data_root = data.datasets['validation'].data_root if ('validation' in data.datasets) and hasattr(data.datasets['validation'], "data_root") else None

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                rank_zero_info("[main] Summoning checkpoint.")
                rank_zero_info(f"[main] Current logdir: {logdir}")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)
        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()
        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        if opt.only_val:
            rank_zero_info(f'[main] Validating {resume_from_checkpoint}.')
            trainer.validate(model, datamodule=data, ckpt_path=resume_from_checkpoint)
            exit()
        # run
        if opt.train:
            try:
                opt.not_compile = True
                rank_zero_info(f'[main] Not compiling model.')
                if version.parse(torch.__version__) < version.parse("2.0.0") or opt.not_compile:
                    trainer.fit(model, datamodule=data, ckpt_path=resume_from_checkpoint)
                else:
                    compiled_model = torch.compile(model)
                    trainer.fit(compiled_model, datamodule=data, ckpt_path=resume_from_checkpoint)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            rank_zero_info(f'[main] Testing {resume_from_checkpoint}.')
            trainer.test(model, datamodule=data, ckpt_path=resume_from_checkpoint)
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            rank_zero_info(f"[main] Current logdir: {logdir}")
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            rank_zero_info(f"[main] Current logdir: {logdir}")
            # rank_zero_info(trainer.profiler.summary())
