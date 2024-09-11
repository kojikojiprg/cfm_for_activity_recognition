import argparse
import os
import shutil
import sys
from glob import glob

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

sys.path.append(".")
from src.data import NTU_RGBD
from src.model import FlowMatching
from src.utils import yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=str)
    parser.add_argument("-p", "--pretrain", action="store_true", default=False)
    parser.add_argument("-g", "--gpu_ids", type=int, nargs="*", default=None)
    parser.add_argument("-ckpt", "--checkpoint", type=str, default=None)
    args = parser.parse_args()
    data_root = args.dataset_root
    is_pretrain = args.pretrain
    gpu_ids = args.gpu_ids
    checkpoint = args.checkpoint
    if checkpoint is not None:
        assert os.path.exists(checkpoint)

    # load config
    config_path = "configs/model.yaml"
    config = yaml_handler.load(config_path)

    # create checkpoint directory of this version
    checkpoint_dir = "models/"
    ckpt_dirs = glob(os.path.join(checkpoint_dir, "*/"))
    ckpt_dirs = [d for d in ckpt_dirs if "version_" in d]
    if len(ckpt_dirs) > 0:
        max_v_num = 0
        for d in ckpt_dirs:
            last_ckpt_dir = os.path.dirname(d)
            v_num = int(last_ckpt_dir.split("/")[-1].replace("version_", ""))
            if v_num > max_v_num:
                max_v_num = v_num
        v_num = max_v_num + 1
    else:
        v_num = 0
    checkpoint_dir = os.path.join(checkpoint_dir, f"version_{v_num}")

    if "WORLD_SIZE" not in os.environ:
        # copy config
        os.makedirs(checkpoint_dir, exist_ok=False)
        copy_config_path = os.path.join(checkpoint_dir, "model.yaml")
        shutil.copyfile(config_path, copy_config_path)

    # model checkpoint callback
    filename = "cfm"
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename=filename + "-best-{epoch}",
        monitor="loss",
        mode="min",
        save_last=True,
    )
    model_checkpoint.CHECKPOINT_NAME_LAST = filename + "-last-{epoch}"

    # load dataset
    dataset = NTU_RGBD(
        data_root, config, True, split_type="cross_subject"
    )
    dataloader = DataLoader(
        dataset,
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # create model
    model = FlowMatching(config, is_pretrain=is_pretrain)
    ddp = DDPStrategy(find_unused_parameters=False, process_group_backend="nccl")

    logger = TensorBoardLogger("logs/", name="")
    epochs = config.pre_epochs if is_pretrain else config.epochs
    trainer = Trainer(
        accelerator="cuda",
        strategy=ddp,
        devices=gpu_ids,
        logger=logger,
        callbacks=[model_checkpoint],
        max_epochs=epochs,
        benchmark=True,
    )
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=checkpoint)
