# %load train.py
import os
import time
import argparse
import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import obj_from_dict, load_checkpoint, save_checkpoint
from mmcv.runner.log_buffer import LogBuffer

from mmdet.datasets import get_dataset, build_dataloader
from mmdet.models import build_detector, detectors
from utils.util import set_random_seed, batch_processor, get_current_lr
from utils.reorganize import reorganize_data
from utils import lr_scheduler as LRschedule
from utils.logger import init_logger
from utils.deep_lesion_eval import evaluate_deep_lesion
from utils.parallel_test import parallel_test
import torch
from torch.nn.utils import clip_grad
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--cfg', help='train config file path', default='./configs/fpn_msb.py')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.cfg)
    work_dir = cfg.work_dir
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device_id) for device_id in cfg.device_ids)
    log_dir = os.path.join(work_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = init_logger(log_dir)
    seed = cfg.seed
    logger.info('Set random seed to {}'.format(seed))
    set_random_seed(seed)

    train_dataset = get_dataset(cfg.data.train)
    train_data_loader = build_dataloader(train_dataset,
                                         cfg.data.imgs_per_gpu,
                                         cfg.data.workers_per_gpu,
                                         len(cfg.device_ids),
                                         dist=False,
                                        )
    val_dataset = get_dataset(cfg.data.val)
    val_data_loader = build_dataloader(val_dataset,
                                       1,
                                       cfg.data.workers_per_gpu,
                                       1,
                                       dist=False,
                                       shuffle=False
                                       )

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model = MMDataParallel(model).cuda()
    optimizer = obj_from_dict(cfg.optimizer, torch.optim, dict(params=model.parameters()))
    lr_scheduler = obj_from_dict(cfg.lr_scedule, LRschedule, dict(optimizer=optimizer))

    checkpoint_dir = os.path.join(cfg.work_dir, 'checkpoint_dir')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    start_epoch = cfg.start_epoch
    if cfg.resume_from:
        checkpoint = load_checkpoint(model, cfg.resume_from)
        start_epoch = 0
        logger.info('resumed epoch {}, from {}'.format(start_epoch, cfg.resume_from))
    
    log_buffer = LogBuffer()
    for epoch in range(start_epoch, cfg.end_epoch):
        train(train_data_loader, model, optimizer, epoch, lr_scheduler, log_buffer, cfg, logger)
        tmp_checkpoint_file = os.path.join(checkpoint_dir, 'tmp_val.pth')
        meta_dict = cfg._cfg_dict
        logger.info('save tmp checkpoint to {}'.format(tmp_checkpoint_file))
        save_checkpoint(model, tmp_checkpoint_file, optimizer, meta=meta_dict)
        if len(cfg.device_ids) == 1:
            sensitivity = val(val_data_loader, model, cfg, logger, epoch)
        else:
            model_args = cfg.model.copy()
            model_args.update(train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
            model_type = getattr(detectors, model_args.pop('type'))
            results = parallel_test(cfg,
                                    model_type,
                                    model_args,
                                    tmp_checkpoint_file,
                                    val_dataset,
                                    np.arange(len(cfg.device_ids)).tolist(),
                                    workers_per_gpu=1,
                                    )

            sensitivity = evaluate_deep_lesion(results, val_dataset, cfg.cfg_3dce, logger)
        save_file = os.path.join(checkpoint_dir, 'epoch_{}_sens@4FP_{:.5f}_{}.pth'.format(epoch + 1, sensitivity,
                                                                                     time.strftime('%m-%d-%H-%M',
                                                                                                   time.localtime(
                                                                                                       time.time()))))
        os.rename(tmp_checkpoint_file, save_file)
        logger.info('save checkpoint to {}'.format(save_file))
        if epoch > cfg.lr_scedule.T_max:
            os.remove(save_file)

def train(data_loader, model, optimizer, epoch, lr_scheduler, log_buffer, cfg, logger):
    log_buffer.clear()
    model.train()
    end = time.time()
    start = time.time()
    for i, data_batch in enumerate(data_loader):
        data_batch = reorganize_data(data_batch, cfg.cfg_3dce.num_images_3dce, cfg.cfg_3dce.num_slices)
        losses = batch_processor(model, data_batch)
        lr_scheduler.step(i / len(data_loader) + epoch)

        optimizer.zero_grad()
        losses['loss'].backward()
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                  max_norm=cfg.clip_grad.max_norm)
        optimizer.step()

        if not isinstance(losses, dict):
            raise TypeError('batch_processor() must return a dict')
        if 'log_vars' in losses:
            log_buffer.update(losses['log_vars'], losses['num_samples'])
        lr = get_current_lr(optimizer)
        log_str = 'Epoch [{}][{}/{}]\tlr:{:.5f}, '.format(epoch + 1, i + 1, len(data_loader), lr)
        log_buffer.update({'batch_time': time.time() - end})
        epoch_time = time.time() - start
        end = time.time()

        if (i + 1) % cfg.print_interval == 0:
            log_buffer.average(cfg.print_interval)
            log_items = []
            for name, val in log_buffer.output.items():
                log_items.append('{}: {:.2f}'.format(name, val))
            log_str += ', '.join(log_items)
            log_str += ', epoch_time:{:.2f}'.format(epoch_time)
            logger.info(log_str)
            log_buffer.clear_output()


def val(data_loader, model, cfg, logger, epoch):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(data_loader))
    with torch.no_grad():
        for i, data_batch in enumerate(data_loader):
            data_batch = reorganize_data(data_batch, cfg.cfg_3dce.num_images_3dce, cfg.cfg_3dce.num_slices)
            result = model(return_loss=False, rescale=True, **data_batch)
            results.append(result)
            batch_size = 1
            for _ in range(batch_size):
                prog_bar.update()
    # eval val
    sensitivity = evaluate_deep_lesion(results, dataset, cfg.cfg_3dce, logger)
    return sensitivity

if __name__ == '__main__':
    main()