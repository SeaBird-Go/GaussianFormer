'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-05-14 16:50:03
Email: haimingzhang@link.cuhk.edu.cn
Description: Evaluate the pretraining stage.
'''
import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist
from copy import deepcopy

import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim import build_optim_wrapper
from mmengine.logging import MMLogger
from mmengine.utils import symlink
from mmseg.models import build_segmentor
from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20508")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1
    
    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))
        from misc.tb_wrapper import WrappedTBWriter
        writer = WrappedTBWriter('selfocc', log_dir=osp.join(args.work_dir, 'tf'))
        WrappedTBWriter._instance_dict['selfocc'] = writer
    else:
        writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_dataloader
    from loss import OPENOCC_LOSS

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')

    # logger.info(f'Params require grad: {[n for n, p in my_model.named_parameters() if p.requires_grad]}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model
    logger.info('done ddp model')

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        iter_resume=args.iter_resume)

    amp = cfg.get('amp', False)
    if amp:
        scaler = torch.cuda.amp.GradScaler()
        os.environ['amp'] = 'true'
    else:
        os.environ['amp'] = 'false'
    
    # resume and load
    epoch = 0
    global_iter = 0
    last_iter = 0

    logger.info('load from: ' + args.load_from)
    logger.info('work dir: ' + args.work_dir)

    if args.load_from is not None:
        ckpt = torch.load(args.load_from, map_location='cpu')
    else:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    try:
        print(raw_model.load_state_dict(state_dict, strict=False))
    except:
        from misc.checkpoint_util import refine_load_from_sd
        print(raw_model.load_state_dict(
            refine_load_from_sd(state_dict), strict=False))
        
    my_model.eval()
    os.environ['eval'] = 'true'

    from vis_utils import visualize_elements, VisElement

    save_dir = args.work_dir + f'/vis'
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop('img')
            
            with torch.cuda.amp.autocast(amp):
                result_dict = my_model(imgs=input_imgs, metas=data)

                loss_input = {
                    'metas': data,
                    'global_iter': global_iter,
                    'target_imgs': data['target_imgs']
                }
                for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                    if loss_input_val in result_dict:
                        loss_input.update({
                            loss_input_key: result_dict[loss_input_val]})
                    elif loss_input_val in data:
                        loss_input.update({
                            loss_input_key: data[loss_input_val]})
                    else:
                        pass
                
            if local_rank == 0:
                ## visualize the results
                render_rgb = result_dict['render_rgb']
                gt_img = data['target_imgs']

                vis_elements_list = [
                    VisElement(
                        gt_img[0],
                        type='rgb',
                        need_denormalize=False,
                    ),
                    VisElement(
                        render_rgb[0],
                        type='rgb',
                        need_denormalize=False,
                    )
                ]
                
                if 'render_gt_depth' in loss_input:
                    render_depth = result_dict['render_depth'].squeeze(2)
                    gt_depth = loss_input['render_gt_depth']

                    print(f"render depth: min: {render_depth.min()} max: {render_depth.max()}")
                    print(f"render_gt_depth depth: min: {gt_depth.min()} max: {gt_depth.max()}")

                    vis_elements_list.extend(
                        [
                            VisElement(
                                render_depth[0],
                                type='depth',
                            ),
                            VisElement(
                                gt_depth[0],
                                type='depth',
                                is_sparse=True,
                            )
                        ]
                    )
                    
                target_size = (render_rgb.shape[-2], render_rgb.shape[-1])  # (H, W)
                # target_size = (180, 320)
                visualize_elements(
                    vis_elements_list,
                    target_size=target_size,
                    save_dir=save_dir
                )

            ## when pretraining, we only need to visualize the results
            break
                        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--load-from', type=str, default=None)
    parser.add_argument('--iter-resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='nuscenes')
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
