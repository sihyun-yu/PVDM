import os
import json

import torch

from tools.trainer import first_stage_train
from tools.dataloader import get_loaders
from models.autoencoder.autoencoder_vit import ViTAutoencoder 
from losses.perceptual import LPIPSWithDiscriminator

from utils import file_name, Logger

#----------------------------------------------------------------------------

_num_moments    = 3             # [num_scalars, sum_of_scalars, sum_of_squares]
_reduce_dtype   = torch.float32 # Data type to use for initial per-tensor reduction.
_counter_dtype  = torch.float64 # Data type to use for the internal counters.
_rank           = 0             # Rank of the current process.
_sync_device    = None          # Device to use for multiprocess communication. None = single-process.
_sync_called    = False         # Has _sync() been called yet?
_counters       = dict()        # Running counters on each device, updated by report(): name => device => torch.Tensor
_cumulative     = dict()        # Cumulative counters on the CPU, updated by _sync(): name => torch.Tensor

#----------------------------------------------------------------------------

def init_multiprocessing(rank, sync_device):
    r"""Initializes `torch_utils.training_stats` for collecting statistics
    across multiple processes.
    This function must be called after
    `torch.distributed.init_process_group()` and before `Collector.update()`.
    The call is not necessary if multi-process collection is not needed.
    Args:
        rank:           Rank of the current process.
        sync_device:    PyTorch device to use for inter-process
                        communication, or None to disable multi-process
                        collection. Typically `torch.device('cuda', rank)`.
    """
    global _rank, _sync_device
    assert not _sync_called
    _rank = rank
    _sync_device = sync_device

#----------------------------------------------------------------------------

def first_stage(rank, args):
    device = torch.device('cuda', rank)

    temp_dir = './'
    if args.n_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.n_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.n_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.n_gpus > 1 else None
    init_multiprocessing(rank=rank, sync_device=sync_device)

    """ ROOT DIRECTORY """
    if rank == 0:
        fn = file_name(args)
        logger = Logger(fn)
        logger.log(args)
        logger.log(f'Log path: {logger.logdir}')
        rootdir = logger.logdir
    else:
        logger = None

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    """ Get Image """
    if rank == 0:
        log_(f"Loading dataset {args.data} with resolution {args.res}")
    train_loader, test_loader, total_vid = get_loaders(rank, args.data, args.res, args.timesteps, args.skip, args.batch_size, args.n_gpus, args.seed, cond=False)

    """ Get Model """
    if rank == 0:
        log_(f"Generating model")

    torch.cuda.set_device(rank)
    model = ViTAutoencoder(args.embed_dim, args.ddconfig)
    model = model.to(device)

    criterion = LPIPSWithDiscriminator(disc_start   = args.lossconfig.params.disc_start,
                                       timesteps    = args.ddconfig.timesteps).to(device)


    opt = torch.optim.AdamW(model.parameters(), 
                             lr=args.lr, 
                             betas=(0.5, 0.9)
                             )

    d_opt = torch.optim.AdamW(list(criterion.discriminator_2d.parameters()) + list(criterion.discriminator_3d.parameters()), 
                             lr=args.lr, 
                             betas=(0.5, 0.9))

    if args.resume and rank == 0:
        model_ckpt = torch.load(os.path.join(args.first_stage_folder, 'model_last.pth'))
        model.load_state_dict(model_ckpt)
        opt_ckpt = torch.load(os.path.join(args.first_stage_folder, 'opt.pth'))
        opt.load_state_dict(opt_ckpt)

        del model_ckpt
        del opt_ckpt

    if rank == 0:
        torch.save(model.state_dict(), rootdir + f'net_init.pth')

    if args.n_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[device], 
                                                          broadcast_buffers=False,
                                                          find_unused_parameters=False)
        criterion = torch.nn.parallel.DistributedDataParallel(criterion,
                                                      device_ids=[device],
                                                      broadcast_buffers=False,
                                                      find_unused_parameters=False)

    fp = args.amp
    first_stage_train(rank, model, opt, d_opt, criterion, train_loader, test_loader, args.first_model, fp, logger)

    if rank == 0:
        torch.save(model.state_dict(), rootdir + f'net_meta.pth')
