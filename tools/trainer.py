import os
import random
import numpy as np
import sys; sys.path.extend([sys.path[0][:-4], '/app'])

import time
import tqdm
import copy
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


from utils import AverageMeter
from evals.eval import test_psnr, test_ifvd, test_fvd_ddpm
from models.ema import LitEma
from einops import rearrange
from torch.optim.lr_scheduler import LambdaLR


def latentDDPM(rank, first_stage_model, model, opt, criterion, train_loader, test_loader, scheduler, ema_model=None, cond_prob=0.3, logger=None):
    scaler = GradScaler()

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    if rank == 0:
        rootdir = logger.logdir

    device = torch.device('cuda', rank)

    losses = dict()
    losses['diffusion_loss'] = AverageMeter()
    check = time.time()

    #lr_scheduler = LambdaLR(opt, scheduler)
    if ema_model == None:
        ema_model = copy.deepcopy(model)
        ema = LitEma(ema_model)
        ema_model.eval()
    else:
        ema = LitEma(ema_model)
        ema.num_updates = torch.tensor(11200,dtype=torch.int)
        ema_model.eval()

    first_stage_model.eval()
    model.train()

    for it, (x, _) in enumerate(train_loader):
        x = x.to(device)
        x = rearrange(x / 127.5 - 1, 'b t c h w -> b c t h w') # videos
        c = None

        # conditional free guidance training
        model.zero_grad()

        if model.module.diffusion_model.cond_model:
            p = np.random.random()

            if p < cond_prob:
                c, x = torch.chunk(x, 2, dim=2)
                mask = (c+1).contiguous().view(c.size(0), -1) ** 2
                mask = torch.where(mask.sum(dim=-1) > 0, 1, 0).view(-1, 1, 1)

                with autocast():
                    with torch.no_grad():
                        z = first_stage_model.module.extract(x).detach()
                        c = first_stage_model.module.extract(c).detach()
                        c = c * mask + torch.zeros_like(c).to(c.device) * (1-mask)

            else:
                c, x_tmp = torch.chunk(x, 2, dim=2)
                mask = (c+1).contiguous().view(c.size(0), -1) ** 2
                mask = torch.where(mask.sum(dim=-1) > 0, 1, 0).view(-1, 1, 1, 1, 1)

                clip_length = x.size(2)//2
                prefix = random.randint(0, clip_length)
                x = x[:, :, prefix:prefix+clip_length, :, :] * mask + x_tmp * (1-mask)
                with autocast():
                    with torch.no_grad():
                        z = first_stage_model.module.extract(x).detach()
                        c = torch.zeros_like(z).to(device)

            (loss, t), loss_dict = criterion(z.float(), c.float())

        else:
            if it == 0:
                print("Unconditional model")
            with autocast():    
                with torch.no_grad():
                    z = first_stage_model.module.extract(x).detach()

            (loss, t), loss_dict = criterion(z.float())

        """
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        """
        loss.backward()
        opt.step()

        losses['diffusion_loss'].update(loss.item(), 1)

        # ema model
        if it % 25 == 0 and it > 0:
            ema(model)

        if it % 500 == 0:
            #psnr = test_psnr(rank, model, test_loader, it, logger)
            if logger is not None and rank == 0:
                logger.scalar_summary('train/diffusion_loss', losses['diffusion_loss'].average, it)

                log_('[Time %.3f] [Diffusion %f]' %
                     (time.time() - check, losses['diffusion_loss'].average))

            losses = dict()
            losses['diffusion_loss'] = AverageMeter()


        if it % 10000 == 0 and rank == 0:
            torch.save(model.module.state_dict(), rootdir + f'model_{it}.pth')
            ema.copy_to(ema_model)
            torch.save(ema_model.module.state_dict(), rootdir + f'ema_model_{it}.pth')
            fvd = test_fvd_ddpm(rank, ema_model, first_stage_model, test_loader, it, logger)


            if logger is not None and rank == 0:
                logger.scalar_summary('test/fvd', fvd, it)

                log_('[Time %.3f] [FVD %f]' %
                     (time.time() - check, fvd))


def first_stage_train(rank, model, opt, d_opt, criterion, train_loader, test_loader, first_model, fp, logger=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    if rank == 0:
        rootdir = logger.logdir

    device = torch.device('cuda', rank)
    
    losses = dict()
    losses['ae_loss'] = AverageMeter()
    losses['d_loss'] = AverageMeter()
    check = time.time()

    accum_iter = 3
    disc_opt = False

    if fp:
        scaler = GradScaler()
        scaler_d = GradScaler()

        try:
            scaler.load_state_dict(torch.load(os.path.join(first_model, 'scaler.pth')))
            scaler_d.load_state_dict(torch.load(os.path.join(first_model, 'scaler_d.pth')))
        except:
            print("Fail to load scalers. Start from initial point.")


    model.train()
    disc_start = criterion.module.discriminator_iter_start
    
    for it, (x, _) in enumerate(train_loader):

        if it > 1000000:
            break
        batch_size = x.size(0)

        x = x.to(device)
        x = rearrange(x / 127.5 - 1, 'b t c h w -> b c t h w') # videos

        if not disc_opt:
            with autocast():
                x_tilde, vq_loss  = model(x)

                if it % accum_iter == 0:
                    model.zero_grad()
                ae_loss = criterion(vq_loss, x, 
                                    rearrange(x_tilde, '(b t) c h w -> b c t h w', b=batch_size),
                                    optimizer_idx=0,
                                    global_step=it)

                ae_loss = ae_loss / accum_iter
            
            scaler.scale(ae_loss).backward()

            if it % accum_iter == accum_iter - 1:
                scaler.step(opt)
                scaler.update()

            losses['ae_loss'].update(ae_loss.item(), 1)

        else:
            if it % accum_iter == 0:
                criterion.zero_grad()

            with autocast():
                with torch.no_grad():
                    x_tilde, vq_loss = model(x)
                d_loss = criterion(vq_loss, x, 
                         rearrange(x_tilde, '(b t) c h w -> b c t h w', b=batch_size),
                         optimizer_idx=1,
                         global_step=it)
                d_loss = d_loss / accum_iter
            
            scaler_d.scale(d_loss).backward()

            if it % accum_iter == accum_iter - 1:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler_d.unscale_(d_opt)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(criterion.module.discriminator_2d.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(criterion.module.discriminator_3d.parameters(), 1.0)

                scaler_d.step(d_opt)
                scaler_d.update()

            losses['d_loss'].update(d_loss.item() * 3, 1)

        if it % accum_iter == accum_iter - 1 and it // accum_iter >= disc_start:
            if disc_opt:
                disc_opt = False
            else:
                disc_opt = True

        if it % 2000 == 0:
            fvd = test_ifvd(rank, model, test_loader, it, logger)
            psnr = test_psnr(rank, model, test_loader, it, logger)
            if logger is not None and rank == 0:
                logger.scalar_summary('train/ae_loss', losses['ae_loss'].average, it)
                logger.scalar_summary('train/d_loss', losses['d_loss'].average, it)
                logger.scalar_summary('test/psnr', psnr, it)
                logger.scalar_summary('test/fvd', fvd, it)

                log_('[Time %.3f] [AELoss %f] [DLoss %f] [PSNR %f]' %
                     (time.time() - check, losses['ae_loss'].average, losses['d_loss'].average, psnr))

                torch.save(model.module.state_dict(), rootdir + f'model_last.pth')
                torch.save(criterion.module.state_dict(), rootdir + f'loss_last.pth')
                torch.save(opt.state_dict(), rootdir + f'opt.pth')
                torch.save(d_opt.state_dict(), rootdir + f'd_opt.pth')
                torch.save(scaler.state_dict(), rootdir + f'scaler.pth')
                torch.save(scaler_d.state_dict(), rootdir + f'scaler_d.pth')

            losses = dict()
            losses['ae_loss'] = AverageMeter()
            losses['d_loss'] = AverageMeter()

        if it % 2000 == 0 and rank == 0:
            torch.save(model.module.state_dict(), rootdir + f'model_{it}.pth')

