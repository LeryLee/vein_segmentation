#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import os
import os.path as osp
import socket
import getpass
import argparse
import json
import pickle
import copy
from collections import OrderedDict
import glog as log
import time
import random
import numpy as np
import torch
import torch.utils.data as data
from tensorboardX import SummaryWriter
from utils.loss import dice_bce_loss
from utils.loss import compute_iou
from networks.corenet import CoRE_Net
from networks.framework import MyFrame
from utils.data_iden import ImageFolder as ImageFolder_iden
from utils.metrics import calculate_IoU_Dice

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp-dir', default='./debug/pretrain_new3', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--data-root', default='./data/LVD2021', type=str,
                        help='directory to load training or testing data')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='batch size')

    parser.add_argument('--total-epoch', default=300, type=int,
                        help='if early stopping does not perform, it will run the max epoch')
    parser.add_argument('--dataset', default='36_Holly_labels', type=str,
                        help='dataset to apply')
    parser.add_argument('--dataset-size', default=10, type=int,
                        help='dataset to apply')

    parser.add_argument('--num-subdivision-points', default=28 * 28, type=int,
                        help='number of most uncertain points selected')
    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup-steps", default=20, type=int,
                        help="Step of training to perform learning rate warmup for.")

    parser.add_argument('--initial-epoch-loss', default=100000, type=float,
                        help='initial epoch loss, a large number, so as to minimize it')
    parser.add_argument("--eval_every", default=5, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument('--num-early-stop', default=20, type=int,
                        help='stop training when the loss is not optimized for this certain epochs')
    parser.add_argument('--lambda-dice-iou-loss', default=0.5, type=float,
                        help='the penalty of dice-iou loss')
    parser.add_argument('--num-update-lr', default=10, type=int,
                        help='break when the loss is not optimized for this certain epochs and lr is lower than 5e-7')
    
    parser.add_argument('--tensorboard', action='store_true',
                        help='whether to record in tensorboardX or not '
                             'If set to True, we will record all output samples during training')
    parser.add_argument('--seed', default=1234, type=int,
                        help='random seed')
    parser.add_argument('--ssh', action='store_true',
                        help='whether or not we are executing command via ssh. '
                             'If set to True, we will not print anything to screen and only redirect them to log file')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def set_log_file(fname, file_only=False):
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    if file_only:
        # we only output messages to file, and stdout/stderr receives nothing.
        # this feature is designed for executing the script via ssh:
        # since ssh has a windowing kind of flow control, i.e., if the controller does not read data from a
        # ssh channel and its buffer fills up, the execution machine will not be able to write anything into the
        # channel and the process will be set to sleeping (S) status until someone reads all data from the channel.
        # this is not desired since we do not want to read stdout/stderr from the controller machine.
        # so, here we use a simple solution: disable output to stdout/stderr and only output messages to log file.
        log.logger.handlers[0].stream = log.handler.stream = sys.stdout = sys.stderr = open(fname, 'w', buffering=1)
    else:
        # we output messages to both file and stdout/stderr
        import subprocess
        tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))


def get_random_dir_name(prefix):
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    dirname = prefix + '-' + dirname + '-' + ''.join(random.choice(vocab) for _ in range(8))
    return dirname

def test(args, solver, writer, record, data_loader, epoch, mode='test'):   

    with torch.no_grad():
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        train_epoch_iou = 0
        index = 0

        # set model to evaluation mode
        solver.eval()

        results = list()
        gt_seg_maps = list()
        for img, mask, b_map in data_loader_iter:
            # print(img.size())
            img = img.cuda()
            solver.set_input(img, mask, b_map)
            train_loss, pred, _ = solver.test()
            train_epoch_loss += train_loss

            for i in range(len(pred)):
                results.append(pred[i].squeeze().cpu().numpy())
                gt_seg_maps.append(mask[i].squeeze().cpu().numpy())

            index = index + 1

        show_image = (img + 1.6) / 3.2
        show_mask = mask * b_map
        show_pred = pred

        train_epoch_loss = train_epoch_loss / len(data_loader_iter)
        # train_epoch_iou = train_epoch_iou / len(data_loader.dataset)
        evaluation_results = calculate_IoU_Dice(results, gt_seg_maps, logger=log)

        log.info('-'*10)
        log.info(f'{mode}_loss: {train_epoch_loss}')
        # log.info(f'{mode}_iou: {train_epoch_iou}')

        if args.tensorboard:
            writer.add_image(f'{mode}_images', show_image[0, :, :, :], epoch)
            writer.add_image(f'{mode}_masks', show_mask[0, :, :, :], epoch)
            writer.add_image(f'{mode}_prediction', show_pred[0, :, :, :], epoch)
            writer.add_scalar(f'{mode}_loss', train_epoch_loss, epoch)

        record[mode][epoch]= OrderedDict()
        # record[mode][outer_epoch]['inner_epoch'] = inner_epoch
        record[mode][epoch][f'{mode}_loss'] = train_epoch_loss

        return train_epoch_loss

def main(args):
    log.info('Host: {}, user: {}, CUDA_VISIBLE_DEVICES: {}, cwd: {}'.format(
        socket.gethostname(), getpass.getuser(), os.environ.get('CUDA_VISIBLE_DEVICES', ''), os.getcwd()))

    log.info('torch version: {}'.format(torch.__version__))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args(args)

    # dump config.json
    with open(osp.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # backup scripts
    fname = __file__
    if fname.endswith('pyc'):
        fname = fname[:-1]
    os.system('cp {} {}'.format(fname, args.exp_dir))
    os.system('cp -r *.py {}'.format(args.exp_dir))
    os.system('cp -r ./networks {}'.format(args.exp_dir))
    os.system('cp -r ./utils {}'.format(args.exp_dir))

    # set random seed before init model
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)

    # start to do things here
    log.info('Pretrain start')

    writer = None
    if args.tensorboard:
        dir_tensorboard = osp.join(args.exp_dir, 'runs')
        os.makedirs(dir_tensorboard, exist_ok=True)
        writer = SummaryWriter(osp.join(dir_tensorboard, 'results'))

    record = OrderedDict()
    record['train'] = OrderedDict()
    record['valid'] = OrderedDict()
    record['test'] = OrderedDict()

    # load training, validation and testing dataset
    log.info('loading data')
    # training data is allowed to do augmentation (random True)
    # validation and testing data do not use augmentation (random False)
    log.info('training data') 
    trainset = ImageFolder_iden(root_path=args.data_root, datasets=args.dataset, mode='train', dataset_size=args.dataset_size, is_random=True)
    log.info('valid data')
    validset = ImageFolder_iden(root_path=args.data_root, datasets=args.dataset, mode='valid', is_random=False)
    log.info('testing data')
    testset = ImageFolder_iden(root_path=args.data_root, datasets=args.dataset, mode='test', is_random=False)
    log.info('done!')
    # load data to dataloader
    batchsize = args.batch_size
    train_data_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4)

    valid_data_loader = torch.utils.data.DataLoader(
        validset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=4)

    test_data_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=4)

    # build model
    log.info('building model')
    #TODO: not use t_total to eval and save model
    iter_per_epoch = len(train_data_loader)
    max_epoch = args.total_epoch # 400
    args.t_total = max_epoch * iter_per_epoch
    solver = MyFrame(CoRE_Net, dice_bce_loss, args)

    # ready to train
    tic = time.time() 
    no_optim = 0
    total_epoch = args.total_epoch
    train_epoch_best_loss = args.initial_epoch_loss
    valid_epoch_best_loss = args.initial_epoch_loss
    # print('ready to train')
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(train_data_loader)
        train_epoch_loss = 0
        train_epoch_iou = 0
        index = 0

        # set model to training mode
        solver.train()

        results = list()
        gt_seg_maps = list()

        for img, mask, b_map in data_loader_iter:
            # print(img.size())
            img = img.cuda()
            solver.set_input(img, mask, b_map)
            train_loss, pred = solver.optimize()
            train_epoch_loss += train_loss

            # compute iou
            # logit = pred.cpu()
            for i in range(len(pred)):
                results.append(pred[i].detach().squeeze().cpu().numpy())
                gt_seg_maps.append(mask[i].detach().squeeze().cpu().numpy())
            # train_epoch_iou += compute_iou(pred.detach(), mask)
            index = index + 1
            

        show_image = (img + 1.6) / 3.2
        show_mask = mask * b_map
        show_pred = pred

        train_epoch_loss = train_epoch_loss / len(data_loader_iter)

        evaluation_results = calculate_IoU_Dice(results, gt_seg_maps, logger=log, is_printTable=False)
        

        if args.tensorboard:
            writer.add_image('train_images', show_image[0, :, :, :], epoch)
            writer.add_image('train_labels', show_mask[0, :, :, :], epoch)
            writer.add_image('train_prediction', show_pred[0, :, :, :], epoch)

            writer.add_scalar('train_loss', train_epoch_loss, epoch)


        log.info('-' * 10)
        log.info('epoch: {}, time: {}'.format(epoch, int(time.time() - tic)))
        log.info('train_loss: {}'.format(train_epoch_loss))

        record['train'][epoch] = OrderedDict()
        record['train'][epoch]['train_loss'] = train_epoch_loss

        if epoch % args.eval_every == 0: 
            # test
            valid_epoch_loss = test(args, solver, writer, record, valid_data_loader, epoch, mode='valid')
            test_epoch_loss = test(args, solver, writer, record, test_data_loader, epoch, mode='test')

            if train_epoch_loss >= train_epoch_best_loss:
                pass
            else:
                train_epoch_best_loss = train_epoch_loss
                # save the best params
                log.info('-----saving best training params-----')
                solver.save(osp.join(args.exp_dir, 'weights/train.th'))

            if valid_epoch_loss >= valid_epoch_best_loss:
                no_optim += 1
            else:
                no_optim = 0
                valid_epoch_best_loss = valid_epoch_loss
                # save the best params
                log.info('-----saving best validation params-----')
                solver.save(osp.join(args.exp_dir, 'weights/valid.th'))

        # early stop
        # if no_optim > args.num_early_stop:
        #     log.info('early stop at {} epoch, for {} epoch!'.format(epoch, no_optim))
            # break

        if no_optim > args.num_update_lr:
            if solver.old_lr < 5e-7:
                log.info('zero learning rate!')
                break
            log.info('no optim for {} epoch!'.format(no_optim))


    log.info('Finish!')
    log.info('Pretrain end')

    # finished, create empty file thus others could check whether or not this task is done
    with open(f'{args.exp_dir}/result.pkl', 'wb') as f:
        pickle.dump(record, f)
    open(osp.join(args.exp_dir, 'done'), 'a').close()


if __name__ == '__main__':
    xargs = parse_args()

    prefix = f'ds_{xargs.dataset_size}-lr_{xargs.learning_rate}'
    xargs.exp_dir = osp.join(xargs.exp_dir, get_random_dir_name(prefix))

    os.makedirs(xargs.exp_dir, exist_ok=True)

    os.makedirs(osp.join(xargs.exp_dir, 'weights'), exist_ok=True)
    os.makedirs(osp.join(xargs.exp_dir, 'modules'), exist_ok=True)

    # set log file
    set_log_file(osp.join(xargs.exp_dir, 'run.log'), file_only=xargs.ssh)

    # do the business
    main(xargs)
