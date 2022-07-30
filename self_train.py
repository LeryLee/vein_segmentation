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
import h5py
import pickle 
import time
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
from networks.corenet import CoRE_Net
from networks.framework import MyFrame
from utils.data_selftrain import ImageFolder as ImageFolder_selftrain
import utils.infer as my_infer
from utils.metrics import calculate_IoU_Dice


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp-dir', default='./debug/selftrain', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--data-root', default='./data/LVD2021', type=str,
                        help='directory to load training or testing data')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='batch size')

    parser.add_argument('--total-epoch', default=30, type=int,
                        help='if early stopping does not perform, it will run the max epoch')
    parser.add_argument('--total-inner-epoch', default=10, type=int,
                        help='the number of inner iteration epoch during self-training')
    parser.add_argument('--dataset', default='36_Holly_labels', type=str,
                        help='dataset to apply')

    parser.add_argument('--if-threshold', default='0.99,0.05', type=str,
                        help='the inference threshold of the head1 output, '
                             '[0] higher border, [1] lower border')
    parser.add_argument('--load-pretrained-path', type=str,
                        help='the loading path of pretrained model parameters')
    parser.add_argument('--mask-point-on', action='store_true',
                        help='whether to use point refiner.')
    parser.add_argument('--confidence-on', action='store_true',
                        help='whether to use CMM.')
    parser.add_argument('--point-correction-on', action='store_true',
                        help='whether to use PCM.')
    parser.add_argument('--num-subdivision-points', default=28*28, type=int,
                        help='number of most uncertain points selected')
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup-steps", default=10, type=int,
                        help="Step of training to perform learning rate warmup for.")

    parser.add_argument('--initial-epoch-loss', default=100000, type=float,
                        help='initial epoch loss, a large number, so as to minimize it')
    parser.add_argument('--num-early-stop', default=20, type=int,
                        help='stop training when the loss is not optimized for this certain epochs')
    parser.add_argument('--num-update-lr', default=10, type=int,
                        help='break when the loss is not optimized for this certain epochs and lr is lower than 5e-7')
    parser.add_argument('--lambda-dice-iou-loss', default=0.5, type=float,
                        help='the penalty of dice-iou loss')
    
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


def get_random_dir_name():
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    dirname = dirname + '-' + ''.join(random.choice(vocab) for _ in range(8))
    return dirname


def test(args, solver, writer, record, data_loader, outer_epoch, inner_epoch, mode='test'):

    with torch.no_grad():
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        index = 0

        # set model to evaluation mode
        solver.eval()

        results = list()
        gt_seg_maps = list()
        for img, mask, b_map in data_loader_iter:
            # print(img.size())
            img = img.cuda()
            solver.set_input(img, mask)
            if args.mask_point_on:
                train_loss, pred, _ = solver.test()
            else:
                train_loss, pred = solver.test()
            train_epoch_loss += train_loss
            index = index + 1
            for i in range(len(pred)):
                results.append(pred[i].squeeze().cpu().numpy())
                gt_seg_maps.append(mask[i].squeeze().cpu().numpy())

        show_image = (img + 1.6) / 3.2
        show_mask = mask * b_map
        show_pred = pred

        train_epoch_loss = train_epoch_loss / len(data_loader_iter)
        iters = outer_epoch * args.total_inner_epoch + inner_epoch
        evaluation_results = calculate_IoU_Dice(results, gt_seg_maps, logger=log)

        log.info('-'*10)
        log.info(f'{mode}_loss: {train_epoch_loss}')

        if args.tensorboard:
            writer.add_image(f'{mode}_images', show_image[0, :, :, :], iters)
            writer.add_image(f'{mode}_masks', show_mask[0, :, :, :], iters)
            writer.add_image(f'{mode}_prediction', show_pred[0, :, :, :], iters)
            writer.add_scalar(f'{mode}_loss', train_epoch_loss, iters)
        
        record[mode][outer_epoch][inner_epoch] = OrderedDict()
        # record[mode][outer_epoch]['inner_epoch'] = inner_epoch
        record[mode][outer_epoch][inner_epoch][f'{mode}_loss'] = train_epoch_loss.cpu()

        train_epoch_metric = OrderedDict()
        train_epoch_metric['loss'] = train_epoch_loss
        return train_epoch_metric

def Self_Train(args, solver, trainset, validset, testset, outer_epoch, writer, record, train_epoch_best_metric, valid_epoch_best_metric):

    tic = time.time()
    log.info('========start inference======= time: {}'.format(int(time.time() - tic)))
    trainset = my_infer.infer_labels(args, trainset, testset, solver, outer_epoch)
    log.info('========inference end======= time: {}'.format(int(time.time() - tic)))
    #trainset = sample_balance_dataset(unbalance_data, dataset)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4)

    valid_loader = torch.utils.data.DataLoader(
        validset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4)
    
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4)

    no_optim = 0
    log.info('========outer epoch: %d=========' % outer_epoch)
    log.info('========before retrain========show the valid and test loss=======')

    record['train'][outer_epoch] = OrderedDict()
    record['valid'][outer_epoch] = OrderedDict()
    record['test'][outer_epoch] = OrderedDict()

    valid_epoch_metric = test(args, solver, writer, record, valid_loader, outer_epoch, inner_epoch=-1, mode='valid')
    test_epoch_metric = test(args, solver, writer, record, test_loader, outer_epoch, inner_epoch=-1, mode='test')
    log.info('========show valid and test loss done!==========')

    valid_epoch_best_metric = valid_epoch_best_metric
    train_epoch_best_metric = None

    log.info('ready to do self-training')

    for inner_epoch in range(args.total_inner_epoch):
        # print('ready to train')
        solver.train()
        train_epoch_loss = 0
        train_epoch_metric = OrderedDict()

        results = list()
        gt_seg_maps = list()

        for batch_idx, (img, mask, b_map) in enumerate(train_loader):
            img = img.cuda()
            solver.set_input(img, mask, b_map)
            train_loss, pred = solver.optimize()
            train_epoch_loss += train_loss

            for i in range(len(pred)):
                results.append(pred[i].detach().squeeze().cpu().numpy())
                gt_seg_maps.append(mask[i].detach().squeeze().cpu().numpy())

        show_image = (img + 1.6) / 3.2
        show_mask = mask * b_map
        show_pred = pred

        train_epoch_loss = train_epoch_loss / (batch_idx + 1)
        iters = outer_epoch * args.total_inner_epoch + inner_epoch
        evaluation_results = calculate_IoU_Dice(results, gt_seg_maps, logger=log)

        if args.tensorboard:
            writer.add_image('train_images', show_image[0, :, :, :], iters)
            writer.add_image('train_labels', show_mask[0, :, :, :], iters)
            writer.add_image('train_prediction', show_pred[0, :, :, :], iters)
            writer.add_scalar('train_loss', train_epoch_loss, iters)
        log.info('-'*10)
        log.info('outer epoch: {}, inner epoch: {}, time: {}'.format(outer_epoch, inner_epoch, int(time.time() - tic)))
        
        log.info('train_loss: {}'.format(train_epoch_loss))

        record['train'][outer_epoch][inner_epoch] = OrderedDict()
        
        record['train'][outer_epoch][inner_epoch]['train_loss'] = train_epoch_loss.cpu()
        
        train_epoch_metric['loss'] = train_epoch_loss
        # test
        # if inner_epoch % 5 == 0:
        valid_epoch_metric = test(args, solver, writer, record, valid_loader, outer_epoch, inner_epoch, mode='valid')
        test_epoch_metric = test(args, solver, writer, record, test_loader, outer_epoch, inner_epoch, mode='test')


        if train_epoch_best_metric is None:
            train_epoch_best_metric = train_epoch_metric
            if not osp.exists(osp.join(args.exp_dir, 'weights/train.th')):
                solver.save(osp.join(args.exp_dir, 'weights/train.th'))
            if not osp.exists(osp.join(args.exp_dir, 'weights/valid.th')):
                solver.save(osp.join(args.exp_dir, 'weights/valid.th'))
        
        if train_epoch_metric['loss'] >= train_epoch_best_metric['loss']:
            pass
        else:
            log.info('-----saving best training params-----')
            solver.save(osp.join(args.exp_dir, 'weights/train.th'))


        if valid_epoch_metric['loss'] >= valid_epoch_best_metric['loss']:
            no_optim += 1
        else:
            no_optim = 0
            log.info('-----saving best validation params-----')
            solver.save(osp.join(args.exp_dir, 'weights/valid.th'))

        # early stop
        # if no_optim > args.num_early_stop:
        #     log.info('early stop at %d epoch' % inner_epoch)
        #     # break

        if no_optim > args.num_update_lr:
            if solver.old_lr < 5e-7:
                log.info('zero learning rate!')
                break
            log.info('----no optim for {} epoch----'.format(args.num_update_lr))

    log.info('Outer Finish!')

    solver_save_name = osp.join(args.exp_dir, 'weights/train.th')

    return solver_save_name, train_epoch_best_metric, valid_epoch_best_metric


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
    log.info('Self-training start')

    writer = None
    if args.tensorboard:
        dir_tensorboard = osp.join(args.exp_dir, 'runs')
        os.makedirs(dir_tensorboard, exist_ok=True)
        writer = SummaryWriter(osp.join(dir_tensorboard, 'results'))
    
    record = OrderedDict()
    record['train'] = OrderedDict()
    record['valid'] = OrderedDict()
    record['test'] = OrderedDict()

    # build model
    log.info('building model with evalmode')

    args.mask_point_on = args.confidence_on or args.point_correction_on
    solver = MyFrame(CoRE_Net, dice_bce_loss, args, evalmode=True, pointmode=args.mask_point_on)
    solver_save_path = args.load_pretrained_path

    # load training, validation and testing dataset
    log.info('loading data') 
    # training data is allowed to do augmentation (random True)
    # But the first epoch we set it to False, so as to obtain the infered label
    # after that we set it to True again, such as to do the self-training
    # validation and testing data do not use augmentation (random False)
    log.info('training data')
    trainset = ImageFolder_selftrain(root_path=args.data_root, datasets=args.dataset, mode='retrain', is_random=False)
    log.info('valid data')
    validset = ImageFolder_selftrain(root_path=args.data_root, datasets=args.dataset, mode='valid', is_random=False)
    log.info('testing data')
    testset = ImageFolder_selftrain(root_path=args.data_root, datasets=args.dataset, mode='test', is_random=False)
    log.info('done!')

    # initialize loss
    train_epoch_best_metric = OrderedDict()
    valid_epoch_best_metric = OrderedDict()
    
    train_epoch_best_metric['loss'] = args.initial_epoch_loss
    valid_epoch_best_metric['loss'] = args.initial_epoch_loss

    for epoch in range(args.total_epoch):
        # load the params
        log.info('loading retrained model')
        solver.load(solver_save_path)
        # self training
        solver_save_path, train_epoch_best_metric, valid_epoch_best_metric = Self_Train(args,
                                                                                    solver,
                                                                                    trainset,
                                                                                    validset,
                                                                                    testset,
                                                                                    epoch,
                                                                                    writer,
                                                                                    record,
                                                                                    train_epoch_best_metric,
                                                                                    valid_epoch_best_metric)

    log.info('Self-training end')
    # finished, create empty file thus others could check whether or not this task is done
    with open(f'{args.exp_dir}/result.pkl', 'wb') as f:
        pickle.dump(record, f)
    open(osp.join(args.exp_dir, 'done'), 'a').close()


if __name__ == '__main__':
    xargs = parse_args()

    xargs.exp_dir = osp.join(xargs.exp_dir, get_random_dir_name())
    os.makedirs(xargs.exp_dir, exist_ok=True)

    os.makedirs(osp.join(xargs.exp_dir, 'weights'), exist_ok=True)
    os.makedirs(osp.join(xargs.exp_dir, 'modules'), exist_ok=True)

    # set log file
    set_log_file(osp.join(xargs.exp_dir, 'run.log'), file_only=xargs.ssh)

    main(xargs)
