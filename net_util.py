from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
from copy import deepcopy
from torch.autograd import Variable




def set_parameters(opts):
    '''
    This function is called before training/testing to set parameters
    :param opts:
    :return opts:
    '''

    if not opts.__contains__('train_losses'):
        opts.train_losses=[]

    if not opts.__contains__('train_accuracies'):
        opts.train_accuracies = []

    if not opts.__contains__('valid_losses'):
        opts.valid_losses = []
    if not opts.__contains__('valid_accuracies'):
        opts.valid_accuracies = []

    if not opts.__contains__('test_losses'):
        opts.test_losses = []
    if not opts.__contains__('test_accuracies'):
        opts.test_accuracies = []

    if not opts.__contains__('best_acc'):
        opts.best_acc = 0.0

    if not opts.__contains__('lowest_loss'):
        opts.lowest_loss = 1e4

    if not opts.__contains__('checkpoint_path'):
        opts.checkpoint_path = 'checkpoint'

    if not os.path.exists(opts.checkpoint_path):
        os.mkdir(opts.checkpoint_path)

    if not opts.__contains__('checkpoint_epoch'):
        opts.checkpoint_epoch = 5








def train_net(net,opts):



    print('training at epoch {}'.format(opts.epoch+1))


    if opts.use_gpu:
        net=net.to(opts.device)
        if opts.multi_gpu:
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    net.train()



    train_loss = 0
    total_time=0
    data_time=0
    total=1e-3
    correct=0
    extra=0.

    optimizer=opts.current_optimizer



    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(opts.data_loader):
        #ff

        if opts.use_gpu:
            inputs=inputs.to(opts.device)
            targets = targets.to(opts.device)

        data_time += (time.time() - end_time)#loading time
        optimizer.zero_grad()  # flush

        if opts.arch != 'cvnet' and opts.arch != 'cv_resnet' and opts.arch != 'preact' and opts.arch != 'prototype':
            outputs = net(inputs)
        else:
            msg = None
            if (batch_idx %  opts.viz_T  == 0) and opts.visualize and (opts.epoch % 5 == 0):
                msg = ("train_ep%d_id%d" % (opts.epoch,batch_idx))
            outputs = net(inputs,targets,msg)

        loss = opts.criterion(outputs, targets)

        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()


        #bp
        loss.backward()
        optimizer.step()

        total_time += (time.time() - end_time)
        end_time = time.time()


        if opts.msg:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))



        opts.train_batch_logger.log({
            'epoch': (opts.epoch+1),
            'batch': batch_idx + 1,
            'loss': train_loss / (batch_idx + 1),
            'acc': correct / total,
            'extra': extra/(batch_idx + 1)
        })

    train_loss /= (batch_idx + 1)

    opts.train_epoch_logger.log({
        'epoch': (opts.epoch+1),
        'loss': train_loss,
        'acc': correct / total,
        'time': total_time,
        'extra': extra / (batch_idx + 1)
    })


    print('Loss: %.3f | Acc: %.3f%% (%d/%d), elasped time: %3.f seconds.'
          % (train_loss, 100. * correct / total, correct, total, total_time))
    opts.train_accuracies.append(correct / total)

    opts.train_losses.append(train_loss)




def eval_net(net,opts):
    if opts.validating:
        print('Validating at epoch {}'.format(opts.epoch + 1))

    if opts.testing:
        print('Testing at epoch {}'.format(opts.epoch + 1))


    if opts.use_gpu:
        net=net.to(opts.device)
        if opts.multi_gpu:
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))


    if not opts.__contains__('validating'):
        opts.validating = False
    if not opts.__contains__('testing'):
        opts.testing = False


    net.eval()
    eval_loss = 0
    correct = 0
    total = 1e-3
    total_time = 0
    extra=0.



    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(opts.data_loader):

        with torch.no_grad():

            if opts.use_gpu:
                inputs = inputs.to(opts.device)
                targets = targets.to(opts.device)

                if opts.arch != 'cvnet' and opts.arch != 'cv_resnet' and opts.arch != 'preact' and opts.arch != 'prototype':
                    outputs = net(inputs)
                else:
                    msg = None
                    if (batch_idx % opts.viz_T == 1) and opts.visualize and (opts.epoch % 5 == 0):
                        msg = ("test_ep%d_id%d" % (opts.epoch, batch_idx))
                    outputs = net(inputs, targets, msg)

                loss = opts.criterion(outputs, targets)

            eval_loss += loss.item()


            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

            total_time += (time.time() - end_time)
            end_time = time.time()

            if opts.msg:
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (eval_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    eval_loss /= (batch_idx + 1)
    extra/= (batch_idx + 1)
    eval_acc = correct / total


    if  opts.testing:
        opts.test_losses.append(eval_loss)
        opts.test_accuracies.append(eval_acc)

        opts.test_epoch_logger.log({
            'epoch': (opts.epoch + 1),
            'loss': eval_loss,
            'acc': correct / total,
            'time': total_time,
            'extra': extra/(batch_idx + 1)
        })

    if  opts.validating:
        opts.valid_losses.append(eval_loss)
        opts.valid_accuracies.append(eval_acc)

        opts.valid_epoch_logger.log({
            'epoch': (opts.epoch + 1),
            'loss': eval_loss,
            'acc': correct / total,
            'time': total_time,
            'extra': extra
        })
    # Save checkpoint.

    states = {
        'state_dict': net.state_dict(),
        'epoch': opts.epoch+1,
        'train_losses': opts.train_losses,
        'optimizer': opts.current_optimizer.state_dict()
    }

    if opts.__contains__('acc'):
        states['acc']=eval_acc,

    if opts.__contains__('valid_losses'):
        states['valid_losses']=opts.valid_losses
    if opts.__contains__('test_losses'):
        states['test_losses'] = opts.test_losses

    if eval_acc > opts.best_acc:
        if not os.path.isdir(opts.checkpoint_path):
            os.mkdir(opts.checkpoint_path)
        torch.save(states, os.path.join(opts.checkpoint_path, 'best_net.pth'))
        opts.best_acc = eval_acc


    if opts.epoch % opts.checkpoint_epoch == 0:
        save_file_path = os.path.join(opts.checkpoint_path, 'save_{}.pth'.format(opts.epoch))
        torch.save(states, save_file_path)

    print('Loss: %.3f | Acc: %.3f%% (%d/%d), elasped time: %3.f seconds. Best Acc: %.3f%%'
          % (eval_loss , 100. * correct / total, correct, total, total_time, opts.best_acc*100))




import csv

class Logger(object):
    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()
