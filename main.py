# library
from numpy.random import choice
from torch.utils.data import DataLoader
import os
import torch
import argparse
import torch.optim as optim
import json
import torch.nn as nn
from torch.optim import lr_scheduler

# local
from utils import *
from dataset import MTA, GTZAN

parser = argparse.ArgumentParser(description='Siamusic')
parser.add_argument('--save_path', default='./exp', type=str,
                    help='data path')
parser.add_argument('--dataset', default='MTA', type=str,
                    help='dataset',choice=['MTA','GTZAN'])
parser.add_argument('--input_length', default=48000, type=int,
                    help='input length')
parser.add_argument('--batch_size', default=16, type=int,
                    help='batch size')
parser.add_argument('--optim', default='adam', type=str,
                    help='optimizer', choice=['sgd','adam','adagrad'])
parser.add_argument('--lr', default=1e-3, type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=0.00001, type=float,
                    help='weight_decay')
parser.add_argument('--epochs', default=300, type=int,
                    help='train epoch')
parser.add_argument('--inference_only', default=False, type=bool,
                    action=argparse.BooleanOptionalAction,
                    help='How To Make TRUE? : --no-inference_only')
									
    

parser.add_argument('--gpu_id', default='0', type=str,
                    help='How To Check? : cmd -> nvidia-smi')
args = parser.parse_args()

# local
from dataset import MTA, GTZAN


def train():
    pass


def validation():
    pass


def test():
    pass


def main():
    # save path
    save_path=args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        # Save configuration
        with open(save_path + '/configuration.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # define architecture
    # dataset의 input ouput shape에 맞게 
    # backbone을 부르고 siam모델에 넣는 구조
    # model = Siamusic()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


    # dataset loading
    if args.dataset == 'MTA':
        train_dataset = MTA(split='train',input_length=args.input_length)
        val_dataset = MTA(split='validation',input_length=args.input_length)
        test_dataset = MTA(split='test',input_length=args.input_length)
    elif args.dataset == 'GTZAN':
        train_dataset = GTZAN(split='train',input_length=args.input_length)
        val_dataset = GTZAN(split='validation',input_length=args.input_length)
        test_dataset = GTZAN(split='test',input_length=args.input_length)
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=2,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size,num_workers=2,shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,num_workers=2,shuffle=False)
    print('=== DataLoader R.e.a.d.y ===')

    # define criterion
    criterion = nn.CrossEntropyLoss().cuda()
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    milestones = [int(args.epochs/3),int(args.epochs/2)]
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.7)

    # logger
    train_logger = Logger(os.path.join(save_path, 'train_loss.log'))
    val_logger = Logger(os.path.join(save_path, 'val_loss.log'))
    test_logger = Logger(os.path.join(save_path, 'test_loss.log'))
    
    if args.inference_only:
        pass #학습된 모델 불러와서 테스트만 진행
    else:
        pass #학습과 validation 모델저장


if __name__ == '__main__':
    main()