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
from tqdm import tqdm
from torchmetrics import F1,AUROC,Accuracy,Recall
import time
import sys

# local
from utils import *
from dataset import MTA, GTZAN, WAVAudio
from loss import SiamusicLoss
from augmentation import sungrae_pedal, random_mix
from simsiam import Siamusic

parser = argparse.ArgumentParser(description='Siamusic')
parser.add_argument('--save_path', default='./exp', type=str,
                    help='save path')
parser.add_argument('--backbone', default='ResNet34', type=str,
                    help='backbone network for simsiam',
                    choices=['resnet50','resnet101','resnet152','Transformer'])
parser.add_argument('--dataset', default='MTA', type=str,
                    help='dataset',choices=['MTA','GTZAN'])
parser.add_argument('--input_length', default=48000, type=int,
                    help='input length')
parser.add_argument('--batch_size', default=16, type=int,
                    help='batch size')
parser.add_argument('--optim', default='adam', type=str,
                    help='optimizer', choices=['sgd','adam','adagrad'])
parser.add_argument('--lr', default=1e-3, type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=0.00001, type=float,
                    help='weight_decay')
parser.add_argument('--epochs', default=300, type=int,
                    help='train epoch')
parser.add_argument('--augmentation', default='pedalboard', type=str,
                    help='train epoch',choices=['pedalboard','randommix','iamge'])
parser.add_argument('--from_scratch', default=False, type=bool,
                    action=argparse.BooleanOptionalAction,
                    help='How To Make TRUE? : --from_scratch, Flase : --no-from_scratch')
									
parser.add_argument('--gpu_id', default='0', type=str,
                    help='How To Check? : cmd -> nvidia-smi')
args = parser.parse_args()
start = time.time()

# siam model pretrain
def siam_train(model, trn_loader, criterion, optimizer, epoch, num_epoch, train_logger):
    model.train()
    train_loss = AverageMeter()
    for i, audio in enumerate(trn_loader):
        if args.augmentation == 'pedalboard':
            x1, x2 = sungrae_pedal(audio), sungrae_pedal(audio)
        elif args.augmentation == 'randommix':
            x1, x2 = random_mix(audio), random_mix(audio)
        elif args.augmentation == 'image':
            sys.exit('곧 업데이트 예정')
        x1, x2 = x1.cuda(), x2.cuda()
        p1, z2, p2, z1 = model(x1,x2) # backbone(+projection) + predictor
        loss = criterion(p1,z2,p2,z1)
        
        train_loss.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 and i != 0:
            print('Epoch : [{0}/{1}] [{2}/{3}]  Train Loss : {loss:.4f}'.format(
                epoch, num_epoch, i, len(trn_loader), loss=loss))
    train_logger.write([epoch, train_loss.avg])


# downstream task validation
def validation(model, val_loader, criterion, epoch, num_epochs, val_logger):
    model.eval()
    val_loss = AverageMeter()
    with torch.no_grad():
        for i, audio in enumerate(val_loader):
            audio = audio.cuda()
            x1, x2 = sungrae_pedal(audio), sungrae_pedal(audio) # 어그멘트 해야하나??
            p1,z2,p2,z1 = model(x1, x2)
            loss = criterion(p1,z2,p2,z1)
            val_loss.update(loss.item()*10000)

        print("=================== Validation Start ====================")
        print('Epoch : [{0}/{1}]  Test Loss : {loss:.4f}'.format(
                epoch, num_epochs, loss=val_loss.avg))
        print("=================== TEST(Validation) End ======================")
        val_logger.write([epoch, val_loss.avg])


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
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    model = Siamusic(backbone=args.backbone).cuda()

    
    # pre-training or fine-tuning
    if args.from_scratch: ## pre-training
        print('스크래치부터 학습됩니다.')
        # dataset loading
        train_dataset = WAVAudio(split='train',input_length=args.input_length)
        #val_dataset = WAVAudio(split='validation',input_length=args.input_length)
        train_loader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=2,shuffle=True)
        #val_loader = DataLoader(val_dataset,batch_size=args.batch_size,num_workers=2,shuffle=False)
        print('=== DataLoader R.e.a.d.y ===')

        # define criterion
        criterion = SiamusicLoss().cuda()
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
        #val_logger = Logger(os.path.join(save_path, 'val_loss.log'))

        # 학습시작
        for epoch in tqdm(range(1,args.epochs+1)):
            siam_train(model, train_loader, criterion ,optimizer, epoch, args.epochs, train_logger)
            #validation(model, val_loader, criterion, epoch, args.epochs, val_logger)
            scheduler.step()
            if epoch%20 == 0 or epoch == args.epochs :
                path = '{0}/{1}_{2}_{3}.pth'.format(save_path,
                                                    args.backbone,
                                                    args.dataset,
                                                    epoch)
                torch.save(model.state_dict(), path)    
        draw_curve(save_path, train_logger)#, val_logger)

        # 모델저장
    
    else: ## fine-tuning
        print('Fine-tuning을 시작합니다.')
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
    print("Process Complete : it took {time:.2f} minutes".format(time=(time.time()-start)/60))

if __name__ == '__main__':
    main()