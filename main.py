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
from dataset import MTA, GTZAN, JsonAudio
from loss import SiamusicLoss
from augmentation import sungrae_pedal, random_mix
from simsiam import Siamusic, Evaluator

parser = argparse.ArgumentParser(description='Siamusic')
parser.add_argument('--save_path', default='./exp', type=str,
                    help='save path')
parser.add_argument('--backbone', default='ResNet34', type=str,
                    help='backbone network for simsiam',
                    choices=['resnet50','resnet101','resnet152','transformer'])
parser.add_argument('--dim', default=2048, type=int,
                    help='output dimension')
parser.add_argument('--nhead', default=4, type=int,
                    help='the number of transformer heads',choices=[1,2,4])
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
parser.add_argument('--patchs', default=12, type=int,
                     help='ramdom mix augmentation patchs')
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
        audio = audio.float()
        if args.augmentation == 'pedalboard':
            x1, x2 = sungrae_pedal(audio), sungrae_pedal(audio)
        elif args.augmentation == 'randommix':
            x1, x2 = random_mix(audio,args.patchs), random_mix(audio,args.patchs)
        elif args.augmentation == 'image':
            sys.exit('곧 업데이트 예정')
        x1, x2 = x1.cuda(), x2.cuda()
        p1, z2, p2, z1 = model(x1,x2) # backbone(+projection) + predictor
        loss = criterion(p1,z2,p2,z1)
        # loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        train_loss.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 and i != 0:
            print('Epoch : [{0}/{1}] [{2}/{3}]  Train Loss : {loss:.4f}'.format(
                epoch, num_epoch, i, len(trn_loader), loss=loss))
    train_logger.write([epoch, train_loss.avg])



# downstream task train
def train(model, trn_loader, criterion, optimizer, epoch, num_epoch, train_logger):
    model.train()
    train_loss = AverageMeter()
    for i, audio,target in enumerate(trn_loader):
        audio, target = audio.float().cuda(), target.float().cuda()
        y_pred = model(audio)
        loss = criterion(target,y_pred)
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
        for i, audio, target in enumerate(val_loader):
            audio, target = audio.cuda(), target.cuda()
            y_pred = model(audio)
            loss = criterion(target,y_pred)
            val_loss.update(loss.item())

        print("=================== Validation Start ====================")
        print('Epoch : [{0}/{1}]  Test Loss : {loss:.4f}'.format(
                epoch, num_epochs, loss=val_loss.avg))
        print("=================== Validation End ======================")
        val_logger.write([epoch, val_loss.avg])
    return val_loss.avg


def test(model, test_loader, dataset):
    # MTA : ACC@5, ACC@10, RECALL@5, RECALL@10
    # GTZAN : F1, AUROC, AUPRC, ACC@1, ACC@5
    model.eval()
    if dataset == 'MTA':
        pass
    elif dataset == 'GTZAN':
        pass
    
    with torch.no_grad():
        for i,audio,target in enumerate(test_loader):
            pass
        print("=================== Test Start ====================")

        print("=================== Test End ====================")


def main():
    # define environment
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    model = Siamusic(backbone=args.backbone,
                     dim=args.dim,
                     nhead=args.nhead).cuda()

    
    # pre-training or fine-tuning
    if args.from_scratch: ## pre-training
        print('스크래치부터 학습됩니다.')

        # save path
        save_path=args.save_path+'_'+args.backbone+'_'+args.augmentation+'_'+args.optim
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Save configuration
        with open(save_path + '/configuration.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # dataset loading
        train_dataset = JsonAudio('D:/SiamRec/data/json_audio',args.input_length)
        train_loader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=2,shuffle=True,drop_last=True)
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

        # 학습시작
        for epoch in tqdm(range(1,args.epochs+1)):
            siam_train(model, train_loader, criterion ,optimizer, epoch, args.epochs, train_logger)
            scheduler.step()
            if epoch%20 == 0 or epoch == args.epochs :
                path = '{0}/{1}_{2}_{3}.pth'.format(save_path,
                                                    args.backbone,
                                                    args.augmentation,
                                                    epoch)
                torch.save(model.state_dict(), path)    
        draw_curve(save_path, train_logger, train_logger)

        # 모델저장
    
    else: ## fine-tuning
        print('Fine-tuning을 시작합니다.')
        # save path
        save_path=args.save_path+'_'+args.dataset+'_'+args.backbone+'_'+args.augmentation+'_'+args.optim
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Save configuration
        with open(save_path + '/configuration.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # dataset loading
        if args.dataset == 'MTA':
            train_dataset = MTA(split='train',input_length=args.input_length)
            val_dataset = MTA(split='validation',input_length=args.input_length)
            test_dataset = MTA(split='test',input_length=args.input_length)
            num_classes = 50
        elif args.dataset == 'GTZAN':
            train_dataset = GTZAN(split='train',input_length=args.input_length)
            val_dataset = GTZAN(split='validation',input_length=args.input_length)
            test_dataset = GTZAN(split='test',input_length=args.input_length)
            num_classes = 10
        train_loader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=2,shuffle=True)
        val_loader = DataLoader(val_dataset,batch_size=args.batch_size,num_workers=2,shuffle=False)
        test_loader = DataLoader(test_dataset,batch_size=args.batch_size,num_workers=2,shuffle=False)
        print('=== DataLoader R.e.a.d.y ===')

        # 모델 불러오기 & pretrain모델 주입
        PATH = './exp_' + args.backbone + '_' + args.augmentation + '_' + args.optim
        pth_file = args.backbone+'_'+args.aumentation+'_300.pth'
        model.load_state_dict(torch.load(PATH+'/'+pth_file))
        evaluation_model = Evaluator(model.encoder,num_classes,args.backbone,args.dim) # 내부에서 encoder freeze잊지말기
        
        # define criterion
        criterion = nn.CrossEntropyLoss().cuda()
        if args.optim == 'sgd':
            optimizer = optim.SGD(evaluation_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == 'adam':
            optimizer = optim.Adam(evaluation_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == 'adagrad':
            optimizer = optim.Adagrad(evaluation_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        milestones = [int(args.epochs/3),int(args.epochs/2)]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.7)

        # logger
        train_logger = Logger(os.path.join(save_path, 'train_loss.log'))
        val_logger = Logger(os.path.join(save_path, 'val_loss.log'))
        test_logger = Logger(os.path.join(save_path, 'test_loss.log'))

        # 학습시작
        best_loss = 10000
        worse = 0
        model_dict = None
        for epoch in tqdm(range(1,args.epochs+1)):
            train(evaluation_model, train_loader, criterion ,optimizer, epoch, args.epochs, train_logger)
            val_loss = validation(evaluation_model, val_loader, criterion, epoch, args.epochs, val_logger)
            scheduler.step()
            if val_loss < best_loss:
                best_loss = val_loss
                worse = 0
                model_dict = evaluation_model.state_dict()
            else:
                worse += 1
    
            if worse > 4 :
                print('== Early Stopping ==')
                path = '{0}/{1}_{2}_{3}.pth'.format(save_path,
                                                    args.backbone,
                                                    args.augmentation,
                                                    epoch)
                torch.save(model_dict, path)
                break    
        draw_curve(save_path, train_logger, val_logger)

        # 테스트시작
        test(evaluation_model, test_loader, args.dataset)

    print("Process Complete : it took {time:.2f} minutes".format(time=(time.time()-start)/60))

if __name__ == '__main__':
    main()