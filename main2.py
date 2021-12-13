# library
from numpy.random import choice
from torch.nn.modules.activation import Threshold
from torch.utils.data import DataLoader
import os
import torch
import argparse
import torch.optim as optim
import json
import torch.nn as nn
from torch.optim import lr_scheduler
from torchmetrics.classification import accuracy
from tqdm import tqdm
from torchmetrics import F1,AUROC,Accuracy,Recall
import time
import sys

# local
from utils import *
from dataset import MTA, GTZAN, JsonAudio
from loss import SiamusicLoss
from augmentation import image_augmentation, sungrae_pedal, random_mix
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
parser.add_argument('--fma', default='small', type=str,
                    help='어떤 데이터셋으로 pre-train 할건가?',
                    choices=['medium','small'])
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
parser.add_argument('--threshold', default=0.5, type=float,
                    help='MTA 에서 confidence가 얼마 이상이면 1로 예측했다고 할 것인가?')
									
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
            x1, x2 = image_augmentation(audio), image_augmentation(audio) 
            # 이부분에서 모델 수정이 필요한지? 수정해도 문제 없는지(내 생각에는 없다)
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
    for i, (audio,target) in enumerate(trn_loader):
        audio, target = audio.float().cuda(), target.cuda()
        y_pred = model(audio)
        loss = criterion(y_pred,target)
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
    print("=================== Validation Start ====================")
    model.eval()
    val_loss = AverageMeter()
    with torch.no_grad():
        for i, (audio, target) in enumerate(val_loader):
            audio, target = audio.float().cuda(), target.cuda()
            y_pred = model(audio)
            loss = criterion(y_pred,target)
            val_loss.update(loss.item())

        print('Epoch : [{0}/{1}]  Validation Loss : {loss:.4f}'.format(
                epoch, num_epochs, loss=val_loss.avg))
        print("=================== Validation End ======================")
        val_logger.write([epoch, val_loss.avg])
    return val_loss.avg


def test(model, test_loader, dataset, save_path):
    # MTA : ACC@1, Recall@1, Recall@5
    # GTZAN : ACC@1, ACC@5, Recall@1, Recall@5, F1@1, F1@5
    print("=================== Test Start ====================")
    model.eval()

    if dataset == 'MTA':
        threshold = args.threshold
        accuracy = Accuracy(num_classes=50,threshold=threshold) # TN이 많아서 ACC자체는 높게 나온다
        aves = [AverageMeter(), # acc@1
                AverageMeter(), # recall@1
                AverageMeter()] # recall@5
        with torch.no_grad():
            for i,(audio,target) in enumerate(test_loader):
                audio, target = audio.float().cuda(), target
                y_pred = model(audio)
                y_pred = y_pred.detach().cpu()
                aves[0].update(accuracy(y_pred,target.int()))
                aves[1].update(recall_at_k(y_pred,target,1))
                aves[2].update(recall_at_k(y_pred,target,5))

            print(f'ACC@1 : {aves[0].avg:.2f}±{aves[0].std:.2f}')
            print(f'Recall@1 : {aves[1].avg:.2f}±{aves[1].std:.2f}')
            print(f'Recall@5 : {aves[2].avg:.2f}±{aves[2].std:.2f}')
            result = {'ACC@1' : f'{aves[0].avg:.2f}±{aves[0].std:.2f}',
                      'Recall@1' : f'{aves[1].avg:.2f}±{aves[1].std:.2f}',
                      'Recall@5' : f'{aves[2].avg:.2f}±{aves[2].std:.2f}'
                      }

                    
                
    elif dataset == 'GTZAN':
        # https://nittaku.tistory.com/295
        accuracy_at_1 = Accuracy(10)
        accuracy_at_5 = Accuracy(10,top_k=5)
        recall_at_1 = Recall(10)
        recall_at_5 = Recall(10,top_k=5)
        f1_at_1 = F1(num_classes=10)
        f1_at_5 = F1(num_classes=10,top_k=5)
        aves = [AverageMeter(), # acc@1
                AverageMeter(), # acc@5
                AverageMeter(), # recall@1
                AverageMeter(), # recall@5
                AverageMeter(), # f1@1
                AverageMeter()] # f1@5
        with torch.no_grad():
            for i,(audio,target) in enumerate(test_loader):
                audio, target = audio.float().cuda(), target
                y_pred = model(audio)
                y_pred = y_pred.detach().cpu()
                aves[0].update(accuracy_at_1(y_pred,target))
                aves[1].update(accuracy_at_5(y_pred,target))
                aves[2].update(recall_at_1(y_pred,target))
                aves[3].update(recall_at_5(y_pred,target))
                aves[4].update(f1_at_1(y_pred,target))
                aves[5].update(f1_at_5(y_pred,target))
                
            print(f'ACC@1 : {aves[0].avg:.2f}±{aves[0].std:.2f}')
            print(f'ACC@5 : {aves[1].avg:.2f}±{aves[1].std:.2f}')
            print(f'Recall@1 : {aves[2].avg:.2f}±{aves[2].std:.2f}')
            print(f'Recall@5 : {aves[3].avg:.2f}±{aves[3].std:.2f}')
            print(f'F1@1 : {aves[4].avg:.2f}±{aves[4].std:.2f}')
            print(f'F1@5 : {aves[5].avg:.2f}±{aves[5].std:.2f}')
            result = {'ACC@1' : f'{aves[0].avg:.2f}±{aves[0].std:.2f}',
                      'ACC@5' : f'{aves[1].avg:.2f}±{aves[1].std:.2f}',
                      'Recall@1' : f'{aves[2].avg:.2f}±{aves[2].std:.2f}',
                      'Recall@5' : f'{aves[3].avg:.2f}±{aves[3].std:.2f}',
                      'F1@1' : f'{aves[4].avg:.2f}±{aves[4].std:.2f}',
                      'F1@5' : f'{aves[5].avg:.2f}±{aves[5].std:.2f}'
                      }


    # Save configuration
    with open(save_path + '/result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("=================== Test End ====================")


def main():
    # define environment
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    model = Siamusic(backbone=args.backbone,
                     dim=args.dim,
                     nhead=args.nhead).cuda()
    model = nn.DataParallel(model)
    
    # pre-training or fine-tuning
    if args.from_scratch: ## pre-training
        if args.MTA: # MTA로 pre-train
            print('스크래치부터 학습됩니다 : MTA')

            # save path
            save_path=args.save_path+'_'+args.backbone+'_'+args.augmentation+'_'+args.optim+'_'+'MTA'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Save configuration
            with open(save_path + '/configuration.json', 'w') as f:
                json.dump(args.__dict__, f, indent=2)

            # dataset loading
            train_dataset = MTA('train')
            fine_dataset = MTA('fine')
            test_dataset = MTA('test')
            train_loader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=2,shuffle=True)
            fine_loader = DataLoader(fine_dataset,batch_size=args.batch_size,num_workers=2,shuffle=True)
            test_loader = DataLoader(test_dataset,batch_size=args.batch_size,num_workers=2,shuffle=False)
            print(f'=== DataLoader R.e.a.d.y | Length : {len(train_dataset)} | {len(fine_dataset)} | {len(test_dataset)} ===')

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
                # 모델저장
                if epoch%20 == 0 or epoch == args.epochs :
                    path = '{0}/{1}_{2}_{3}.pth'.format(save_path,
                                                        args.backbone,
                                                        args.augmentation,
                                                        epoch)
                    torch.save(model.state_dict(), path)    
            draw_curve(save_path, train_logger, train_logger)

            
        elif args.GTZAN: # GTZAN으로 pre-train
            sys.exit('NonImplementError')

        else: # FMA로 pre-train
            print(f'스크래치부터 학습됩니다 : FMA_{args.fma}')

            # save path
            save_path=args.save_path+'_'+args.backbone+'_'+args.augmentation+'_'+args.optim+'_'+args.fma
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            # Save configuration
            with open(save_path + '/configuration.json', 'w') as f:
                json.dump(args.__dict__, f, indent=2)

            # dataset loading
            data_path = './dataset/fma_'+args.fma+'_json'
            train_dataset = JsonAudio(data_path,args.input_length)
            train_loader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=2,shuffle=True,drop_last=True)
            print(f'=== DataLoader R.e.a.d.y | Length : {len(train_dataset)} ===')

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
                # 모델저장
                if epoch%20 == 0 or epoch == args.epochs :
                    path = '{0}/{1}_{2}_{3}.pth'.format(save_path,
                                                        args.backbone,
                                                        args.augmentation,
                                                        epoch)
                    torch.save(model.state_dict(), path)    
            draw_curve(save_path, train_logger, train_logger)
    
    else: ## fine-tuning
        print('Fine-tuning을 시작합니다.')
        # save path
        save_path=args.save_path+'_MTA_'+args.backbone+'_'+args.augmentation+'_'+args.optim+'_MTA'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Save configuration
        with open(save_path + '/configuration.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # dataset loading
        if args.dataset == 'MTA':
            train_dataset = MTA(split='fine',input_length=args.input_length)
            val_dataset = MTA(split='test',input_length=args.input_length)
            test_dataset = MTA(split='test',input_length=args.input_length)
            num_classes = 50
        elif args.dataset == 'GTZAN':
            train_dataset = GTZAN(split='fine',input_length=args.input_length)
            val_dataset = GTZAN(split='test',input_length=args.input_length)
            test_dataset = GTZAN(split='test',input_length=args.input_length)
            num_classes = 10
        train_loader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=2,shuffle=True)
        val_loader = DataLoader(val_dataset,batch_size=args.batch_size,num_workers=2,shuffle=False)
        test_loader = DataLoader(test_dataset,batch_size=args.batch_size,num_workers=2,shuffle=False)
        print('=== DataLoader R.e.a.d.y ===')

        # 모델 불러오기 & pretrain모델 주입
        '''
        https://justkode.kr/deep-learning/pytorch-save
        https://tutorials.pytorch.kr/beginner/saving_loading_models.html
        '''
        PATH = './exp_' + args.backbone + '_' + args.augmentation + '_' + args.optim + '_MTA'
        pth_file = args.backbone+'_'+args.augmentation+'_100.pth'
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            model = Siamusic(backbone=args.backbone,
                             dim=args.dim,
                             nhead=args.nhead).cuda()
            model = nn.DataParallel(model) # 뒤엔 지우자..
            model.load_state_dict(torch.load(PATH+'/'+pth_file))
        except:
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'
            model = Siamusic(backbone=args.backbone,
                             dim=args.dim,
                             nhead=args.nhead).cuda()
            model = nn.DataParallel(model) # 뒤엔 지우자..
            model.load_state_dict(torch.load(PATH+'/'+pth_file))
        for para in model.parameters(): # endoer freeze
            para.requires_grad = False
        evaluation_model = Evaluator(model.module.encoder,num_classes,args.backbone,args.dim).cuda()
        # evaluation_model = Evaluator(model.encoder,num_classes,args.backbone,args.dim).cuda()
        
        # define criterion
        if args.dataset == 'MTA': # 50 multi-classes
            criterion = nn.BCEWithLogitsLoss().cuda()
        elif args.dataset == 'GTZAN': # 10 single classes
            criterion = nn.CrossEntropyLoss().cuda()

        # define optimizer
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

        # 학습시작 (아래는 early sttoping이나 test 성능이 나빠서 일단 주석처리)
        best_loss = 10000
        worse = 0
        model_dict = None
        for epoch in tqdm(range(1,args.epochs+1)):
            train(evaluation_model, train_loader, criterion ,optimizer, epoch, args.epochs, train_logger)
            val_loss = validation(evaluation_model, val_loader, criterion, epoch, args.epochs, val_logger)
            scheduler.step()
            # if val_loss < best_loss:
            #     best_loss = val_loss
            #     worse = 0
            #     model_dict = evaluation_model.state_dict()
            # else:
            #     worse += 1
    
            # if worse > 4 :
            #     print('== Early Stopping ==')
            #     path = '{0}/{1}_{2}_{3}.pth'.format(save_path,
            #                                         args.backbone,
            #                                         args.augmentation,
            #                                         epoch)
            #     torch.save(model_dict, path)
            #     break
            if epoch == args.epochs:
                path = '{0}/{1}_{2}_{3}.pth'.format(save_path,
                                                    args.backbone,
                                                    args.augmentation,
                                                    epoch)
                torch.save(evaluation_model.state_dict(), path)
        draw_curve(save_path, train_logger, val_logger)

        # 테스트시작
        test(evaluation_model, test_loader, args.dataset, save_path)

    print("Process Complete : it took {time:.2f} minutes".format(time=(time.time()-start)/60))

if __name__ == '__main__':
    main()