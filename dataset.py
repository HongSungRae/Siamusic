import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import librosa
import json
# from scipy.io.wavfile import read



class MTA(Dataset):
    def __init__(self,split=None,input_length=48000):
        self.split = split
        self.input_length = input_length
        if split not in ['train', 'test', 'validation', 'fine']:
            raise ValueError("Please tell the data split : train, test, validation or fine")
        
        self.data_path = './dataset/MTA/waveform'
        self.df = pd.read_csv('./dataset/MTA/annotations_final.csv', sep="\t", index_col=0)
        self.TAGS = ['guitar','classical', 'slow','techno','strings','drums','electronic','rock',
                     'fast','piano','ambient','beat','violin','vocal','synth','female','indian',
                     'opera','male','singing','vocals','no vocals','harpsichord','loud','quiet',
                     'flute', 'woman', 'male vocal', 'no vocal', 'pop','soft','sitar', 'solo',
                     'man', 'classic', 'choir', 'voice', 'new age', 'dance', 'male voice',
                     'female vocal', 'beats', 'harp', 'cello', 'no voice', 'weird', 'country',
                     'metal', 'female voice', 'choral']
        # Filter out, un-annnotated dataset
        df_filter =  self.df[self.TAGS].sum(axis=1)
        use_id = df_filter[df_filter != 0].index
        self.df = self.df.loc[use_id]

        # Data Split & Preprocessing
        split_list = []
        self.id_to_path = {}
        for idx in range(len(self.df)):
            item = self.df.iloc[idx]
            id = item.name
            path = item['mp3_path']
            folder = path.split("/")[0]
            self.id_to_path[id] = path
            if split=='train' and (folder in "012ab"):
                split_list.append(id)
            elif split=='validation' and (folder == "c"):    
                split_list.append(id)
            elif split=='fine' and (folder == "c"):    
                split_list.append(id)
            elif split=='test' and (folder in "d"):
                split_list.append(id)
        self.df = self.df[self.TAGS]
        self.df = self.df.loc[split_list]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        waveform = self.item_to_waveform(item)
        return waveform.astype(np.float32), item.values.astype(np.float32)

    def item_to_waveform(self, item):
        id = item.name
        path = os.path.join(self.data_path, self.id_to_path[id].replace(".mp3", ".npy"))
        waveform = np.load(path) # shape : [128000]
        
        # ?????? ?????????(waveform)?????? input_length ????????? ?????? ?????????
        random_idx = np.random.randint(low=0, high=int(waveform.shape[0] - self.input_length))
        waveform = waveform[random_idx:random_idx+self.input_length] # extract 48000 sequence
        audio = np.expand_dims(waveform, axis = 0) # expand to [1,48000]
        return audio # [1,48000]




class GTZAN(Dataset):
    '''
    This dataset has 10 classes which has 100 songs each.
    All songs are 30 seconds length.
    '''
    def __init__(self,split=None,input_length=48000):
        self.data_path =  './dataset/GTZAN/genres_original'
        self.split = split
        self.input_length = input_length
        self.genres = ['blues','classical','country','disco','hiphop',
                      'jazz','metal','pop','reggae','rock']
        if split not in ['train', 'test', 'validation']:
            raise ValueError("Please tell the data split : train, test,validation")
        
        # Data Split & Preprocessing
        self.df = pd.DataFrame(data={'Path':[], 'Name':[], 'Label':[]})
        self.Path_list = []
        self.Name_list = []
        self.Label_list = []
        if self.split == 'train':
            for genre in self.genres:
                file_list = os.listdir(self.data_path + '/' + genre)
                self.Name_list += file_list[0:80]
                self.Path_list += [self.data_path + '/' + genre for _ in range(80)]
                self.Label_list += [genre for _ in range(80)]
        elif self.split == 'test':
            for genre in self.genres:
                file_list = os.listdir(self.data_path + '/' + genre)
                self.Name_list += file_list[80:90]
                self.Path_list += [self.data_path + '/' + genre for _ in range(10)]
                self.Label_list += [genre for _ in range(10)]
        else:
            for genre in self.genres:
                file_list = os.listdir(self.data_path + '/' + genre)
                self.Name_list += file_list[90:]
                self.Path_list += [self.data_path + '/' + genre for _ in range(10)]
                self.Label_list += [genre for _ in range(10)]
        self.df = pd.DataFrame(data={'Path':self.Path_list, 
                                     'Name':self.Name_list, 
                                     'Label':self.Label_list})

    def __len__(self):
        return len(self.df)

    def get_waveform(self,data_path):
        waveform,_ = librosa.load(data_path,sr=22050)
        waveform = np.array(waveform,dtype=float)
        #waveform = read(data_path)
        #waveform = np.array(waveform[1],dtype=float) # shape:[661794]
        random_idx = np.random.randint(low=0, high=int(waveform.shape[0] - self.input_length))
        waveform = waveform[random_idx:random_idx+self.input_length] # extract 48000 sequence
        audio = np.expand_dims(waveform, axis = 0) # expand to [1,48000]
        return audio
        
    def __getitem__(self, index):
        genre = self.df['Label'][index]
        # label = torch.zeros(10) # one-hot
        # label[self.genres.index(genre)] = 1 # one-hot
        label = torch.tensor(self.genres.index(genre))
        data_path = self.df['Path'][index] + '/' + self.df['Name'][index]
        audio = self.get_waveform(data_path)
        return audio, label



class MPAudio(Dataset):
    '''
    wav audio??? ????????? ????????? train/val???????????? ???????????????
    mp3??? ?????? ?????? ????????? ?????? python library??? ?????? ????????????
    '''
    def __init__(self,split=None,input_length=48000,type='wav'):
        if split not in ['train','validation']:
            raise ValueError()
        self.split = split
        self.input_length = input_length
        self.dir = 'D:/SiamRec/data/' + type + '_audio'
        self.audios = os.listdir(self.dir)
        if split == 'train':
            self.data_list = self.audios[0:int(len(self.audios)*0.85)]
        else:
            self.data_list = self.audios[int(len(self.audios)*0.85):]

    def __len__(self):
        return len(self.data_list)

    def get_waveform(self,data_path):#22050
        waveform,_ = librosa.load(data_path,sr=22050,duration=60)
        waveform = np.array(waveform,dtype=float)
        try:
            random_idx = np.random.randint(low=0, high=int(waveform.shape[0] - self.input_length))
        except:
            random_idx = 0
        waveform = waveform[random_idx:random_idx+self.input_length] # extract 48000 sequence
        audio = np.expand_dims(waveform, axis = 0) # expand to [1,48000]
        return audio

    def __getitem__(self, idx):
        data_path = self.dir + '/' + self.data_list[idx]
        waveform = self.get_waveform(data_path)
        return waveform.astype(np.float32)




class JsonAudio(Dataset):
    def __init__(self,data_dir,input_length):
        self.data_dir = data_dir
        self.data_list = os.listdir(data_dir)
        self.input_length = input_length

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        with open(self.data_dir+'/'+self.data_list[idx], 'r') as f:
            waveform = np.array(json.load(f)['audio'],dtype=float)
        try:
            random_idx = np.random.randint(low=0, high=int(waveform.shape[-1] - self.input_length))
            waveform = waveform[0][random_idx:random_idx+self.input_length]
            audio = np.expand_dims(waveform, axis = 0) # expand to [1,48000]
        except:
            temp = np.zeros((48000))
            temp[0:int(waveform.shape[-1])] = waveform
            audio = np.expand_dims(temp, axis = 0) # expand to [1,48000]
        return audio
        




if __name__ == '__main__':
    # MTA
    mta_data = MTA('test')
    mta_dataloader = DataLoader(mta_data,batch_size=16,drop_last=True)
    mta_x, mta_y = next(iter(mta_dataloader))
    print(f'mta_x : {mta_x.shape} | mta_y : {mta_y.shape}')

    # # GTZAN
    gtzan_data = GTZAN('validation')
    gtzan_dataloader = DataLoader(gtzan_data,batch_size=16,drop_last=True)
    gtzan_x, gtzan_y = next(iter(gtzan_dataloader))
    print(f'gtzan_x : {gtzan_x.shape} | gtzan_y : {gtzan_y.shape}')

    # # MPAudio
    # mp3_data = MPAudio('validation',48000,'mp3')
    # mp3_dataloader = DataLoader(mp3_data,batch_size=4,drop_last=True)
    # mp3_x = next(iter(mp3_dataloader))
    # print(f'mp3_x : {mp3_x.shape}')

    # JsonAudio
    # json_data = JsonAudio('D:/SiamRec/data/json_audio',48000)
    # json_dataloader = DataLoader(json_data,batch_size=16,drop_last=True)
    # json_x = next(iter(json_dataloader))
    # print(f'json_x : {json_x.shape}')