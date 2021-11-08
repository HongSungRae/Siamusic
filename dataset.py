import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os



class MTA(Dataset):
    def __init__(self,split=None,input_length=48000):
        self.split = split
        self.input_length = input_length
        if split not in ['train', 'test', 'validation']:
            raise ValueError("Please tell the data split : train, test,validation")
        
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
        path = os.path.join(self.data_path, self.id_to_path[id].replace(".mp3", ".npy")) # pre-extract waveform, for fast loader # npy->mp3
        waveform = np.load(path) # 128000의 array형태
        if self.split in ['train','validation']:
        # 전체 시퀀스(waveform)에서 input_length 만큼만 떼서 사용함
            random_idx = np.random.randint(low=0, high=int(waveform.shape[0] - self.input_length))
            waveform = waveform[random_idx:random_idx+self.input_length] # extract input # [48000]
            audio = np.expand_dims(waveform, axis = 0)# 1 x samples [1,48000]
        elif self.split == 'test':
            # 전체 시퀀스(waveform)에서 첫 48000, 두번째 48000시퀀스 떼어온다
            chunk_number = waveform.shape[0] // self.input_length # 128000/48000 = 2 라고 나옴
            chunk = np.zeros((chunk_number, self.input_length)) # [2,48000]
            for idx in range(chunk.shape[0]): # == chunk_number = 2
                chunk[idx] = waveform[idx * self.input_length:(idx+1) * self.input_length]
            audio = chunk
        return audio # Train에서 [1,48000], Test에서 [2,48000]


class GTZAN(Dataset):
    '''
    This dataset has 10 classes which has 100 songs each.
    All songs are 30 seconds length.
    '''
    def __init__(self,split=None):
        self.data_path =  './dataset/GTZAN/genres_original'
        self.split = split
        self.genre = ['blues','classical','country','disco','hiphop',
                      'jazz','metal','pop','reggae','rock']
        if split not in ['train', 'test', 'validation']:
            raise ValueError("Please tell the data split : train, test,validation")

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass



if __name__ == '__main__':
    # MTA
    mta_data = MTA('test')
    mta_dataloader = DataLoader(mta_data,batch_size=16,drop_last=True)
    mta_x, mta_y = next(iter(mta_dataloader))
    print(f'mta_x : {mta_x.shape} | mta_y : {mta_y.shape}')

    # GTZAN
    gtzan_data = GTZAN('train')
    #gtzan_dataloader = DataLoader(gtzan_data,batch_size=16,drop_last=True)