import numpy as np
import IPython.display as ipd
from scipy.io.wavfile import read
from torch.utils import data
from playsound import playsound
import pygame # 이유는 모르겠는데 얘도 같이 import해야 playsound가 작동함

def listen(data_path):
    if 'MTA' in data_path:
        waveform = np.load(data_path.replace(".mp3",".npy"))
        ipd.Audio(waveform, rate=16000)
        pygame.mixer.init()
        pygame.mixer.music.load(data_path)
        pygame.mixer.music.play()
    elif 'GTZAN' in data_path:
        playsound(data_path)

   


if __name__ == '__main__':
    # MTA
    data_path = './dataset/MTA/waveform/b/altri_stromenti-uccellini-01-confitebor_monteverdi-0-29.npy'
    listen(data_path)

    # GTZAN
    # Master_Of_Puppets = './dataset/GTZAN/genres_original/metal/metal.00033.wav'
    # The_Trooper = './dataset/GTZAN/genres_original/metal/metal.00034.wav'
    # listen(Master_Of_Puppets)
    # listen(The_Trooper)