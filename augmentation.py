'''
input : torch.tensor() (16,1,48000)
output : augmented audio tensor
'''
import pedalboard
import random
import time
from utils import listen
from dataset import MTA, GTZAN
import torch
from torch.utils.data import DataLoader
import numpy as np
from pedalboard import (Pedalboard, 
                        Convolution, 
                        Compressor, 
                        Chorus, 
                        Gain, 
                        Distortion,
                        Reverb, 
                        LadderFilter, 
                        Phaser)
# official git : https://github.com/spotify/pedalboard
# for more inforamtion and demo audio :
# https://colab.research.google.com/drive/1bHjhJj1aCoOlXKl_lOfG99Xs3qWVrhch#scrollTo=iusi0_xLyz97

'''
== 패달보드 설명서 ==
$ : 영향큼
1. Compressor
    - $threshold_db : 시끄러운정도 또는 소리의크기(db)가 얼마 이상되는 것을 기준으로 삼을 것인가? 0 = 오디오의 최대 소리크기
    - $ratio : threshold_db를 넘은 음역을 얼마나 깍을(조용하게 만들) 것인가?
    - attack_ms : This controls what the compressor does to your input signal. If you want to hear the hard pluck of your pick strokes, turn the Attack knob up.
    - release_ms : This controls the release times of your notes. 
                   Not all compressors control sustain, as their true function is actually to suppress loud notes so that quiet notes sound comparatively louder.

2. Gain
    - $gain_db : 얼마나 소리를 증폭할 것인가? 증폭하되 찌그러뜨리며 증폭하는 특징이 있다 -20~30

3. Chorus : 소리를 흔들어 퍼지게하여 풍성해보이게 해주는 효과
    - $rate_hz : 흔들림의 스피드 1~50
    - $depth : 흔들림의 깊이감 0~7
    - centre_delay_ms : 모르겠음 패스
    - $feedback : 건들지 말기
    - mix : 출력값으로 전달할 원본 신호의 볼륨을 나타내며 0.0 에서 1.0 사이의 값 -> 바꿔도 별일 없다

4. LadderFilter : 소리를 걸러냄
    - cutoff_hz : 이거 이상 다 자르는듯
    - $drive : drive걸어줌 1~5

5. Phaser : 위상차를 이용한 이펙터로 제트기가 지나가듯이 웅-하는 소리를 내게 함
    - $rate_hz : 흔들림의 스피드 -5~5
    - $depth : 흔들림의 깊이 -100~100
    - $feedback : 건들지 말기
    - mix

6. Distortion
    - $drive_db : -5~20

7. Reverb : 공연장, 홀에 있는 듯한 공간감을 만들어냄
    리버브는 파라미터들 말로 설명하기 힘들다 9ㅅ9
    - $room_size : 더 큰 공간감 0~1
    - $damping : 0~1 
    - $wet_level : 풍성함(방이 커진다기보다 여러명이 불러서 풍성해지는 느낌?) 0~1
    - $dry_level : 0~1
    - $width : 0~1
    - $freeze_mode : 0~1
'''

def sungrae_pedal(audio,sample_rate=44100):
    # audio.shape = (B, 1, 48000)
    aug_audio = audio.clone()
    sample_rate = sample_rate
    for idx in range(audio.shape[0]):
        pedal_board = Pedalboard([Compressor(threshold_db=0, ratio=1, attack_ms=1, release_ms = 100),
                                  # threshold_db: float = 0, ratio: float = 1, attack_ms: float = 1.0, release_ms: float = 100
                                  Gain(gain_db=random.randrange(-15,30)),
                                  # gain_db: float = 1.0
                                  Chorus(rate_hz=random.randrange(1,7), 
                                         depth = random.randrange(0,4), 
                                         centre_delay_ms= 4.0, feedback = 0.0, mix = 0.5), 
                                  # rate_hz: float = 1.0, depth: float = 0.25, centre_delay_ms: float = 7.0, feedback: float = 0.0, mix: float = 0.5
                                  LadderFilter(mode=LadderFilter.Mode.HPF12, cutoff_hz=200, resonance=0, drive=random.randrange(1,6)), 
                                  # cutoff_hz: float = 200, resonance: float = 0, drive: float = 1.0
                                  Phaser(rate_hz=random.randrange(-5,6), 
                                         depth=random.randrange(-10,10), feedback=0.25, mix=1.0), 
                                  # rate_hz: float = 1.0, depth: float = 0.5, centre_frequency_hz: float = 1300.0, feedback: float = 0.0, mix: float = 0.5
                                  Distortion(drive_db=random.randrange(0,20)),
                                  # drive_db: float = 25
                                  Reverb(room_size=random.random(),
                                         damping=random.random(),
                                         wet_level=random.random(), 
                                         dry_level=random.random(),
                                         width=random.random(),
                                         freeze_mode=random.random()) 
                                  #room_size: float = 0.5, damping: float = 0.5, wet_level: float = 0.33, dry_level: float = 0.4, width: float = 1.0, freeze_mode: float = 0.0
                                  ], 
                                  sample_rate=sample_rate)
        pedal_out = pedal_board(audio[idx,0])
        aug_audio[idx] = torch.unsqueeze(torch.tensor(pedal_out),0)
        
    return aug_audio # (B, 1, 48000)




def random_mix(audio,n_patchs=None):
    assert n_patchs != None
    if audio.shape[-1]%n_patchs != 0:
        raise ValueError('오디오 시퀀스가 patch수 만큼 정수로 나눠지지 않습니다')
    aug_audio = audio.clone()
    divide = int(audio.shape[-1]/n_patchs)
    for idx in range(audio.shape[0]):
        random_idx = [i for i in range(n_patchs)]
        random.shuffle(random_idx)
        temp = audio[idx].clone()
        for i,molayo in enumerate(random_idx):
            temp[0][i*divide:(i+1)*divide] = audio[idx][0][molayo*divide:(molayo+1)*divide]
        aug_audio[idx] = temp
    return aug_audio



if __name__ == '__main__':
    input_length = 48000
    # MTA
    print('== MTA ==')
    mta_data = MTA('test',input_length=input_length)
    mta_dataloader = DataLoader(mta_data,batch_size=1,drop_last=True,shuffle=True)
    mta_x, mta_y = next(iter(mta_dataloader))
    print('Origianl Sounds')
    listen(mta_x[0,0])
    aug = sungrae_pedal(mta_x)
    time.sleep(1)
    print('Now on pedal sounds')
    listen(aug[0,0])

    # GTZAN
    print('== GTZAN ==')
    gtzan_data = GTZAN('validation',input_length=input_length)
    gtzan_dataloader = DataLoader(gtzan_data,batch_size=1,drop_last=True,shuffle=True)
    gtzan_x, gtzan_y = next(iter(gtzan_dataloader))
    print('Origianl Sounds')
    listen(gtzan_x[0,0])
    aug = sungrae_pedal(gtzan_x)
    time.sleep(1)
    print('Now on pedal sounds')
    listen(aug[0,0])