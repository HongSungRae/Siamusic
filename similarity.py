'''
This module verifies that the trained model classifies
the presentation of POP music well.

== The Artist List ==
['Beenzino', 'Oasis', '2NE1', 'IU', 'Green Day'
 'NELL', 'LOCO', 'Dinimic Duo', 'Red Velvet', 'Pink Floyd']
'''
import torch
import torch.nn as nn

### 1
# 50곡 넣어서 50x50 similarity뽑기
# 다양한 measure 쓸 수 있게 하기
def get_similarity(model,model_path,distance_measure=None,input_length=48000):
    if distance_measure == None:
        return ValueError('choice : ["cosine", "euclidean"]')
    if model == 'ResNet50':
        model = ResNet50()
    elif model == 'ResNet101':
        model == ResNet101()
    elif model == 'Transformer':
        model = Transformer()
    ## state dict

    ## pop folder에서 음악 50개 for문




### 2
# 아티스트별로 구분하여 비슷한 아티스트끼리 묶기



if __name__ == '__main__':
    pass