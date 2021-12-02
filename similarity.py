'''
This module verifies that the trained model classifies
the presentation of POP music well.

== The Artist List ==
['Beenzino', 'Oasis', '2NE1', 'IU', 'Green Day'
 'NELL', 'LOCO', 'Dinimic Duo', 'Red Velvet', 'Pink Floyd']
'''
# library
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import time
import torch.functional as F


parser = argparse.ArgumentParser(description='Measure Similarity')
parser.add_argument('--save_path', default='./exp', type=str,
                    help='save path')
args = parser.parse_args()
start = time.time()



# 두 vector 사이의 유사도를 return
def get_similarity(a,b,distance_measure=None):
    if distance_measure not in ['euclidean','cosine']:
        raise ValueError('Wrong distance metric')
    if distance_measure == 'euclidean':
        sim = torch.sum(torch.sqrt(torch.square(a-b)))
    elif distance_measure == 'cosine':
        cos = nn.CosineSimilarity()
        sim = cos(a,b)
    return sim



### 2
# 아티스트별로 구분하여 비슷한 아티스트끼리 묶기


def main():

    print("Process Complete : it took {time:.2f} minutes".format(time=(time.time()-start)/60))


if __name__ == '__main__':
    main()