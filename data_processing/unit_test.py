import numpy as np
import adapt_norm_sp as AN

if __name__ == '__main__':

    sequence = np.array([
        [
            [0,1,2,3,4,5],
            [10,11,12,13,14,15],
        ],
        [
            [20,21,22,23,24,25],
            [30,31,32,33,34,35],
        ],
    ])

    ma_win = 2
    MA = AN.EMA(sequence, ma_win)

    sl_win = 3
    #res = AN.get_R(sequence, MA, sl_win)
    print(MA.shape, '\n\n')
    print(sequence[:,:,0:sl_win].shape)