DEVICE = 'cpu'

BIT_LEN = 32
RING_MAX = 2**BIT_LEN
HALF_RING = 2**(BIT_LEN - 1)

triple_path = './data/triples_data/'

# 定点数设置
DTYPE = 'float'
SCALE = 1009
INVERSE = 2668924177
LEN_DECIMAL = 10
LEN_INTEGER = 4
KAPPA = 2