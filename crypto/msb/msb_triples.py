import torch

Ring = 2 ** 32
data_path = "./data/msb/msb_triples/"
k = 35
n = 2000


# ################################################################################

# def randBool(*size):
#     return torch.randint(0, 2, size) == torch.randint(0, 2, size)
#
#
# def share_bool(t):
#     t_0 = randBool(n, k)
#     t_1 = t ^ t_0
#     return t_0, t_1
#
#
# a = randBool(n, k)
# b = randBool(n, k)
#
# c = a & b
#
# a_0, a_1 = share_bool(a)
# b_0, b_1 = share_bool(b)
# c_0, c_1 = share_bool(c)
#
# torch.save(a_0, data_path + "a_0.pth")
# torch.save(a_1, data_path + "a_1.pth")
# torch.save(b_0, data_path + "b_0.pth")
# torch.save(b_1, data_path + "b_1.pth")
# torch.save(c_0, data_path + "c_0.pth")
# torch.save(c_1, data_path + "c_1.pth")

# #############################################################################

a_0 = torch.load(data_path + "a_0.pth")
b_0 = torch.load(data_path + "b_0.pth")
c_0 = torch.load(data_path + "c_0.pth")
a_1 = torch.load(data_path + "a_1.pth")
b_1 = torch.load(data_path + "b_1.pth")
c_1 = torch.load(data_path + "c_1.pth")


def get_triples_msb(p, n, l):
    '''
    :param p: 客户端还是服务器
    :param n: 取几维的数据
    :param l: 每一维度去多少数据
    :return: a,b,c三元组
    '''

    if p == 0:
        return a_0[:n, :l], b_0[:n, :l], c_0[:n, :l]
    else:
        return a_1[:n, :l], b_1[:n, :l], c_1[:n, :l]
