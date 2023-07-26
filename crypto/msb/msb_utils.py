import torch

from crypto.msb import msb_triples as triples

Ring = 2 ** 32


class msb_class(object):
    def __init__(self, x):
        self.value = x.ring_tensor.tensor
        self.p = x.party.party_id
        self.tcp = x.party.tcp


def get_carry_bit_sonic(x, n):
    # 获取 P 和 G 数组，这里使用 x.value 的值作为 P 的初始值    P 只需要进行异或操作，所以直接继承过来。  G需要与操作，需要做转化
    P_i_layer1 = x.value
    G_i_layer1 = get_G_array(x, n)

    # layer 1
    # 调用 get_P_and_G_array 函数进行一系列位运算，得到 P_pre 和 G_pre
    P_pre, G_pre = get_P_and_G_array(x, 31, 15, P_i_layer1, G_i_layer1, True, n)

    # 初始化大小为 (n, 16) 的全零张量，并将 P_pre 和 G_pre 的值拷贝进去，实现位拼接
    P_i_2 = torch.zeros(size=(n, 16)).bool()
    G_i_2 = torch.zeros(size=(n, 16)).bool()

    P_i_2[:, 1:] = P_pre
    P_i_2[:, 0] = P_i_layer1[:, 0]

    G_i_2[:, 1:] = G_pre
    G_i_2[:, 0] = G_i_layer1[:, 0]

    # layer2
    # 调用 get_P_and_G_array 函数进行一系列位运算，得到 P_i_layer3 和 G_i_layer3
    P_i_layer3, G_i_layer3 = get_P_and_G_array(x, 16, 8, P_i_2, G_i_2, False, n)

    # layer3
    # 调用 get_P_and_G_array 函数进行一系列位运算，得到 P_i_layer4 和 G_i_layer4
    P_i_layer4, G_i_layer4 = get_P_and_G_array(x, 8, 4, P_i_layer3, G_i_layer3, False, n)

    # layer4
    # 调用 get_P_and_G_array 函数进行一系列位运算，得到 P_i_layer5 和 G_i_layer5
    P_i_layer5, G_i_layer5 = get_P_and_G_array(x, 4, 2, P_i_layer4, G_i_layer4, False, n)

    # layer5
    # 调用 triples.get_triples_msb 函数获取三元组 a, b, c 的值
    a, b, c = triples.get_triples_msb(x.p, 1, 1)

    # 提取 G_i_layer5 的第 2 列（索引为 1）
    G2_i_layer5 = G_i_layer5[:, 1]

    # 进行位运算得到 F_i_P2_layer5 和 E_i_G1_layer5
    F_i_P2_layer5 = P_i_layer5[:, 1] ^ b
    E_i_G1_layer5 = G_i_layer5[:, 0] ^ a

    # 将 F_i_P2_layer5 和 E_i_G1_layer5 作为输入，通过通信协议发送给另一方并接收对方的结果
    x.tcp.send_torch_array(torch.stack((F_i_P2_layer5, E_i_G1_layer5), dim=1))
    get_arr_layer5 = x.tcp.receive_torch_array()

    # 对接收到的结果进行异或运算得到 F_of_P2_layer5 和 E_of_G1_layer5
    F_of_P2_layer5 = F_i_P2_layer5 ^ get_arr_layer5[:, 0]
    E_of_G1_layer5 = E_i_G1_layer5 ^ get_arr_layer5[:, 1]

    # 调用 C_and_2party 函数进行一系列位运算，得到 Cb_i
    Cb_i = C_and_2party(x.p, E_of_G1_layer5, F_of_P2_layer5, a, b, c) ^ G2_i_layer5

    # 返回结果 Cb_i
    return Cb_i


def C_and_2party(p, E, F, a, b, c):
    # 相当于ASS中的乘法协议
    if p == 0:
        return (E & F) ^ (E & b) ^ (F & a) ^ c
    else:
        return (E & b) ^ (F & a) ^ c


def get_G_array(x, n):
    # 调用 triples.get_triples_msb 函数获取三元组 a, b, c 的值
    a, b, c = triples.get_triples_msb(x.p, 1, 32)

    # 初始化大小为 (n, 32) 的全零张量 x_j
    x_j = torch.zeros(size=(n, 32)).bool()

    # 根据 x.p 的值选择 E_i 和 F_i 的值
    if x.p == 1:
        E_i = x.value ^ a
        F_i = x_j ^ b
    else:
        E_i = x_j ^ a
        F_i = x.value ^ b

    # 将 E_i 和 F_i 拼接成一个张量，并通过通信协议发送给另一方，然后接收对方的结果
    x.tcp.send_torch_array(torch.cat((E_i, F_i), dim=0))
    get_array = x.tcp.receive_torch_array()

    # 计算 G 数组的长度
    len = int(get_array.shape[0] / 2)

    # 根据接收到的结果计算 E_of_G 和 F_of_G
    E_of_G = get_array[:len] ^ E_i
    F_of_G = get_array[len:] ^ F_i

    # 调用 C_and_2party 函数进行一系列位运算，得到 G_i_layer1
    G_i_layer1 = C_and_2party(x.p, E_of_G, F_of_G, a, b, c)

    # 返回 G 数组 G_i_layer1
    return G_i_layer1


def get_P_and_G_array(x, end, l, P_pre, G_pre, is_layer1, n):
    # 根据 is_layer1 的值决定 begin 的值
    if is_layer1:
        begin = 1
    else:
        begin = 0

    # 从 G_pre 中提取第 2 列（索引为 1）到 end 列的子数组，得到 G2_pre
    G2_pre = G_pre[:, begin + 1:end:2]

    # 调用 triples.get_triples_msb 函数获取三元组 P1_a, P2_bP, c_P1P2 和 G1_A, P2_bG, c_G1P2 的值
    P1_a, P2_bP, c_P1P2 = triples.get_triples_msb(x.p, 1, l)
    G1_A, P2_bG, c_G1P2 = triples.get_triples_msb(x.p, 1, l)

    # 计算 E_i_P1、E_i_G1、FP_i_P2 和 FG_i_P2
    E_i_P1 = P_pre[:, begin:end:2] ^ P1_a
    E_i_G1 = G_pre[:, begin:end:2] ^ G1_A
    FP_i_P2 = P_pre[:, begin + 1:end:2] ^ P2_bP
    FG_i_P2 = P_pre[:, begin + 1:end:2] ^ P2_bG

    # 将 E_i_P1、FP_i_P2、E_i_G1 和 FG_i_P2 拼接成一个张量，并通过通信协议发送给另一方，然后接收对方的结果
    send_array = torch.cat((E_i_P1, FP_i_P2, E_i_G1, FG_i_P2), dim=1)
    x.tcp.send_torch_array(send_array)
    get_arr = x.tcp.receive_torch_array()

    # 计算各个数组的长度
    len = int(get_arr.shape[1] / 4)

    # 根据接收到的结果计算 E_of_P1、FP_of_P2、E_of_G1 和 FG_of_P2
    E_of_P1 = get_arr[:, 0:len] ^ E_i_P1
    FP_of_P2 = get_arr[:, len:len * 2] ^ FP_i_P2
    E_of_G1 = get_arr[:, len * 2:len * 3] ^ E_i_G1
    FG_of_P2 = get_arr[:, len * 3:len * 4] ^ FG_i_P2

    # 调用 C_and_2party 函数进行一系列位运算，得到 P 和 G 数组
    P = C_and_2party(x.p, E_of_P1, FP_of_P2, P1_a, P2_bP, c_P1P2)
    G = C_and_2party(x.p, E_of_G1, FG_of_P2, G1_A, P2_bG, c_G1P2) ^ G2_pre

    # 返回 P 和 G 数组
    return P, G


def int2bite_arr(x, size):
    # 创建一个大小为 (size, 32) 的全零张量，并将其类型设置为 bool 类型
    arr = torch.zeros(size=(size, 32)).bool()

    # 循环从 0 到 31
    for i in range(0, 32):
        # 通过右移操作将整数 x 的第 i 位提取出来，并和 0x01 进行与运算
        # 这将把 x 的第 i 位转换成一个 0 或 1，并生成一个形状为 (1, size) 的张量
        # 最后将该张量赋值给 arr 的第 i 列
        arr[:, i] = ((x >> i) & 0x01).reshape(1, size)

    # 返回转换后的结果
    return arr


def get_MSB(x):
    # 调用一个名为 msb_class 的函数，它可能是返回包含输入 x 的最高有效位的对象
    x = msb_class(x)

    # 获取 x 值的形状和元素数量
    shape = x.value.shape
    size = x.value.numel()

    # 将 x 的值从原始形状转换为一个形状为 (1, size) 的张量
    x.value = x.value.reshape(1, size)

    # 调用一个名为 int2bite_arr 的函数，将 x 转换成一个大小为 (size, 32) 的布尔型张量
    x.value = int2bite_arr(x.value, size)

    # 调用一个名为 get_carry_bit_sonic 的函数，得到x的最高进位，大小为（size, 1)
    x = get_carry_bit_sonic(x, size)

    # 将 x 从形状为 (size, 1)转换回原始形状 (1， size)
    x = x.reshape(1, size)

    # 将 x 从形状为 (1, size) 转换回原始形状 shape
    x = x.reshape(shape)

    # 返回处理后的结果
    return x


def debug(cb, tcp):
    tcp.send_tensor(cb)
    res = tcp.receive_tensor() ^ cb

    return res
