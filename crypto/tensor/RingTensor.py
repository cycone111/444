import torch
from configs.base_config import RING_MAX, SCALE, INVERSE, BIT_LEN


class RingTensor(object):
    def __init__(self, ring_tensor, dtype='int', scale=1):
        self.tensor = ring_tensor
        self.shape = ring_tensor.shape
        self.dtype = dtype
        self.scale = scale
        self.bit_len = BIT_LEN

    def __str__(self):
        return "[{}\n value:{} \n dtype:{} \n scale:{}]".format(self.__class__.__name__, self.tensor, self.dtype,
                                                                self.scale)

    @staticmethod
    def convert_to_ring(torch_tensor):
        if not isinstance(torch_tensor, torch.Tensor):
            raise TypeError("unsupported data type(s): ", type(torch_tensor))
        if torch_tensor.dtype == torch.int64 or torch_tensor.dtype == torch.int32:
            v = torch_tensor & (RING_MAX - 1)  # the conversion from int64 to int32 is done here
            dtype = 'int'
            scale = 1
        elif torch_tensor.dtype == torch.float64 or torch_tensor.dtype == torch.float32:
            dtype = 'float'
            scale = SCALE
            # 这里取整函数使用round，如果使用floor或ceil函数，会出现 (-0.4 + 0.4 != 0)的情况
            v = torch.round(torch_tensor * scale).long()
            v = v & (RING_MAX - 1)
        else:
            raise TypeError("unsupported torch.dtype(s): ", torch_tensor.dtype)
        r = RingTensor(v)
        r.dtype = dtype
        r.scale = scale
        return r

    @staticmethod
    def random(shape, dtype='int', scale=1):
        if dtype == 'int' or dtype == 'float':
            v = torch.randint(0, RING_MAX, shape)
        else:
            raise TypeError("unsupported dtype(s): ", dtype)
        return RingTensor(v, dtype, scale)

    # convert ring field to real field
    def convert_to_real_field(self):
        negative_mask = self.tensor > RING_MAX / 2
        tensor = torch.where(negative_mask, self.tensor - RING_MAX, self.tensor)
        if self.dtype == 'float':
            tensor = tensor / self.scale
        return tensor

    # sum along axis
    def sum(self, axis):
        new_value = torch.sum(self.tensor, dim=axis)
        return RingTensor(new_value, self.dtype, self.scale)

    # transpose
    def T(self):
        new_value = self.tensor.T
        return RingTensor(new_value, self.dtype, self.scale)

    # add function on ring (plaintext + plaintext)
    def __add__(self, other):
        if isinstance(other, RingTensor):
            new_value = (self.tensor + other.tensor) % RING_MAX
        else:
            raise TypeError(
                "unsupported operand type(s) for + 'RingTensor' and ", type(other), 'please convert to ring first')
        return RingTensor(new_value, self.dtype, self.scale)

    # sub function on ring (plaintext - plaintext)
    def __sub__(self, other):
        if isinstance(other, RingTensor):
            new_value = (self.tensor - other.tensor) % RING_MAX
        else:
            raise TypeError(
                "unsupported operand type(s) for - 'RingTensor' and ", type(other), 'please convert to ring first')
        return RingTensor(new_value, self.dtype, self.scale)

    # mul function on ring (plaintext * plaintext)
    def __mul__(self, other):
        if isinstance(other, RingTensor):
            if self.dtype == 'int':
                new_value = (self.tensor * other.tensor) % RING_MAX
            elif self.dtype == 'float':
                # 首先将两个矩阵中大于RING_MAX/2的元素转换为负数，然后进行乘法运算,先截断再取模
                negative_mask = self.tensor > RING_MAX / 2
                self.tensor = torch.where(negative_mask, self.tensor - RING_MAX, self.tensor)
                negative_mask = other.tensor > RING_MAX / 2
                other.tensor = torch.where(negative_mask, other.tensor - RING_MAX, other.tensor)

                new_value = ((self.tensor * other.tensor) / SCALE).long() % RING_MAX
            else:
                raise TypeError("unsupported dtype(s): ", self.dtype)
        else:
            raise TypeError(
                "unsupported operand type(s) for * 'RingTensor' and ", type(other), 'please convert to ring first')
        return RingTensor(new_value, self.dtype, self.scale)

    # mod function on ring (plaintext % int)
    def __mod__(self, other):
        if isinstance(other, int):
            new_value = (self.tensor % other)
        else:
            raise TypeError(
                "unsupported operand type(s) for % 'RingTensor' and ", type(other), 'please convert to ring first')
        return RingTensor(new_value, self.dtype, self.scale)

    # neg function on ring (-plaintext)
    def __neg__(self):
        new_value = (- self.tensor) % RING_MAX
        return RingTensor(new_value, self.dtype, self.scale)

    def save(self, file_path):
        torch.save(self.tensor, file_path)
        print("Successfully save to ", file_path)

    @staticmethod
    def load_from_file(file_path):
        return RingTensor(torch.load(file_path))

    @staticmethod
    def load_from_value(v, dtype, scale):
        return RingTensor(v, dtype, scale)

    # clone a ring tensor to new ring tensor
    def clone(self):
        return RingTensor(self.tensor.clone(), self.dtype, self.scale)

    def __getitem__(self, item):
        return RingTensor(self.tensor[item], self.dtype, self.scale)


    def __setitem__(self, key, value):
        self.tensor[key] = value.tensor

    # new tensor maybe ?
    def reshape(self, shape):
        self.tensor = self.tensor.reshape(shape)
        return self
