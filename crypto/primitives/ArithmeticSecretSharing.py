import torch

from crypto.tensor.RingTensor import RingTensor
from configs.base_config import RING_MAX, INVERSE, SCALE


class ArithmeticSecretSharing(object):
    def __init__(self, ring_tensor, party):
        self.DEBUG = False
        self.party = party
        self.ring_tensor = ring_tensor
        self.shape = ring_tensor.shape
        # if tensor is IntTensor then dtype is INT else if tensor is FloatTensor then dtype is FLOAT
        if not isinstance(ring_tensor, RingTensor):
            raise TypeError("ArithmeticSecretSharing only supports RingTensor")

    def __str__(self):
        return "[{}\n value:{},\n party:{}]".format(self.__class__.__name__, self.ring_tensor.tensor,
                                                    self.party.party_id)


    def sum(self, axis):
        new_tensor = self.ring_tensor.sum(axis)
        return ArithmeticSecretSharing(new_tensor, self.party)

    def T(self):
        new_tensor = self.ring_tensor.T()
        return ArithmeticSecretSharing(new_tensor, self.party)


    def __add__(self, other):
        if isinstance(other, ArithmeticSecretSharing):
            # for share tensor, each party add its share of the tensor locally
            new_tensor = self.ring_tensor + other.ring_tensor
        elif isinstance(other, RingTensor):  # for ring tensor, only party 0 add it to the share tensor
            if self.party.id == 0:
                new_tensor = self.ring_tensor + other
            else:
                new_tensor = self.ring_tensor
        else:
            raise TypeError("unsupported operand type(s) for + 'ArithmeticSecretSharing' and ", type(other))
        return ArithmeticSecretSharing(new_tensor, self.party)

    def __mul__(self, other):
        if isinstance(other, ArithmeticSecretSharing):
            return beaver_mul(self.party.beaver_provider, self, other)
        elif isinstance(other, RingTensor):
            # if other is a ring tensor, then it can be computed locally
            new_tensor = self.ring_tensor * other
        elif isinstance(other, int):
            # 如果相乘的另一个数是int，那么可以直接计算，并且是不调用RingTensor乘法的   这里先保留这种写法
            new_tensor = (self.ring_tensor.tensor * other) % RING_MAX
            new_tensor = RingTensor(new_tensor, self.ring_tensor.dtype, self.ring_tensor.scale)
        else:
            raise TypeError("unsupported operand type(s) for * 'ArithmeticSecretSharing' and ", type(other))

        return ArithmeticSecretSharing(new_tensor, self.party)


    def __sub__(self, other):
        if isinstance(other, ArithmeticSecretSharing):
            # for share tensor, each party add its share of the tensor locally
            new_tensor = self.ring_tensor - other.ring_tensor
            return ArithmeticSecretSharing(new_tensor, self.party)
        elif isinstance(other, RingTensor):
            if self.party.party_id == 0:
                new_tensor = self.ring_tensor - other
            else:
                new_tensor = self.ring_tensor
            return ArithmeticSecretSharing(new_tensor, self.party)


    def __getitem__(self, item):
        new_tensor = self.ring_tensor[item]
        return ArithmeticSecretSharing(new_tensor, self.party)


    @staticmethod
    def restore_two_shares(share_0, share_1):
        return share_0.ring_tensor + share_1.ring_tensor

    def restore(self):
        # send shares to other parties
        self.party.send_shares(self)
        # receive shares from other parties
        other = self.party.receive_shares()
        return self.ring_tensor + other.ring_tensor

    def save(self, file_path):
        self.ring_tensor.save(file_path)

    @staticmethod
    def load_from_file(file_path, party):
        ring_tensor = RingTensor.load_from_file(file_path)
        return ArithmeticSecretSharing(ring_tensor, party)

    @staticmethod
    def load_from_ring_tensor(ring_tensor, party):
        return ArithmeticSecretSharing(ring_tensor, party)

    # 放这里看看会不会有错
    @staticmethod
    def share(tensor: RingTensor, num_of_party: int):
        shares = []
        last_x = tensor.clone()
        for party_id in range(num_of_party - 1):
            r = torch.randint(0, RING_MAX, tensor.shape, dtype=torch.int64)
            x_i = RingTensor(r, dtype=tensor.dtype, scale=tensor.scale)
            shares.append(x_i)
            last_x = last_x - x_i
        shares.append(last_x)
        return shares

    def reshape(self, shape):
        self.ring_tensor.reshape(shape)
        return self


def beaver_mul(beaver_provider, x: ArithmeticSecretSharing, y: ArithmeticSecretSharing):
    a, b, c = beaver_provider.get_triples(x.shape)
    a.ring_tensor.dtype = b.ring_tensor.dtype = c.ring_tensor.dtype = x.ring_tensor.dtype
    a.ring_tensor.scale = b.ring_tensor.scale = c.ring_tensor.scale = x.ring_tensor.scale

    # compute e, f locally
    e = x - a
    f = y - b

    # restore e,f to plaintext
    plaintext_e = e.restore()
    plaintext_f = f.restore()

    sign = 0
    if x.party.party_id == 0:
        sign = 1

    res1 = (plaintext_e.tensor * plaintext_f.tensor * sign) % RING_MAX

    res2 = (a.ring_tensor.tensor * plaintext_f.tensor) % RING_MAX
    res3 = (plaintext_e.tensor * b.ring_tensor.tensor) % RING_MAX
    res = (res1 + res2 + res3 + c.ring_tensor.tensor) % RING_MAX

    res = RingTensor(res, dtype=x.ring_tensor.dtype, scale=x.ring_tensor.scale)
    res = ArithmeticSecretSharing(res, x.party)

    if x.ring_tensor.dtype == 'float':
        res = truncate(res)

    return res


def truncate(share: ArithmeticSecretSharing) -> ArithmeticSecretSharing:
    if share.party.party_id == 0:
        mask = RingTensor.random(share.shape, share.party.dtype, share.party.scale)
        mask_low = mask % share.party.scale
        share.party.send_ring_tensor(share.ring_tensor + mask)

        b_mask_low = share.party.receive_ring_tensor()
        b_low = b_mask_low - mask_low
        res = share - b_low

        res = res * INVERSE
        return res

    if share.party.party_id == 1:
        mask_low = share.party.receive_ring_tensor()
        b_masked = share.ring_tensor + mask_low
        b_masked_low = b_masked % SCALE
        share.party.send_ring_tensor(b_masked_low)

        res = share * INVERSE
        return res