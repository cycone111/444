import os
from crypto.tensor.RingTensor import RingTensor
from crypto.primitives.ArithmeticSecretSharing import ArithmeticSecretSharing
from configs.base_config import triple_path
from functools import reduce
import operator


class BeaverProvider(object):
    def __init__(self):
        pass

    @staticmethod
    def gen_triples(shape):
        a = RingTensor.random(shape)
        b = RingTensor.random(shape)
        c = a * b
        return a, b, c


class BeaverOfflineProvider(BeaverProvider):
    def __init__(self, path=triple_path):
        super(BeaverOfflineProvider, self).__init__()
        self.triple_path = path
        self.a_tensor = None
        self.b_tensor = None
        self.c_tensor = None
        self.ptr = 0

    # Run by trusted third party (maybe?)
    def generate_triple_for_parties(self, num_of_triples, num_of_party=2):
        # All triples should be one-dimensional tensor
        if type(num_of_triples) != int:
            raise Exception("All triples should be one-dimensional tensor! Please make sure that the type of "
                            "num_of_triples is int.")
        a, b, c = self.gen_triples([num_of_triples])
        # a_list = a.share(num_of_party)
        # b_list = b.share(num_of_party)
        # c_list = c.share(num_of_party)

        a_list = ArithmeticSecretSharing.share(a, num_of_party)
        b_list = ArithmeticSecretSharing.share(b, num_of_party)
        c_list = ArithmeticSecretSharing.share(c, num_of_party)

        for party_id in range(num_of_party):
            # save a, b, c to file
            # 根据party的数量创建储存文件夹
            if not os.path.exists(self.triple_path + '/' + str(num_of_party) + 'party/'):
                os.makedirs(self.triple_path + '/' + str(num_of_party) + 'party/')
            a_list[party_id].save(self.triple_path + '/' + str(num_of_party) + 'party/' + 'a_{}.pth'.format(party_id))
            b_list[party_id].save(self.triple_path + '/' + str(num_of_party) + 'party/' + 'b_{}.pth'.format(party_id))
            c_list[party_id].save(self.triple_path + '/' + str(num_of_party) + 'party/' + 'c_{}.pth'.format(party_id))

    # Run by parties
    def load_triples(self, party, num_of_party=2):
        # 读取文件
        # self.a_tensor = RingTensor.load(self.triple_path + '/' + str(num_of_party) + 'party/' + 'a_{}.pth'.format(party.id))
        # self.b_tensor = RingTensor.load(self.triple_path + '/' + str(num_of_party) + 'party/' + 'b_{}.pth'.format(party.id))
        # self.c_tensor = RingTensor.load(self.triple_path + '/' + str(num_of_party) + 'party/' + 'c_{}.pth'.format(party.id))
        self.a_tensor = ArithmeticSecretSharing.load_from_file(self.triple_path + '/' + str(num_of_party) + 'party/' + 'a_{}.pth'.format(party.party_id), party)
        self.b_tensor = ArithmeticSecretSharing.load_from_file(self.triple_path + '/' + str(num_of_party) + 'party/' + 'b_{}.pth'.format(party.party_id), party)
        self.c_tensor = ArithmeticSecretSharing.load_from_file(self.triple_path + '/' + str(num_of_party) + 'party/' + 'c_{}.pth'.format(party.party_id), party)

    # Get multiple triples by pointer
    def get_triples_by_pointer(self, number_of_triples):
        if self.a_tensor is None:
            raise Exception("Please load triples first!")
        a = self.a_tensor[self.ptr:self.ptr + number_of_triples]
        b = self.b_tensor[self.ptr:self.ptr + number_of_triples]
        c = self.c_tensor[self.ptr:self.ptr + number_of_triples]
        # update ptr
        self.ptr += number_of_triples
        # judge if ptr is out of range
        if self.ptr > self.a_tensor.shape[0]:
            raise Exception("The pointer is out of range! Need to generate more triples!")
        return a, b, c

    def get_triples(self, shape):
        # 这里先做一个统一接口，所有Provider都使用get_triples接口获取任意shape的三元组，包括各种类型
        # 需要的三元组数量
        number_of_triples = reduce(operator.mul, shape, 1)
        a, b, c = self.get_triples_by_pointer(number_of_triples)
        a.reshape(shape)
        b.reshape(shape)
        c.reshape(shape)
        return a, b, c

    def set_ptr(self, ptr):
        self.ptr = ptr
