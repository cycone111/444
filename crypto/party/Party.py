from common.network.tcp import create_tcp
from crypto.tensor.RingTensor import RingTensor
from crypto.primitives.ArithmeticSecretSharing import ArithmeticSecretSharing
from common.network.tcp import *
from configs.base_config import BIT_LEN
from configs.network_config import *

class Party(object):
    def __init__(self, party_id):
        self.party_id = party_id



class SemiHonestCS(Party):
    def __init__(self, type='client'):
        # type must be 'client' or 'server'
        if not type in ['client', 'server']:
            raise ValueError("type must be 'client' or 'server'")
        self.type = type
        # client id is 1, server id is 0
        party_id = 0
        if type == 'client':
            party_id = 1
        super(SemiHonestCS, self).__init__(party_id)

        self.tcp_address = TEST_SERVER_ADDRESS
        self.tcp_port = TEST_SERVER_PORT
        self.tcp = create_tcp(self.tcp_address, self.tcp_port, self.type)

        # beaver 先测试用
        self.beaver_provider = None


    def set_scale(self, scale):
        self.scale = scale

    def set_dtype(self, dtype):
        self.dtype = dtype

    def set_address(self, address):
        self.tcp_address = address

    def set_port(self, port):
        self.tcp_port = port

    def set_beaver_provider(self, beaver_provider):
        self.beaver_provider = beaver_provider

    def connect(self):
        self.tcp.run()

    def send_shares(self, x):
        self.tcp.send_torch_array(x.ring_tensor.tensor)

    def send_tensor(self, x):
        self.tcp.send_torch_array(x)

    def send_ring_tensor(self, x):
        self.tcp.send_torch_array(x.tensor)

    def receive_shares(self):
        v = self.tcp.receive_torch_array()
        r = RingTensor.load_from_value(v, self.dtype, self.scale)
        return ArithmeticSecretSharing.load_from_ring_tensor(r, self)

    def receive_tensor(self):
        v = self.tcp.receive_torch_array()
        return v

    def receive_ring_tensor(self):
        v = self.tcp.receive_torch_array()
        return RingTensor.load_from_value(v, self.dtype, self.scale)