import torch
from crypto.party.Party import SemiHonestCS
from crypto.primitives.ArithmeticSecretSharing import ArithmeticSecretSharing
from crypto.tensor.RingTensor import RingTensor
from configs.network_config import *
from crypto.primitives.beaver import BeaverOfflineProvider
from crypto.msb.msb_utils import *

server = SemiHonestCS(type='server')
server.set_address(TEST_SERVER_ADDRESS)
server.set_port(TEST_SERVER_PORT)
server.set_dtype('int')
server.set_scale(1)
server.set_beaver_provider(BeaverOfflineProvider())
server.beaver_provider.load_triples(server, 2)
server.connect()

# test arithmetic secret sharing
x = torch.tensor([[1, 2, 3, 4, 5], [2, -3, -4, 5, 6]])
x_ring = RingTensor.convert_to_ring(x)
x_0, x_1 = ArithmeticSecretSharing.share(x_ring, 2)

server.send_ring_tensor(x_1)
shared_x = ArithmeticSecretSharing(x_0, server)

shared_cb = get_MSB(shared_x)

print(shared_cb)

print(debug(shared_cb, server))
