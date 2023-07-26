import torch
from crypto.party.Party import SemiHonestCS
from crypto.primitives.ArithmeticSecretSharing import ArithmeticSecretSharing
from crypto.tensor.RingTensor import RingTensor
from configs.network_config import *
from configs.base_config import *
from crypto.primitives.beaver import BeaverOfflineProvider

server = SemiHonestCS(type='server')
server.set_address(TEST_SERVER_ADDRESS)
server.set_port(TEST_SERVER_PORT)
server.set_dtype(DTYPE)
server.set_scale(SCALE)
server.set_beaver_provider(BeaverOfflineProvider())
server.beaver_provider.load_triples(server, 2)
server.connect()

# test arithmetic secret sharing
x = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5],[0.2, 0.3, 0.4, 0.5, 0.6]])
x_ring = RingTensor.convert_to_ring(x)
x_0, x_1 = ArithmeticSecretSharing.share(x_ring, 2)

# send other shares to client
server.send_ring_tensor(x_1)
shared_x = ArithmeticSecretSharing(x_0, server)


# test arithmetic secret sharing
y = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6]])
y_ring = RingTensor.convert_to_ring(y)
y_0, y_1 = ArithmeticSecretSharing.share(y_ring, 2)
# send other shares to client
server.send_ring_tensor(y_1)
shared_y = ArithmeticSecretSharing(y_0, server)

z_shared = shared_x * shared_y

print(z_shared.restore().convert_to_real_field())