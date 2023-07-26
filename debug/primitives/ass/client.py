import torch

from crypto.party.Party import SemiHonestCS
from crypto.primitives.ArithmeticSecretSharing import ArithmeticSecretSharing
from crypto.tensor.RingTensor import RingTensor
from configs.network_config import *
from configs.base_config import *
from crypto.primitives.beaver import BeaverOfflineProvider

client = SemiHonestCS(type='client')
client.set_address(TEST_SERVER_ADDRESS)
client.set_port(TEST_SERVER_PORT)
client.set_dtype(DTYPE)
client.set_scale(SCALE)
client.set_beaver_provider(BeaverOfflineProvider())
client.beaver_provider.load_triples(client, 2)
client.connect()

# receive other shares from server
shared_x = ArithmeticSecretSharing(client.receive_ring_tensor(), client)

# receive other shares from server
shared_y = ArithmeticSecretSharing(client.receive_ring_tensor(), client)

z_shared = shared_x * shared_y
z_shared.restore()