import torch

from crypto.party.Party import SemiHonestCS
from crypto.primitives.ArithmeticSecretSharing import ArithmeticSecretSharing
from crypto.tensor.RingTensor import RingTensor
from configs.network_config import *
from crypto.primitives.beaver import BeaverOfflineProvider
from crypto.msb.msb_utils import *

client = SemiHonestCS(type='client')
client.set_address(TEST_SERVER_ADDRESS)
client.set_port(TEST_SERVER_PORT)
client.set_dtype('int')
client.set_scale(1)
client.set_beaver_provider(BeaverOfflineProvider())
client.beaver_provider.load_triples(client, 2)
client.connect()

# receive other shares from server
shared_x = ArithmeticSecretSharing(client.receive_ring_tensor(), client)


shared_cb = get_MSB(shared_x)

print("cb", shared_cb)
debug(shared_cb, client)

