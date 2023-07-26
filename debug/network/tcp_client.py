from common.network import tcp
from configs.network_config import *
import torch

client = tcp.create_tcp(TEST_SERVER_ADDRESS, TEST_SERVER_PORT, 'client')
client.run()

receive_data = client.receive_torch_array()
print("receive_data", receive_data)
