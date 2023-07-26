from common.network import tcp
from configs.network_config import *
import torch

server = tcp.create_tcp(TEST_SERVER_ADDRESS, TEST_SERVER_PORT, 'server')
server.run()

send_data = torch.randint(0, 10, [10])
print("send_data", send_data)

server.send_torch_array(send_data)
