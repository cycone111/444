import torch
from crypto.tensor.RingTensor import RingTensor


# test the sum function in RingTensor
def test_sum():
    print("test sum function in RingTensor")
    # create a tensor
    x = RingTensor.convert_to_ring(torch.tensor([[1, 2, 3], [4, -5, 6]]))
    # x = RingTensor()
    print(x.sum(0))
    print(x.sum(0).convert_to_real_field())
    print()


# test the T function in RingTensor
def test_T():
    print("test T function in RingTensor")
    # create a tensor
    x = RingTensor.convert_to_ring(torch.tensor([[1, 2, 3], [4, -5, 6]]))
    print(x.T())
    print(x.T().convert_to_real_field())
    print()


# test the add function in RingTensor
def test_add():
    print("test add function in RingTensor")
    x = RingTensor.convert_to_ring(torch.tensor([[1, 2, -3], [4, -5, 6]]))
    # add a RingTensor
    y = RingTensor.convert_to_ring(torch.tensor([[1, 2, 3], [4, -5, 6]]))
    print((x + y))
    print((x + y).convert_to_real_field())
    print()


# test the sub function in RingTensor
def test_sub():
    print("test sub function in RingTensor")
    x = RingTensor.convert_to_ring(torch.tensor([[1, 2, -3], [4, -5, 6]]))
    # sub a RingTensor
    y = RingTensor.convert_to_ring(torch.tensor([[1, 2, 3], [4, -5, 6]]))
    print(x - y)
    print((x - y).convert_to_real_field())
    print()


# test the mul function in RingTensor
def test_mul():
    print("test mul function in RingTensor")
    x = RingTensor.convert_to_ring(torch.tensor([[1, 2, -3], [4, -5, 6]]))
    # mul a RingTensor
    y = RingTensor.convert_to_ring(torch.tensor([[1, 2, 3], [4, -5, 6]]))
    print(x * y)
    print((x * y).convert_to_real_field())
    print()


# test the neg function in RingTensor
def test_neg():
    print("test neg function in RingTensor")
    # create a tensor
    x = RingTensor.convert_to_ring(torch.tensor([[1, 2, 3], [4, -5, 6]]))
    print(-x)
    print((-x).convert_to_real_field())
    print()


# test random function in RingTensor
def test_random():
    print("test random function in RingTensor")
    # create a tensor
    x = RingTensor.random([2, 2])
    print(x)
    print()


# test save and load function in RingTensor
def test_save_load():
    print("test save and load function in RingTensor")
    # create a tensor
    x = RingTensor.random([2, 2])
    print(x)
    # save
    x.save('test.pt')
    # load
    y = RingTensor.load_from_file('test.pt')
    print(y)
    print()


def test():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    x_ring = RingTensor(x)
    print(x_ring)



def test_convert():
    x = torch.tensor([[1, 2, 3], [4, -5, 6]])
    x_ring = RingTensor.convert_to_ring((x))
    print(x_ring)

    x = x_ring.convert_to_real_field()
    print(x)


test_convert()
test_sum()
test_T()
test_add()
test_sub()
test_mul()
test_neg()
test_random()
test_save_load()
