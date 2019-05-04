import time

import torch

import dtw_cpp


_dtw_scaling_factor = 100


class PairwiseDistance(torch.nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x, y):
        x_ = x.repeat([1] + list(y.shape[1:])).reshape(*y.shape, -1)
        y_ = y.repeat([1] + list(x.shape[1:])).reshape(*x.shape, -1).transpose(-1, -2)
        return x_.sub(y_).abs().pow(self.p)


class SoftDTWFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = dtw_cpp.forward(input, _dtw_scaling_factor)
        if len(output) > 1:
            ctx.save_for_backward(output[1])
        #print("output", output[0])
        return output[0][:, -1, -1]

    @staticmethod
    def backward(ctx, grad_output):
        (dtw_argmin,) = ctx.saved_tensors
        out=dtw_cpp.backward(dtw_argmin, _dtw_scaling_factor) * grad_output
        #print("grad", out)
        #raise SystemExit
        return out


class SoftDTWLoss(torch.nn.Module):
    def __init__(self, distance_module=PairwiseDistance()):
        super().__init__()
        self.distance_module = distance_module

    def forward(self, input):
        D = self.distance_module(*input)
        return SoftDTWFunction.apply(D)


import unittest


class SoftDTWFunctionTest(unittest.TestCase):
    cost_matrix = PairwiseDistance()
    dtw_func = SoftDTWFunction.apply

    def make_data(self, batch_size=1, seq_len_1=3, seq_len_2=4):
        x1 = torch.randint(
            3, [batch_size, seq_len_1], requires_grad=True, dtype=torch.float32
        )
        x2 = torch.randint(
            3, [batch_size, seq_len_2], requires_grad=True, dtype=torch.float32
        )
        x1 = torch.tensor([[1, 2, 0]], requires_grad=True, dtype=torch.float32)
        x2 = torch.tensor([[2, 0, 0, 1]], requires_grad=True, dtype=torch.float32)
        #print(x1, x2)
        C = self.cost_matrix(x1, x2).to(torch.float64)
        #print(C)
        return C

    def test_grad(self):
        C = self.make_data()

        def func(C):
            return self.dtw_func(C)

        torch.autograd.gradcheck(func, C, eps=1e-3, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
