import torch
import torch.nn as nn


class VariationalConv2d(nn.Conv2d):
    """
    Вариационная свертка
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log_sigma = nn.Parameter(torch.ones(self.weight.shape).to(self.weight.device) * -3.0)
        self.weight.sigma = self.log_sigma

        if self.bias:
            self.log_sigma_b = nn.Parameter(torch.ones(self.bias.shape).to(self.bias.device) * -3.0)
            self.bias.sigma = self.log_sigma_b

    def forward(self, x):
        eps = 1e-8

        conved_mu = nn.functional.conv2d(x, self.weight, self.bias, self.stride,
                                         self.padding, self.dilation, self.groups)

        conved_si = torch.sqrt(eps + nn.functional.conv2d(x * x, 0.01 + torch.exp(2 * self.log_sigma),
                                                          self.bias, self.stride, self.padding, self.dilation,
                                                          self.groups))

        conved = conved_mu + conved_si * torch.normal(torch.zeros_like(conved_mu), torch.ones_like(conved_mu))

        if self.bias:
            conved += (0.01 + torch.exp(2 * self.log_sigma_b)) * torch.normal(torch.zeros_like(conved_mu),
                                                                              torch.ones_like(conved_mu))
        return conved


class VariationalLinear(nn.Linear):
    """
    Вариационный линейный слой
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log_sigma = nn.Parameter(torch.ones(self.weight.shape).to(self.weight.device) * -3.0)
        self.weight.sigma = self.log_sigma

        self.log_sigma_b = nn.Parameter(torch.ones(self.bias.shape).to(self.bias.device) * -3.0)
        self.bias.sigma = self.log_sigma_b

    def forward(self, x):
        mu = x.matmul(self.weight.t())
        eps = 1e-8

        si = torch.sqrt((x * x).matmul(((0.01 + torch.exp(2 * self.log_sigma)) + eps).t()))

        activation = mu + torch.normal(torch.zeros_like(mu),
                                       torch.ones_like(mu)) * si + (0.01 + torch.exp(2 * self.log_sigma_b)) \
                                       * torch.normal(torch.zeros_like(mu), torch.ones_like(mu))
        return activation + self.bias


class DropPath(nn.Module):
    def __init__(self, p=0.):
        """
        Drop path with probability.
        Parameters
        ----------
        p : float
            Probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0.:
            keep_prob = 1. - self.p
            # per data point mask
            mask = torch.zeros((x.size(0), 1, 1, 1), device=x.device).bernoulli_(keep_prob)
            return x / keep_prob * mask

        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool with BN. `pool_type` must be `max` or `avg`.
    """

    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    """
    Standard conv: ReLU - Conv - BN
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            VariationalConv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """
    Factorized conv: ReLU - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            VariationalConv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
            VariationalConv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """
    (Dilated) depthwise separable conv.
    ReLU - (Dilated) depthwise separable - Pointwise - BN.
    If dilation == 2, 3x3 conv => 5x5 receptive field, 5x5 conv => 9x9 receptive field.
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            VariationalConv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                              bias=False),
            VariationalConv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """
    Depthwise separable conv.
    DilConv(dilation=1) * 2.
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise (stride=2).
    """

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = VariationalConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = VariationalConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
