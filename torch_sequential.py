import gamspy as gp
import torch
from torch import nn

class TorchSequential:
    @staticmethod
    def convert_layer(m: gp.Container, elem):
        clz = elem.__class__.__name__
        if clz == "Linear":
            has_bias = elem.bias is not None
            l = gp.formulations.Linear(
                m,
                in_features=elem.in_features,
                out_features=elem.out_features,
                bias=has_bias,
            )
            l.load_weights(
                elem.weight.numpy(),
                elem.bias.numpy() if has_bias else None
            )
            return l
        if clz == "Conv1d":
            if elem.dilation[0] != 1:
                raise NotImplementedError("Conv1d is not supported when dilation is not 1")

            if elem.groups != 1:
                raise NotImplementedError("Conv1d is not supported when groups is not 1")

            if elem.padding_mode != "zeros":
                raise NotImplementedError("Conv1d is only supported with padding_mode zeros")

            has_bias = elem.bias is not None
            l = gp.formulations.Conv1d(
                m,
                in_channels=elem.in_channels,
                out_channels=elem.out_channels,
                kernel_size=elem.kernel_size,
                stride=elem.stride,
                padding=elem.padding,
                bias=has_bias
            )

            l.load_weights(
                elem.weight.numpy(),
                elem.bias.numpy() if has_bias else None
            )
            return l
        if clz == "Conv2d":
            if elem.dilation[0] != 1 or elem.dilation[-1] != 1:
                raise NotImplementedError("Conv2d is not supported when dilation is not 1")

            if elem.groups != 1:
                raise NotImplementedError("Conv2d is not supported when groups is not 1")

            if elem.padding_mode != "zeros":
                raise NotImplementedError("Conv1d is only supported with padding_mode zeros")

            has_bias = elem.bias is not None
            l = gp.formulations.Conv2d(
                m,
                in_channels=elem.in_channels,
                out_channels=elem.out_channels,
                kernel_size=elem.kernel_size,
                stride=elem.stride,
                padding=elem.padding,
                bias=has_bias
            )

            l.load_weights(
                elem.weight.numpy(),
                elem.bias.numpy() if has_bias else None
            )
            return l
        elif clz == "ReLU":
            return gp.math.relu_with_binary_var

        else:
            raise NotImplementedError(f"Formulation for {clz} not implemented!")


    def __init__(self, m: gp.Container, network: nn.Sequential):
        with torch.no_grad():
            self.layers = [TorchSequential.convert_layer(m, layer) for layer in network]

    def __call__(self, input: gp.Variable):
        out = input
        equations = []
        for layer in self.layers:
            out, layer_eqs = layer(out)
            equations.extend(layer_eqs)

        return out, equations
