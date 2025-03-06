from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        *,  # enforce kwargs
        input_dim,
        reduction_factor,
        n_hidden,
        output_dim,
    ):
        super().__init__()

        blocks = []
        for curr_layer in range(n_hidden):
            layer_in = input_dim / reduction_factor**curr_layer
            layer_out = input_dim / reduction_factor ** (curr_layer + 1)
            assert int(layer_in) == layer_in
            assert int(layer_out) == layer_out
            block = _Block(
                input_dim=int(layer_in),
                output_dim=int(layer_out),
            )
            blocks.append(block)
        self.hs = nn.ModuleList(blocks)
        self.proj = nn.Linear(int(layer_out), output_dim)

    def forward(self, x):
        for h in self.hs:
            x = h(x)
        x = self.proj(x)
        return x


class _Block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.h = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.h(x)
        x = self.bn(x)
        x = self.act(x)
        return x
