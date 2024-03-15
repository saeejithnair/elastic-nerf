# %%
import torch
import tinycudann as tcnn
from torch import nn
from elastic_nerf.nerfacc.radiance_fields.mlp import ElasticMLP, ElasticMLPConfig
from elastic_nerf.nerfacc.radiance_fields.ngp import ElasticMLPWithInputEncoding

# batch_size = 128
# features_in = 16
# features_out = 1
# hidden_dim = 32

# # torch net, notice that there should be no bias, because there is no in the tcnn implementation
# mlp_torch = nn.Sequential(
#     nn.Linear(features_in, hidden_dim, bias=False),
#     nn.ReLU(),
#     nn.Linear(hidden_dim, hidden_dim, bias=False),
#     nn.ReLU(),
#     nn.Linear(hidden_dim, features_out, bias=False),
# ).cuda()

# # same net, but in tcnn, should be faster on big batches
# mlp_tcnn = tcnn.Network(
#     n_input_dims=features_in,
#     n_output_dims=features_out,
#     network_config={
#         "otype": "FullyFusedMLP",
#         "activation": "ReLU",
#         "output_activation": "None",
#         "n_neurons": hidden_dim,
#         "n_hidden_layers": 2,
#     },
# )

# input = torch.randn(batch_size, features_in).cuda()

# # but the initialization is obviously different
# output_torch = mlp_torch(input)
# output_tcnn = mlp_tcnn(input)
# print(torch.allclose(output_torch, output_tcnn.float(), rtol=0.01, atol=0.01))  # False

# # in tcnn output layer's width always should be the multiple of 16, so need to pad last layer's params here
# output_layer = mlp_torch[4].weight.data
# output_layer = nn.functional.pad(output_layer, pad=(0, 0, 0, 16 - (features_out % 16)))

# # concatenate all flatten parameters
# params = torch.cat(
#     [
#         mlp_torch[0].weight.data.flatten(),
#         mlp_torch[2].weight.data.flatten(),
#         output_layer.flatten(),
#     ]
# ).half()

# # assign their values to the tcnn net
# mlp_tcnn.params.data[...] = params

# # now both nets are the same (but there could be little differences due to half float usage)
# output_torch = mlp_torch(input)
# output_tcnn = mlp_tcnn(input)
# print(torch.allclose(output_torch, output_tcnn.float(), rtol=0.01, atol=0.01))  # True


# %%
def compare_outputs(elastic_torch, elastic_tcnn, input):
    input = input.cuda().half()
    elastic_torch = elastic_torch.half().cuda()
    elastic_tcnn = elastic_tcnn.half().cuda()
    output_torch = elastic_torch(input)
    output_tcnn = elastic_tcnn(input)
    return torch.allclose(output_torch, output_tcnn, rtol=0.01, atol=0.01)


def make_input(input_dim, batch_size=128):
    return torch.randn(batch_size, input_dim).cuda()


def pad_output_to_16(tensor):
    output_dim = tensor.shape[0]
    pad_size = (16 - (output_dim % 16)) % 16
    return nn.functional.pad(tensor, pad=(0, 0, 0, pad_size))


def pad_input_to_16(tensor):
    output_dim = tensor.shape[1]
    pad_size = (16 - (output_dim % 16)) % 16
    return nn.functional.pad(tensor, pad=(0, pad_size, 0, 0), value=1)


# input_dim = 16
# output_dim = 1
# batch_size = 128
# encoding_input_dim = 3


def make_network_config(net_width=64, net_depth=1):
    ff_supported_widths = [16, 32, 64, 128]
    otype = "FullyFusedMLP" if net_width in ff_supported_widths else "CutlassMLP"
    network_config = {
        "otype": otype,
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": net_width,
        "n_hidden_layers": net_depth,
    }
    return network_config


def make_encoding_config(
    n_levels=5,
    n_features_per_level=2,
    log2_hashmap_size=17,
    base_resolution=16,
    per_level_scale=1.681792830507429,
):
    encoding_config = {
        "otype": "HashGrid",
        "n_levels": n_levels,
        "n_features_per_level": n_features_per_level,
        "log2_hashmap_size": log2_hashmap_size,
        "base_resolution": base_resolution,
        "per_level_scale": per_level_scale,
    }
    return encoding_config


# %%
# Test Encoder


# %% Test ElasticMLP
def test_network(input_dim, output_dim, net_depth=1, net_width=64):
    # Create TCNN network
    network_config = make_network_config()
    elastic_tcnn = tcnn.Network(
        n_input_dims=input_dim, n_output_dims=output_dim, network_config=network_config
    )

    # Create ElasticMLP (un-aligned pytorch)
    elastic_mlp_config = ElasticMLPConfig(output_activation=None, bias_enabled=False)
    elastic_torch: ElasticMLP = elastic_mlp_config.setup(
        input_dim=input_dim,
        output_dim=output_dim,
        net_depth=net_depth,
        net_width=net_width,
        skip_layer=None,
        output_enabled=True,
    )

    input = make_input(input_dim)
    assert not compare_outputs(elastic_torch, elastic_tcnn, input)

    # Now try to figure out how we can make it work by aligning inputs and outputs
    input_dim_padded = (input_dim + 15) // 16 * 16
    elastic_torch: ElasticMLP = elastic_mlp_config.setup(
        input_dim=input_dim_padded,
        output_dim=output_dim,
        net_depth=net_depth,
        net_width=net_width,
        skip_layer=None,
        output_enabled=True,
    )
    output_layer_padded = pad_output_to_16(elastic_torch.output_layer.weight.data)

    # Make standard input and then pad
    input = make_input(input_dim).cuda().half()
    # For inputs, pad along the feature dimension (not batch)
    input_padded = pad_input_to_16(input)
    params = torch.cat(
        [
            elastic_torch.hidden_layers[0].weight.data.flatten(),
            output_layer_padded.flatten(),
        ]
    ).half()
    elastic_tcnn.params.data[...] = params
    elastic_tcnn = elastic_tcnn.half().cuda()
    elastic_torch = elastic_torch.half().cuda()
    output_torch = elastic_torch(input_padded)
    output_tcnn = elastic_tcnn(input)
    assert torch.allclose(output_torch, output_tcnn, rtol=0.01, atol=0.01)


test_network(input_dim=16, output_dim=1)
test_network(input_dim=10, output_dim=1)
# %%
elastic_torch = ElasticMLPWithInputEncoding(
    input_dim=encoding_input_dim,
    output_dim=output_dim,
    encoding_config=encoding_config,
    elastic_mlp=elastic_mlp_config,
    pad_value=1,
    align_inputs=True,
    align_outputs=False,
).cuda()

elastic_tcnn = tcnn.NetworkWithInputEncoding(
    n_input_dims=encoding_input_dim,
    n_output_dims=output_dim,
    encoding_config=encoding_config,
    network_config=network_config,
)
input = make_input(encoding_input_dim)
compare_outputs(elastic_torch, elastic_tcnn, input)


# %%
output_layer_padded = pad_tensor_to_16(
    elastic_torch.elastic_mlp.output_layer.weight.data, dim=0
)
# pad_size = ((output_dim + 15) // 16 * 16) - output_dim
# output_layer_padded = nn.functional.pad(output_layer, pad=(0, pad_size))
# output_layer_padded = nn.functional.pad(output_layer, pad=(0, 0, 0, pad_size))
# output_layer_padded = nn.functional.pad(
#     output_layer, pad=(0, 0, 0, (16 - (output_dim % 16)) % 16)
# )

# concatenate all flatten parameters
params = torch.cat(
    [
        elastic_torch.encoding.params.data.flatten(),
        elastic_torch.elastic_mlp.hidden_layers[0].weight.data.flatten(),
        output_layer_padded.flatten(),
    ]
).half()

elastic_tcnn.params.data[...] = params
compare_outputs(elastic_torch, elastic_tcnn, input)
# output_torch = elastic_torch(input)
# output_tcnn = elastic_tcnn(input)
# print(torch.allclose(output_torch, output_tcnn, rtol=0.01, atol=0.01))  # False

# %%
