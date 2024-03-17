# %%
from numpy import block
import torch
import tinycudann as tcnn
from torch import nn
from elastic_nerf.nerfacc.radiance_fields.mlp import ElasticMLP, ElasticMLPConfig
from elastic_nerf.nerfacc.radiance_fields.ngp import ElasticMLPWithInputEncoding
from elastic_nerf.nerfacc.utils import set_random_seed

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


def compare_outputs(elastic_torch, elastic_tcnn, input):
    input = input.cuda().half()
    elastic_torch = elastic_torch.half().cuda()
    elastic_tcnn = elastic_tcnn.half().cuda()
    output_torch = elastic_torch(input)
    output_tcnn = elastic_tcnn(input)
    return torch.allclose(output_torch, output_tcnn, rtol=0.01, atol=0.01)


def init_params(model):
    set_random_seed(42)
    torch.cuda.seed_all()
    for p in model.parameters():
        nn.init.normal_(p, mean=0, std=1)


def make_input(input_dim, batch_size=128):
    return torch.randn(batch_size, input_dim).cuda()


def pad_input_to_16(tensor, value=0, block_size=16):
    output_dim = tensor.shape[1]
    pad_size = (block_size - (output_dim % block_size)) % block_size
    # return nn.functional.pad(tensor, pad=(0, pad_size, 0, 0), value=value)
    return nn.functional.pad(tensor, pad=(0, pad_size), value=value)


def pad_output_to_16(tensor, value=0, block_size=16):
    output_dim = tensor.shape[0]
    pad_size = (block_size - (output_dim % block_size)) % block_size
    return nn.functional.pad(tensor, pad=(0, 0, 0, pad_size), value=value)


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


# Test ElasticMLP
# def test_network(input_dim, output_dim, net_depth=1, net_width=64):
#     # Create TCNN network
#     network_config = make_network_config()
#     elastic_tcnn = tcnn.Network(
#         n_input_dims=input_dim, n_output_dims=output_dim, network_config=network_config
#     )

#     # Create ElasticMLP (un-aligned pytorch)
#     elastic_mlp_config = ElasticMLPConfig(output_activation=None, bias_enabled=False)
#     elastic_torch: ElasticMLP = elastic_mlp_config.setup(
#         input_dim=input_dim,
#         output_dim=output_dim,
#         net_depth=net_depth,
#         net_width=net_width,
#         skip_layer=None,
#         output_enabled=True,
#     )

#     input = make_input(input_dim)
#     assert not compare_outputs(elastic_torch, elastic_tcnn, input)

#     # Now try to figure out how we can make it work by aligning inputs and outputs
#     input_dim_padded = (input_dim + 15) // 16 * 16
#     elastic_torch: ElasticMLP = elastic_mlp_config.setup(
#         input_dim=input_dim_padded,
#         output_dim=output_dim,
#         net_depth=net_depth,
#         net_width=net_width,
#         skip_layer=None,
#         output_enabled=True,
#     )
#     output_layer_padded = pad_output_to_16(elastic_torch.output_layer.weight.data)

#     # Make standard input and then pad
#     input = make_input(input_dim).cuda().half()
#     # For inputs, pad along the feature dimension (not batch)
#     input_padded = pad_input_to_16(input)
#     params = torch.cat(
#         [
#             elastic_torch.hidden_layers[0].weight.data.flatten(),
#             output_layer_padded.flatten(),
#         ]
#     ).half()
#     elastic_tcnn.params.data[...] = params
#     elastic_tcnn = elastic_tcnn.half().cuda()
#     elastic_torch = elastic_torch.half().cuda()
#     output_torch = elastic_torch(input_padded)
#     output_tcnn = elastic_tcnn(input)
#     assert torch.allclose(output_torch, output_tcnn, rtol=0.01, atol=0.01)


# test_network(input_dim=16, output_dim=1)
# test_network(input_dim=10, output_dim=1)

input_dim = 3
output_dim = 1
mlp_out_dim = 1
mlp_in_dim = 10
net_width = 64
net_depth = 1

# Create encoder
encoding_config = make_encoding_config()
tcnn_encoder = tcnn.Encoding(input_dim, encoding_config)
init_params(tcnn_encoder)

# Create networks
# Torch MLP
input_dim_padded = (mlp_in_dim + 15) // 16 * 16
elastic_mlp_config = ElasticMLPConfig(output_activation=None, bias_enabled=False)
print(f"Input dim padded from {mlp_in_dim} to {input_dim_padded}")
torch_mlp: ElasticMLP = elastic_mlp_config.setup(
    input_dim=mlp_in_dim,
    output_dim=output_dim,
    net_depth=net_depth,
    net_width=net_width,
    skip_layer=None,
    output_enabled=True,
).cuda()
init_params(torch_mlp)
# TCNN MLP
network_config = make_network_config()
tcnn_mlp = tcnn.Network(
    n_input_dims=mlp_in_dim, n_output_dims=output_dim, network_config=network_config
)

# Torch MLP with Encoding
torch_mlp_with_encoding: ElasticMLPWithInputEncoding = ElasticMLPWithInputEncoding(
    input_dim=input_dim,
    output_dim=mlp_out_dim,
    encoding_config=encoding_config,
    elastic_mlp=elastic_mlp_config,
    # align_inputs=True,
).cuda()

# TCNN MLP with Encoding
tcnn_network_with_encoding = tcnn.NetworkWithInputEncoding(
    n_input_dims=input_dim,
    n_output_dims=mlp_out_dim,
    encoding_config=encoding_config,
    network_config=network_config,
)


# Convert all models to cuda and half
tcnn_encoder = tcnn_encoder.half().cuda()
tcnn_mlp = tcnn_mlp.half().cuda()
tcnn_network_with_encoding = tcnn_network_with_encoding.half().cuda()
torch_mlp = torch_mlp.half().cuda()
torch_mlp_with_encoding = torch_mlp_with_encoding.half().cuda()

# Make standard input and then pad
input = make_input(input_dim).cuda().half()
input_clone = input.clone()
encoded_input = tcnn_encoder(input).half()
print(f"Encoded input shape: {encoded_input.shape}")

# TCNN output
input_layer_padded = pad_input_to_16(
    torch_mlp.hidden_layers[0].weight.data, value=0
).half()
output_layer_padded = pad_output_to_16(torch_mlp.output_layer.weight.data)
print(
    f"Padded from {torch_mlp.output_layer.weight.data.shape} to {output_layer_padded.shape}"
)
torch_mlp_packed_weights = torch.cat(
    [
        input_layer_padded.flatten(),
        output_layer_padded.flatten(),
    ]
).cuda()
tcnn_mlp.params.data[...] = torch_mlp_packed_weights
tcnn_mlp_output = tcnn_mlp(encoded_input)

# Torch output
torch_mlp_output = torch_mlp(encoded_input)
print(torch.allclose(torch_mlp_output, tcnn_mlp_output, rtol=0.01, atol=0.01))

# TCNN Network with Encoding output
torch_mlp_with_encoding.encoding.params.data[...] = tcnn_encoder.params.data[...]
torch_mlp_with_encoding.elastic_mlp.hidden_layers[0].weight.data[...] = (
    torch_mlp.hidden_layers[0].weight.data[...]
)
torch_mlp_with_encoding.elastic_mlp.output_layer.weight.data[...] = (
    torch_mlp.output_layer.weight.data[...]
)
# torch_mlp_with_encoding_packed_weights = (
#     torch_mlp_with_encoding.get_packed_weights().cuda().half()
# )

input_layer_padded = nn.functional.pad(
    torch_mlp_with_encoding.elastic_mlp.hidden_layers[0].weight.data.t(),
    pad=(0, 0, 6, 0),
    value=0,
).half()
output_layer_padded = nn.functional.pad(
    torch_mlp_with_encoding.elastic_mlp.output_layer.weight.data.t(),
    pad=(15, 0, 0, 0),
    value=0,
).half()
# input_layer_padded = pad_input_to_16(
#     torch_mlp_with_encoding.elastic_mlp.hidden_layers[0].weight.data, value=0
# ).half()
# output_layer_padded = pad_output_to_16(
#     torch_mlp_with_encoding.elastic_mlp.output_layer.weight.data
# )

torch_mlp_with_encoding_packed_weights_manual = torch.cat(
    [
        torch_mlp_with_encoding.encoding.params.data.flatten(),
        input_layer_padded.data.flatten(),
        output_layer_padded.data.flatten(),
    ]
).cuda()
tcnn_network_with_encoding.params.data[...] = (
    torch_mlp_with_encoding_packed_weights_manual
)
tcnn_mlp_with_encoding_output = tcnn_network_with_encoding(input_clone)

# Torch MLP with Encoding output
torch_mlp_with_encoding_output = torch_mlp_with_encoding(input_clone)
print(
    torch.allclose(
        torch_mlp_with_encoding_output,
        torch_mlp_output,
        rtol=0.01,
        atol=0.01,
    )
)
print(
    torch.allclose(
        torch_mlp_with_encoding_output,
        tcnn_mlp_with_encoding_output,
        rtol=0.01,
        atol=0.01,
    )
)
print(tcnn_mlp_with_encoding_output[:10])
print(torch_mlp_with_encoding_output[:10])
# %%

# TCNN Network with Encoding output
torch_mlp_with_encoding_packed_weights = (
    torch_mlp_with_encoding.get_packed_weights().cuda()
)
# tcnn_network_with_encoding.params.data[...] = torch_mlp_with_encoding_packed_weights
# tcnn_mlp_with_encoding_output = tcnn_network_with_encoding(input_clone)

# Torch MLP with Encoding output
torch_mlp_with_encoding_output = torch_mlp_with_encoding(input)
# Compare outputs

assert torch.allclose(
    torch_mlp_output, tcnn_mlp_with_encoding_output_manual, rtol=0.01, atol=0.01
)
assert torch.allclose(
    torch_mlp_with_encoding_output,
    tcnn_mlp_with_encoding_output_manual,
    rtol=0.01,
    atol=0.01,
)
# assert torch.allclose(
#     torch_mlp_with_encoding_output,
#     tcnn_mlp_with_encoding_output,
#     rtol=0.01,
#     atol=0.01,
# )


# %%
# Test NetworkWithInputEncoding / ElasticMLPWithInputEncoding
def test_network_with_input_encoding(input_dim, mlp_out_dim, net_depth=1, net_width=64):
    # Create TCNN network
    network_config = make_network_config()
    encoding_config = make_encoding_config()
    elastic_tcnn = tcnn.NetworkWithInputEncoding(
        n_input_dims=input_dim,
        n_output_dims=mlp_out_dim,
        encoding_config=encoding_config,
        network_config=network_config,
    )

    # Create ElasticMLP (un-aligned pytorch)
    elastic_mlp_config = ElasticMLPConfig(output_activation=None, bias_enabled=False)
    elastic_torch: ElasticMLPWithInputEncoding = ElasticMLPWithInputEncoding(
        input_dim=input_dim,
        output_dim=mlp_out_dim,
        encoding_config=encoding_config,
        elastic_mlp=elastic_mlp_config,
        align_inputs=True,
    ).cuda()

    tcnn_params = elastic_torch.get_packed_weights().half()
    elastic_tcnn.params.data[...] = tcnn_params

    input = make_input(input_dim)
    assert compare_outputs(elastic_torch, elastic_tcnn, input)

    # # Now try to figure out how we can make it work by aligning inputs and outputs
    # input_dim_padded = (input_dim + 15) // 16 * 16
    # elastic_torch: ElasticMLPWithInputEncoding = ElasticMLPWithInputEncoding(
    #     input_dim=input_dim_padded,
    #     output_dim=output_dim,
    #     encoding_config=encoding_config,
    #     elastic_mlp=elastic_mlp_config,
    #     pad_value=1,
    #     align_inputs=True,
    #     align_outputs=False,
    # ).cuda()
    # output_layer_padded = pad_output_to_16(
    #     elastic_torch.elastic_mlp.output_layer.weight.data
    # )

    # # Make standard input and then pad
    # input = make_input(input_dim).cuda().half()
    # # For inputs, pad along the feature dimension (not batch)
    # input_padded = pad_input_to_16(input)
    # params = torch.cat(
    #     [
    #         elastic_torch.encoding.params.data.flatten(),
    #         elastic_torch.elastic_mlp.hidden_layers[0].weight.data.flatten(),
    #         output_layer_padded.flatten(),
    #     ]
    # ).half()
    # elastic_tcnn.params.data[...] = params
    # elastic_tcnn = elastic_tcnn.half().cuda()
    # elastic_torch = elastic_torch.half().cuda()
    # output_torch = elastic_torch(input_padded)
    # output_tcnn = elastic_tcnn(input)
    # assert torch.allclose(output_torch, output_tcnn, rtol=0.01, atol=0.01)


test_network_with_input_encoding(input_dim=3, mlp_out_dim=1)
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


# Let's say my input is 3x5 and I want it to be 8-aligned, so I want 3x8 so I pad it with zeros.
# Then, I want the first layer of my network (which is actually 5x8) to be padded to 8x8 with zeros.
# This way, the output of the first layer will be 3x8 but since I padded with zeros, the result should
# be the same.
input_dim = 5
batch_dim = 3
input = torch.randn(batch_dim, input_dim)
print(input.shape)
input_padded = nn.functional.pad(input, pad=(0, 8 - input_dim))
print(input_padded.shape)
first_layer = nn.Linear(input_dim, 8, bias=False)
init_params(first_layer)
print(first_layer.weight.shape)
first_layer_padded = nn.Linear(8, 8, bias=False)
# first_layer_padded_weight = nn.functional.pad(
#     first_layer.weight, pad=(0, 8 - input_dim)
# )
first_layer_padded_weight = pad_input_to_16(first_layer.weight, block_size=8)
first_layer_padded.weight.data[...] = first_layer_padded_weight[...]
print(first_layer_padded.weight.shape)
print(first_layer_padded_weight.shape)
first_out = first_layer(input)
first_out_padded = first_layer_padded(input_padded)
print(first_out.shape)
print(first_out_padded.shape)
print(torch.allclose(first_out, first_out_padded))

output_layer = nn.Linear(8, 1, bias=False)
init_params(output_layer)
print(output_layer.weight.shape)
output_layer_padded = nn.Linear(8, 8, bias=False)
# output_layer_padded_weight = nn.functional.pad(
#     output_layer.weight, pad=(0, 0, 0, 8 - 1)
# )
output_layer_padded_weight = pad_output_to_16(output_layer.weight, block_size=8)
output_layer_padded.weight.data[...] = output_layer_padded_weight[...]
print(output_layer_padded.weight.shape)
print(output_layer_padded_weight.shape)
output = output_layer(first_out)
output_padded = output_layer_padded(first_out_padded)
print(output.shape)
print(output_padded.shape)
print(torch.allclose(output.squeeze(), output_padded[:batch_dim, 0]))

# %%
