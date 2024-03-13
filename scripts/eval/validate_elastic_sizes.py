# %%
from pathlib import Path
from elastic_nerf.nerfacc.trainers.ngp_prop import NGPPropTrainer
from training_dynamics_analyzer import SweepDynamicsPlotter as sdp
import numpy as np
import tinycudann as tcnn
import torch
from elastic_nerf.nerfacc.radiance_fields.ngp import pack_weights

# %%
run_id = "kx5mcllz"
results_cache_dir = Path("/home/user/shared/results/elastic-nerf")
log_dir = Path("/home/user/shared/results/elastic-nerf") / run_id
wandb_dir = Path("/home/user/shared/wandb_cache/elastic-nerf") / run_id
config = sdp.get_config(run_id, results_cache_dir)
checkpoints = sdp.get_checkpoints(run_id, results_cache_dir)

trainer = NGPPropTrainer.load_trainer(config, log_dir=log_dir, wandb_dir=wandb_dir)
trainer.config.model_path = checkpoints[20000]
trainer.load_checkpoint()

# %%
# model = trainer.radiance_field
model = trainer.proposal_networks[0]
print(pack_weights(model.mlp_base).numel())
model.load_elastic_width(64, trainer.step)
print(pack_weights(model.mlp_base).numel())
model.load_full_width(trainer.device)
print(pack_weights(model.mlp_base).numel())
print(model.full_widths)
model.load_elastic_width(8, trainer.step)
print(pack_weights(model.mlp_base).numel())
print(model.full_widths)
model.load_full_width(trainer.device)
print(pack_weights(model.mlp_base).numel())
print(model.full_widths)

# %%
width = 16
model.load_elastic_width(width, trainer.step, trainer.device)
ff_mlp_base = model.make_fused_base(width)
# Extract and pack the weights from the ElasticMLP
packed_weights = pack_weights(model.mlp_base)
#%%
print(ff_mlp_base)
print(f"Elastic packed: {packed_weights.numel()}")
print(f"Fused: {ff_mlp_base.params.numel()}")

# %%
width = 8
trainer.radiance_field.load_elastic_width(width, trainer.step)
model = trainer.radiance_field
sum = 0
for name, param in model.mlp_base.named_parameters():
    print(name, param.shape)
    sum += np.prod(param.shape)
print(f"ElasticMLPWithInputEncoding has a total of {sum} params")
# %%
trainer.load_width(width)
trainer.load_full_width()

# %%
# Evaluate the model with different widths
# Width 8
for i in [8, 16, 32, 64]:
    print("****************************************************")
    print(f"Evaluating model with width {i}...")
    trainer.eval([i])
    for name, module in trainer.models_to_watch.items():
        print(name)
        for n, p in module.named_parameters():
            print(n, p.shape)


# %%
width = 8
trainer.load_elastic_width(width)
trainer.radiance_field.load_fully_fused(width, trainer.step)
# model = trainer.radiance_field
# per_level_scale = np.exp(
#     (np.log(model.max_resolution) - np.log(model.base_resolution))
#     / (model.n_levels - 1)
# ).tolist()
# encoding_config = {
#     "otype": "HashGrid",
#     "n_levels": model.n_levels,
#     "n_features_per_level": 2,
#     "log2_hashmap_size": model.log2_hashmap_size,
#     "base_resolution": model.base_resolution,
#     "per_level_scale": per_level_scale,
# }
# num_dim = 3
# otype = "CutlassMLP" if width == 8 else "FullyFusedMLP"
# ff_mlp = tcnn.NetworkWithInputEncoding(
#     n_input_dims=num_dim,
#     n_output_dims=1 + model.geo_feat_dim,
#     encoding_config=encoding_config,
#     network_config={
#         "otype": otype,
#         "activation": "ReLU",
#         "output_activation": "None",
#         "n_neurons": width,
#         "n_hidden_layers": 1,
#     },
# )


# %%
def pack_weights(model):
    # List to hold all the weights and biases
    all_weights = []

    # Loop through each layer and extract weights and biases
    for name, param in model.named_parameters():
        # Flatten then add to the list
        all_weights.append(param.data.cpu().flatten())

    # Concatenate all the weights and biases into a single buffer
    packed_weights = torch.concatenate(all_weights)

    return packed_weights


# %%
# Compute the total number of parameters in the elastic model
sum = 0
for name, param in model.mlp_base.named_parameters():
    print(name, param.shape)
    sum += np.prod(param.shape)
print(f"ElasticMLPWithInputEncoding has a total of {sum} params")
print(ff_mlp)
print(
    f"Total number of parameters in the fully fused MLP model: {np.prod(ff_mlp.params.shape)}"
)
print(f"Difference in number of parameters: {np.prod(ff_mlp.params.shape) - sum}")

# %%
# Extract and pack the weights from the ElasticMLP
packed_weights = pack_weights(model.mlp_base)

# Ensure the buffer size matches the total parameters of the tcnn model
assert (
    packed_weights.size == ff_mlp.params.numel()
), "Mismatch in the number of parameters!"

# Initialize the tcnn model with the packed weights
# We need to ensure the dtype and device match before assignment
ff_mlp.params.data = packed_weights.to(
    dtype=ff_mlp.params.dtype, device=ff_mlp.params.device
)

# %%
