import torch
import triton
import triton.language as tl
from elastic_nerf.nerfacc.radiance_fields.mlp import ElasticMLP


@triton.jit
def relu(x):
    return tl.where(x > 0, x, 0)


@triton.jit
def fully_fused_elastic_mlp_forward(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    hidden_layer_size: tl.constexpr,
    active_neurons_ptr,
    num_layers: tl.constexpr,
    batch_size,
    input_dim,
    output_dim,
    bias_enabled,
    skip_connection,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_blocks = batch_size // BLOCK_SIZE

    act_shmem = tl.zeros((BLOCK_SIZE, hidden_layer_size), dtype=tl.float32)
    active_neurons_shmem = tl.zeros((num_layers,), dtype=tl.int32)

    for i in range(num_layers):
        active_neurons_shmem[i] = active_neurons_ptr[i]

    for i in range(BLOCK_SIZE):
        for j in range(input_dim):
            act_shmem[i, j] = x_ptr[pid * BLOCK_SIZE + i, j]

    for layer in range(num_layers):
        active_neurons = active_neurons_shmem[layer]
        in_features = input_dim if layer == 0 else active_neurons_shmem[layer - 1]

        w_offset = layer * hidden_layer_size * in_features
        b_offset = layer * hidden_layer_size
        w_slice = w_ptr + w_offset
        b_slice = b_ptr + b_offset

        for i in range(BLOCK_SIZE):
            for j in range(active_neurons):
                val = tl.float32(0)
                for k in range(in_features):
                    val += act_shmem[i, k] * tl.load(w_slice + j * in_features + k)
                if bias_enabled:
                    val += tl.load(b_slice + j)
                act_shmem[i, j] = relu(val)

            if skip_connection and layer > 0:
                for j in range(input_dim):
                    act_shmem[i, j] += x_ptr[pid * BLOCK_SIZE + i, j]

    for i in range(BLOCK_SIZE):
        for j in range(output_dim):
            y_ptr[pid * BLOCK_SIZE + i, j] = act_shmem[i, j]


@triton.jit
def fully_fused_elastic_mlp_backward(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    dy_ptr,
    dw_ptr,
    db_ptr,
    dx_ptr,
    hidden_layer_size: tl.constexpr,
    active_neurons_ptr,
    num_layers: tl.constexpr,
    batch_size,
    input_dim,
    output_dim,
    activation_grad,
    bias_enabled,
    skip_connection,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_blocks = batch_size // BLOCK_SIZE

    act_shmem = tl.zeros((BLOCK_SIZE, hidden_layer_size), dtype=tl.float32)
    grad_shmem = tl.zeros((BLOCK_SIZE, hidden_layer_size), dtype=tl.float32)
    active_neurons_shmem = tl.zeros((num_layers,), dtype=tl.int32)

    for i in range(num_layers):
        active_neurons_shmem[i] = active_neurons_ptr[i]

    for i in range(BLOCK_SIZE):
        for j in range(output_dim):
            grad_shmem[i, j] = dy_ptr[pid * BLOCK_SIZE + i, j]

    for layer in reversed(range(num_layers)):
        active_neurons = active_neurons_shmem[layer]
        in_features = input_dim if layer == 0 else active_neurons_shmem[layer - 1]

        w_offset = layer * hidden_layer_size * in_features
        b_offset = layer * hidden_layer_size
        dw_offset = w_offset
        db_offset = b_offset
        w_slice = w_ptr + w_offset
        b_slice = b_ptr + b_offset
        dw_slice = dw_ptr + dw_offset
        db_slice = db_ptr + db_offset

        for i in range(BLOCK_SIZE):
            for j in range(active_neurons):
                grad = grad_shmem[i, j]
                if layer < num_layers - 1:
                    grad *= activation_grad(act_shmem[i, j])
                for k in range(in_features):
                    tl.atomic_add(
                        dw_slice + j * in_features + k, grad * act_shmem[i, k]
                    )
                if bias_enabled:
                    tl.atomic_add(db_slice + j, grad)

        if layer > 0:
            for i in range(BLOCK_SIZE):
                for j in range(in_features):
                    val = tl.float32(0)
                    for k in range(active_neurons):
                        val += grad_shmem[i, k] * tl.load(w_slice + k * in_features + j)
                    grad_shmem[i, j] = val

                    if skip_connection:
                        grad_shmem[i, j] += dy_ptr[pid * BLOCK_SIZE + i, j]

    if dx_ptr is not None:
        for i in range(BLOCK_SIZE):
            for j in range(input_dim):
                dx_ptr[pid * BLOCK_SIZE + i, j] = grad_shmem[i, j]


class FullyFusedElasticMLP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        w,
        b,
        hidden_layer_size: tl.constexpr,
        active_neurons,
        bias_enabled,
        skip_connection=False,
    ):
        y = torch.empty_like(x)
        active_neurons_ptr = active_neurons.data_ptr()
        num_layers = active_neurons.shape[0]
        batch_size, input_dim = x.shape
        output_dim = active_neurons[-1]

        grid = lambda meta: (triton.cdiv(batch_size, meta["BLOCK_SIZE"]),)
        fully_fused_elastic_mlp_forward[grid](
            x,
            w,
            b,
            y,
            hidden_layer_size,
            active_neurons_ptr,
            num_layers,
            batch_size,
            input_dim,
            output_dim,
            bias_enabled,
            skip_connection,
            BLOCK_SIZE=1024,
        )

        ctx.save_for_backward(x, w, b, active_neurons)
        ctx.activation_grad = activation.grad
        ctx.bias_enabled = bias_enabled
        ctx.skip_connection = skip_connection
        ctx.hidden_layer_size = hidden_layer_size
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, active_neurons = ctx.saved_tensors
        dx, dw, db = None, None, None

        if ctx.needs_input_grad[0]:
            dx = torch.empty_like(x)
        if ctx.needs_input_grad[1]:
            dw = torch.empty_like(w)
        if ctx.needs_input_grad[2]:
            db = torch.empty_like(b)

        active_neurons_ptr = active_neurons.data_ptr()
        num_layers = active_neurons.shape[0]
        batch_size, input_dim = x.shape
        output_dim = active_neurons[-1]

        grid = lambda meta: (triton.cdiv(batch_size, meta["BLOCK_SIZE"]),)
        fully_fused_elastic_mlp_backward[grid](
            x,
            w,
            b,
            y,
            dy,
            dw,
            db,
            dx,
            ctx.hidden_layer_size,
            active_neurons_ptr,
            num_layers,
            batch_size,
            input_dim,
            output_dim,
            ctx.activation_grad,
            ctx.bias_enabled,
            ctx.skip_connection,
            BLOCK_SIZE=1024,
        )

        return dx, dw, db, None, None, None, None


class ElasticMLPTriton(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        num_layers,
        hidden_layer_size: tl.constexpr,
        bias_enabled=True,
        skip_connection=False,
    ):
        super().__init__()
        self.bias_enabled = bias_enabled
        self.skip_connection = skip_connection
        self.hidden_layer_size = hidden_layer_size

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.w = torch.nn.Parameter(
            torch.randn(num_layers * hidden_layer_size * input_dim)
        )
        if bias_enabled:
            self.b = torch.nn.Parameter(torch.randn(num_layers * hidden_layer_size))
        else:
            self.b = None

    def forward(self, x, active_neurons=None):
        if active_neurons is None:
            active_neurons = torch.full(
                (self.num_layers,), self.hidden_layer_size, dtype=torch.int32
            )
        else:
            active_neurons = torch.tensor(active_neurons, dtype=torch.int32)

        return FullyFusedElasticMLP.apply(
            x,
            self.w,
            self.b,
            self.hidden_layer_size,
            active_neurons,
            self.bias_enabled,
            self.skip_connection,
        )


# Testing and Benchmarking Suite

import time
import numpy as np


def benchmark_triton(model, x, active_neurons=None, num_runs=100):
    model.cuda()
    x = x.cuda()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(10):
            model(x, active_neurons)

        start_event.record()
        for _ in range(num_runs):
            model(x, active_neurons)
        end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / num_runs

    return elapsed_time


def benchmark_pytorch(model, x, active_neurons=None, num_runs=100):
    model.cuda()
    x = x.cuda()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(10):
            model(x, active_neurons)

        start_event.record()
        for _ in range(num_runs):
            model(x, active_neurons)
        end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / num_runs

    return elapsed_time


def test_correctness(
    triton_model, pytorch_model, x, active_neurons=None, atol=1e-6, rtol=1e-6
):
    triton_model.cuda()
    pytorch_model.cuda()
    x = x.cuda()

    with torch.no_grad():
        triton_output = triton_model(x, active_neurons)
        pytorch_output = pytorch_model(x, active_neurons)

    if not torch.allclose(triton_output, pytorch_output, atol=atol, rtol=rtol):
        raise AssertionError("Triton and PyTorch model outputs do not match!")


def benchmark_and_test(
    triton_model, pytorch_model, x, active_neurons_list=None, num_runs=100
):
    if active_neurons_list is None:
        active_neurons_list = [None]

    for active_neurons in active_neurons_list:
        print(f"Active Neurons: {active_neurons}")

        # Correctness Test
        test_correctness(triton_model, pytorch_model, x, active_neurons)
        print("Correctness test passed!")

        # Triton Benchmark
        triton_time = benchmark_triton(triton_model, x, active_neurons, num_runs)
        print(f"Triton Elapsed Time: {triton_time:.6f} ms")

        # PyTorch Benchmark
        pytorch_time = benchmark_pytorch(pytorch_model, x, active_neurons, num_runs)
        print(f"PyTorch Elapsed Time: {pytorch_time:.6f} ms")

        # Speedup
        speedup = pytorch_time / triton_time
        print(f"Speedup: {speedup:.2f}x")

        print()


if __name__ == "__main__":
    input_dim = 512
    output_dim = 256
    num_layers: tl.constexpr = 4
    hidden_layer_size = 1024
    batch_size = 4096

    triton_model = ElasticMLPTriton(
        input_dim,
        output_dim,
        num_layers,
        hidden_layer_size=1024,
        bias_enabled=True,
        skip_connection=False,
    )
    pytorch_model = ElasticMLP(
        input_dim,
        output_dim,
        num_layers,
        hidden_layer_size,
        skip_layer=None,
        bias_enabled=True,
    )

    x = torch.randn(batch_size, input_dim)

    active_neurons_list = [
        None,
        [1024, 1024, 1024, 1024],
        [512, 512, 512, 512],
        [256, 256, 256, 256],
    ]

    benchmark_and_test(
        triton_model, pytorch_model, x, active_neurons_list, num_runs=100
    )
