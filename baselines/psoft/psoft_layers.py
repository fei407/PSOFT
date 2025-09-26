import types
from tqdm import tqdm
from peft.utils import _get_submodules
import peft

import torch
import torch.nn as nn
import torch.nn.functional as F

class PSOFTLayer(nn.Module):
    def __init__(self, size, orth=True, mag_b=True, mag_a=True, use_cayley_neumann=True, num_cayley_neumann_terms=5):

        super(PSOFTLayer, self).__init__()
        self.size = size
        self.orth = orth
        self.mag_b = mag_b
        self.mag_a = mag_a
        self.use_cayley_neumann = use_cayley_neumann
        self.num_cayley_neumann_terms = num_cayley_neumann_terms

        if self.orth:
            self.a_params = nn.Parameter(torch.zeros((size * (size - 1)) // 2))
            self.register_buffer('indices_lower', torch.tril_indices(size, size, -1))
        else:
            self.a_params = nn.Parameter(torch.eye(size))

        if self.mag_b:
            self.mag_vector_b = nn.Parameter(torch.ones(size))  # Trainable vector initialized to 1
        else:
            self.mag_vector_b = None

        if self.mag_a:
            self.mag_vector_a = nn.Parameter(torch.ones(size))  # Trainable vector initialized to 1
        else:
            self.mag_vector_a = None

    def forward(self, input):
        if self.orth:
            A = torch.zeros((self.size, self.size), device=self.a_params.device)
            A[self.indices_lower[0], self.indices_lower[1]] = self.a_params
            A = A - A.t()
            I = torch.eye(self.size, device=self.a_params.device)
            if self.use_cayley_neumann:
                t = self.num_cayley_neumann_terms
                if t <= 0:
                    Q = I - A
                else:
                    negA = -A
                    R = I.clone()
                    P = negA
                    for _ in range(t):
                        R.add_(P, alpha=2.0)
                        P = P @ negA
                    R.add_(P)
                    Q = R
            else:
                Q = torch.linalg.solve(I + A, I - A, left=False)
        else:
            Q = self.a_params

        if self.mag_b and self.mag_a:
            mag_diag_b = torch.diag(self.mag_vector_b)
            mag_diag_a = torch.diag(self.mag_vector_a)
            Q = mag_diag_b @ Q @ mag_diag_a
        elif self.mag_b:
            mag_diag_b = torch.diag(self.mag_vector_b)
            Q = mag_diag_b @ Q
        elif self.mag_a:
            mag_diag_a = torch.diag(self.mag_vector_a)
            Q = Q @ mag_diag_a

        return torch.matmul(input, Q.t())

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

def get_delta_weight(self, adapter) -> torch.Tensor:
    # This function is introduced in newer PEFT versions. we modify this function instead of modifying
    # the merge function (as we did previously for version 0.4.0 of PEFT).
    """
    Compute the delta weight for the given adapter.

    Args:
        adapter (str):
            The name of the adapter for which the delta weight should be computed.
    """
    device = self.lora_B[adapter].weight.device
    dtype = self.lora_B[adapter].weight.dtype

    # In case users wants to merge the adapter weights that are in
    # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
    cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

    weight_A = self.lora_A[adapter].weight
    weight_B = self.lora_B[adapter].weight

    if cast_to_fp32:
        weight_A = weight_A.float()
        weight_B = weight_B.float()

    size = self.lora_orth[adapter].size
    a_params = self.lora_orth[adapter].a_params
    mag_b = self.lora_orth[adapter].mag_b
    mag_a = self.lora_orth[adapter].mag_a
    mag_vector_b = self.lora_orth[adapter].mag_vector_b
    mag_vector_a = self.lora_orth[adapter].mag_vector_a

    if self.lora_orth[adapter].orth:
        indices_lower = self.lora_orth[adapter].indices_lower
        A = torch.zeros((size, size), device=a_params.device)
        A[indices_lower[0], indices_lower[1]] = a_params
        A = A - A.t()
        I = torch.eye(size, device=a_params.device)
        if self.lora_orth[adapter].use_cayley_neumann:
            t = self.lora_orth[adapter].num_cayley_neumann_terms
            if t <= 0:
                Q = I - A
            else:
                negA = -A
                R = I.clone()
                P = negA
                for _ in range(t):
                    R.add_(P, alpha=2.0)
                    P = P @ negA
                R.add_(P)
                Q = R
        else:
            Q = torch.linalg.solve(I + A, I - A, left=False)
    else:
        Q = a_params

    if mag_b and mag_a:
        mag_diag_b = torch.diag(mag_vector_b)
        mag_diag_a = torch.diag(mag_vector_a)
        Q = mag_diag_b @ Q @ mag_diag_a
    elif mag_b:
        mag_diag_b = torch.diag(mag_vector_b)
        Q = mag_diag_b @ Q
    elif mag_a:
        mag_diag_a = torch.diag(mag_vector_a)
        Q = Q @ mag_diag_a

    output_tensor = transpose(
        weight_B @ Q @ weight_A,
        self.fan_in_fan_out
    ) * self.scaling[adapter]

    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)

        # cast back the weights
        self.lora_A[adapter].weight.data = weight_A.to(dtype)
        self.lora_B[adapter].weight.data = weight_B.to(dtype)

    return output_tensor

def forward_psoft(self, x: torch.Tensor):
    previous_dtype = x.dtype

    if self.active_adapter[0] not in self.lora_A.keys():
        return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    if self.disable_adapters:
        if self.r[self.active_adapter[0]] > 0 and self.merged:
            self.unmerge()
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    elif self.r[self.active_adapter[0]] > 0 and not self.merged:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        x = x.to(self.lora_A[self.active_adapter[0]].weight.dtype)

        # adding lora_orth in the forward loop
        xa = self.lora_A[self.active_adapter[0]](self.lora_dropout[self.active_adapter[0]](x))
        xq = self.lora_orth[self.active_adapter[0]](xa)
        xb = self.lora_B[self.active_adapter[0]](xq)

        result += xb * self.scaling[self.active_adapter[0]]

    else:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    result = result.to(previous_dtype)

    return result

def create_and_insert(model, peft_config, orth=True, mag_out=False, mag_b=True, mag_a=True, use_cayley_neumann=True, num_cayley_neumann_terms=5):
    """
    Insert PSOFT layer

    :param model: 'pissa' initialization
    :param peft_config: PEFT configuration
    """
    is_target_modules_in_base_model = False
    key_list = [key for key, _ in model.named_modules()]
    assert (not isinstance(peft_config.target_modules, str))
    print("Inserting the psoft module between B and A matrices.")
    for key in tqdm(key_list):
        target_module_found = any(key.endswith(target_key) for target_key in peft_config.target_modules)
        if target_module_found:
            if not is_target_modules_in_base_model:
                is_target_modules_in_base_model = True
            _, target, target_name = _get_submodules(model, key)
            if not isinstance(target, peft.tuners.lora.Linear):
                raise NotImplementedError('Only initialization for peft.tuners.lora.Linear type is implemented.')
            else:
                target.forward = types.MethodType(forward_psoft, target)
                target.get_delta_weight = types.MethodType(get_delta_weight, target)
                target.lora_orth = torch.nn.ModuleDict({
                    "default": PSOFTLayer(peft_config.r, orth=orth, mag_b=mag_b, mag_a=mag_a, use_cayley_neumann=use_cayley_neumann, num_cayley_neumann_terms=num_cayley_neumann_terms)
                })
                target.lora_A.default.weight.requires_grad = False
                target.lora_B.default.weight.requires_grad = False

    if not is_target_modules_in_base_model:
        raise ValueError(
            f"Target modules {peft_config.target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )