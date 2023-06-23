####################################################################################################

# Copyright (c) 2021-, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

####################################################################################################

#
# Note: If when running this conversion script you're getting an exception:
#     ModuleNotFoundError: No module named 'megatron.model.enums'
# you need to tell python where to find the clone of Megatron-LM, e.g.:
#
# cd /tmp
# git clone https://github.com/NVIDIA/Megatron-LM
# PYTHONPATH=/tmp/Megatron-LM python src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py ...
#
# if you already have it cloned elsewhere, simply adjust the path to the existing path
#
# If the training was done using a Megatron-LM fork, e.g.,
# https://github.com/microsoft/Megatron-DeepSpeed/ then chances are that you need to have that one
# in your path, i.e., /path/to/Megatron-DeepSpeed/
#


# This script is adapted from:
# https://github.com/bigcode-project/transformers/blob/main/src/transformers/models/megatron_gpt_bigcode/convert_megatron_gpt_bigcode_checkpoint.py


import sys
sys.path.append("/mnt/home/transformers")

import argparse
import os
import re
from  types import SimpleNamespace

import torch

from transformers import StackT5ForConditionalGeneration, StackT5Config

####################################################################################################


def recursive_print(name, val, spaces=0):
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


####################################################################################################

# The simple map of names for "automated" rules.
NAME_MAP = {
    "attention.dense": ".attn.c_proj.",
    "self_attention.dense": ".attn.c_proj.",
    "inter_attention.dense": ".crossattention.c_proj.",
    "mlp.dense_h_to_4h": ".mlp.c_fc.",
    "mlp.dense_4h_to_h": ".mlp.c_proj.",
    "self_attention.key_value": ".attn.c_attn.",
    "self_attention.query": ".attn.q_attn.",
    "inter_attention.query": ".crossattention.q_attn.",
    "inter_attention.key_value": ".crossattention.c_attn.",
    "post_inter_attention_layernorm": ".ln_cross_attn."
}

# Default args in case args are not found in ckpt.
DEFAULT_ARGS_DICT = dict(
    vocab_size=50176,
    max_position_embeddings=2048,
    hidden_size=3072,
    num_layers=24,
    num_attention_heads=16,
    kv_channels=128,
    attention_head_type="multiquery",
    ffn_hidden_size=16384,
    bias_gelu_fusion=True,
    openai_gelu=False,
    hidden_dropout=0.1,
    attention_dropout=0.1,
    layernorm_epsilon=1e-5,
    attention_softmax_in_fp32=False,
    apply_query_key_layer_scaling=True,
    init_method_std=0.02,
)

DEFAULT_ARGS = SimpleNamespace()
for k, v in DEFAULT_ARGS_DICT.items():
    setattr(DEFAULT_ARGS, k, v)


def convert_megatron_checkpoint(input_state_dict):
    # The converted output model.
    output_state_dict = {}
    try:
        ds_args = input_state_dict["args"]
    except KeyError:
        ds_args = DEFAULT_ARGS


    if ds_args.bias_gelu_fusion:
        activation_function = "gelu_pytorch_tanh"
    elif ds_args.openai_gelu:
        activation_function = "gelu_new"
    else:
        activation_function = "gelu"    

    if ds_args.attention_head_type == "multihead":
        multi_query = False
    else:
        assert ds_args.attention_head_type == "multiquery"
        multi_query = True

    attention_softmax_in_fp32 = ds_args.attention_softmax_in_fp32 or ds_args.apply_query_key_layer_scaling

    # The model.
    try:
        model = input_state_dict["model"]["language_model"]
        # Megatron-LM checkpoint version
        checkpoint_version = input_state_dict["checkpoint_version"]
        if checkpoint_version < 2.0:
            raise NotImplementedError(f"Checkpoint version {checkpoint_version} not supported.")
    except KeyError:
        model = input_state_dict["language_model"]

    try:
        vocab_size = ds_args.vocab_size
    except (AttributeError, KeyError) as e:
        vocab_size = model["embedding"]["word_embeddings"]["weight"].size(0)


    # Spell out all parameters in case the defaults change.
    config = StackT5Config(
        architectures=["StackT5ForConditionalGeneration"],
        vocab_size=vocab_size,
        n_positions=ds_args.max_position_embeddings,
        n_embd=ds_args.hidden_size,
        num_encoder_layers=ds_args.num_layers,
        num_decoder_layers=ds_args.num_layers,
        n_head=ds_args.num_attention_heads,
        n_kv_head=ds_args.kv_channels,
        n_inner=ds_args.ffn_hidden_size,
        activation_function=activation_function,
        resid_pdrop=ds_args.hidden_dropout,
        embd_pdrop=ds_args.hidden_dropout,
        attn_pdrop=ds_args.attention_dropout,
        layer_norm_epsilon=ds_args.layernorm_epsilon,
        initializer_range=ds_args.init_method_std,
        scale_attn_weights=ds_args.apply_query_key_layer_scaling,
        use_cache=True,
        bos_token_id=49157,
        eos_token_id=49158,
        pad_token_id=49156,
        attention_softmax_in_fp32=attention_softmax_in_fp32,
        scale_attention_softmax_in_fp32=attention_softmax_in_fp32,
        multi_query=multi_query,
    )

    # from pprint import pprint
    # pprint(vars(ds_args))
    # pprint(config)

    

    # The word embeddings, truncated to to vocab_size rows.
    word_embeddings = model["embedding"]["word_embeddings"]["weight"][: config.vocab_size, :]
    output_state_dict["transformer.wte.weight"] = word_embeddings

    # The position embeddings.
    output_state_dict["transformer.wpe.weight"] = model["embedding"]["position_embeddings"]["weight"]

    # The transformer.
    encoder = model["encoder"]
    decoder = model["decoder"]

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Extract the layers.
    for key, val in encoder.items():
        # Match the name.
        m = layer_re.match(key)

        # Stop if that's not a layer
        if m is None:
            break

        # The index of the layer.
        layer_idx = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)

        # The name of the layer.
        layer_name = f"transformer.encoder.{layer_idx}"

        # For layernorm(s), simply store the layer norm.
        if op_name.endswith("layernorm"):
            ln_name = "ln_1" if op_name.startswith("input") else "ln_2"
            output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = val

        # Copy the parameters.
        else:
            output_state_dict[layer_name + NAME_MAP[op_name] + weight_or_bias] = val

    # DEBUG.
    assert config.num_encoder_layers == layer_idx + 1

    # The final layernorm.
    output_state_dict["transformer.encoder_ln_f.weight"] = encoder["final_layernorm.weight"]
    output_state_dict["transformer.encoder_ln_f.bias"] = encoder["final_layernorm.bias"]

    # Extract the layers.
    for key, val in decoder.items():
        # Match the name.
        m = layer_re.match(key)

        # Stop if that's not a layer
        if m is None:
            break

        # The index of the layer.
        layer_idx = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)

        # The name of the layer.
        layer_name = f"transformer.decoder.{layer_idx}"

        # For layernorm(s), simply store the layer norm.
        if op_name.endswith("layernorm") and "inter_attention" not in op_name:
            ln_name = "ln_1" if op_name.startswith("input") else "ln_2"
            output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = val
            
        # Copy the parameters.
        else:
            output_state_dict[layer_name + NAME_MAP[op_name] + weight_or_bias] = val

    # DEBUG.
    assert config.num_decoder_layers == layer_idx + 1

    # The final layernorm.
    output_state_dict["transformer.decoder_ln_f.weight"] = decoder["final_layernorm.weight"]
    output_state_dict["transformer.decoder_ln_f.bias"] = decoder["final_layernorm.bias"]

    # For LM head, transformers' wants the matrix to weight embeddings.
    output_state_dict["lm_head.weight"] = word_embeddings
    output_state_dict["lm_head.bias"] = input_state_dict["lm_head"]["bias"]

    # It should be done!
    return config, output_state_dict


####################################################################################################


def main(argv=None):
    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    parser.add_argument(
        "--path_to_checkpoint",
        type=str,
        help="Path to the checkpoint file (.zip archive or direct .pt file)",
    )
    parser.add_argument(
        "--save_dir", help="Path where the converted model is saved. Will use the checkpoint directory if not provided"
    )
    parser.add_argument(
        "--hub_push_path",
        help="Optional path hugging face datasets where to push the data.",
    )
    args = parser.parse_args(argv)

    # Extract the basename.
    basename = args.save_dir or os.path.dirname(args.path_to_checkpoint)

    # Load the model.
    print(f"Extracting PyTorch state dictionary from {args.path_to_checkpoint}")
    input_state_dict = torch.load(args.path_to_checkpoint, map_location="cpu")

    # Convert.
    print("Converting")
    config, output_state_dict = convert_megatron_checkpoint(input_state_dict)

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    hf_model = StackT5ForConditionalGeneration(config)
    print(hf_model)
    hf_model.load_state_dict(output_state_dict)
    hf_model.save_pretrained(
        os.path.join(basename, "converted_ckpt")
    )
    if args.hub_push_path is not None:
        hf_model.push_to_hub(args.hub_push_path)


####################################################################################################

if __name__ == "__main__":
    main()

####################################################################################################