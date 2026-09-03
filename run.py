# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.distributed
import torch.optim as optim
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from stokenizer import STokenizer
import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from coconut import Coconut
from finite_cot.rbs_adapter import StrictFiniteStateCoconut
from dataset import (
    MyCollator,
    get_graph_latent_question_dataset,
    get_graph_no_latent_question_dataset,
    get_graph_latent_cot_dataset,
    get_graph_no_cot_dataset,
    get_graph_cot_dataset,
)

from tqdm import tqdm
import os, sys
import yaml
import json
import gc
import argparse
import functools
import math
from utils import Config, set_seed


LORA_RANK = 16
LORA_LEARNING_RATE = 1e-4
FINITE_PROJECTION_LEARNING_RATE = 3e-4
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0


def add_pretrained_lora(model):
    """Freeze a pretrained LM and add the trainable task interface and LoRA."""

    model_type = getattr(model.config, "model_type", None)
    target_modules = None
    if model_type == "qwen3":
        # PEFT 0.15 does not provide an automatic Qwen3 mapping. Cover both
        # attention and MLP projections without adapting embeddings/lm_head.
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj",
        ]

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_RANK,
            lora_alpha=LORA_RANK*2,
            lora_dropout=0.05,
            bias="none",
            # Hugging Face GPT-2 projections use Conv1D's transposed weight
            # layout; standard Linear-based architectures keep the default.
            fan_in_fan_out=model_type == "gpt2",
            target_modules=target_modules,
        ),
    )

    # The symbolic vocabulary is a new task interface. PEFT freezes the whole
    # base model, so explicitly keep the resized/tied input and output weights
    # trainable alongside LoRA. The finite-state modules are constructed later
    # and are trainable by default.
    interface_modules = [
        model.get_input_embeddings(),
        model.get_output_embeddings(),
    ]
    for module in interface_modules:
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True
    return model


def load_training_checkpoint(model, saved_weights, *, coconut, uses_lora):
    """Load weights after the final LoRA/Coconut module topology is present."""

    has_coconut_wrapper = any(
        key.startswith("base_causallm") for key in saved_weights
    )
    has_lora = any("lora_" in key for key in saved_weights)

    if uses_lora and not has_lora:
        raise ValueError(
            "The selected pretrained model uses LoRA, but the checkpoint has no "
            "LoRA weights. Pre-LoRA checkpoints are not compatible; start a new "
            "run or load a checkpoint produced by this LoRA configuration."
        )
    if not coconut and has_coconut_wrapper:
        raise ValueError("Cannot load Coconut model weights into a causal LM model")

    weights_to_load = saved_weights
    if coconut and not has_coconut_wrapper:
        # Preserve the existing ability to initialize a non-pretrained Coconut
        # run from a base causal-LM checkpoint, but load it through the final
        # wrapper topology rather than loading the model in two different places.
        weights_to_load = {
            f"base_causallm.{key}": value for key, value in saved_weights.items()
        }

    incompatible = model.load_state_dict(weights_to_load, strict=False)
    if uses_lora:
        missing_lora = [key for key in incompatible.missing_keys if "lora_" in key]
        unexpected_lora = [
            key for key in incompatible.unexpected_keys if "lora_" in key
        ]
        if missing_lora or unexpected_lora:
            raise ValueError(
                "The checkpoint LoRA topology does not match the current model: "
                f"missing={missing_lora}, unexpected={unexpected_lora}"
            )
    return incompatible


def trainable_parameter_counts(model):
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def optimizer_parameter_groups(
    model,
    learning_rate,
    *,
    lora_learning_rate=LORA_LEARNING_RATE,
    finite_projection_learning_rate=FINITE_PROJECTION_LEARNING_RATE,
):
    """Split trainable weights into base, LoRA, and finite-projection groups."""

    grouped = {
        "base": {"params": [], "lr": learning_rate},
        "lora": {"params": [], "lr": lora_learning_rate},
        "finite_projection": {
            "params": [],
            "lr": finite_projection_learning_rate,
        },
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        normalized_name = name.replace("_fsdp_wrapped_module.", "")
        if (
            "bottleneck.up." in normalized_name
            or "bottleneck.down." in normalized_name
        ):
            group_name = "finite_projection"
        elif "lora_" in normalized_name:
            group_name = "lora"
        else:
            group_name = "base"
        grouped[group_name]["params"].append(parameter)

    return [
        {**group, "name": name}
        for name, group in grouped.items()
        if group["params"]
    ]


def create_optimizer(model, learning_rate, weight_decay):
    return optim.AdamW(
        optimizer_parameter_groups(model, learning_rate),
        weight_decay=weight_decay,
    )


def create_lr_scheduler(optimizer, num_training_steps, warmup_ratio=WARMUP_RATIO):
    """Use linear warm-up followed by cosine decay to zero."""

    if num_training_steps < 1:
        raise ValueError("num_training_steps must be positive")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    warmup_steps = min(
        num_training_steps,
        max(1, math.ceil(num_training_steps * warmup_ratio)),
    )

    def lr_multiplier(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        decay_steps = max(1, num_training_steps - warmup_steps)
        progress = min(1.0, (current_step - warmup_steps) / decay_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


def trainable_state_dict(model, state_dict=None):
    """Return only parameters that are updated during training.

    Tied embedding aliases are intentionally deduplicated: loading any one of
    their state-dict entries updates the shared input/output parameter. FSDP's
    internal wrapper component is normalized because it may be present in
    parameter names while omitted from state-dict keys.
    """

    def canonical_name(name):
        return name.replace("_fsdp_wrapped_module.", "")

    trainable_names = {
        canonical_name(name)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if state_dict is None:
        state_dict = model.state_dict()
    selected = {
        name: value
        for name, value in state_dict.items()
        if canonical_name(name) in trainable_names
    }
    if not selected:
        raise ValueError("No trainable parameters were found for the checkpoint")
    return selected


def checkpoint_path(save_dir, epoch, *, save_best_only=False):
    """Return the output path for an epoch or single-best checkpoint."""

    filename = "best_model.pt" if save_best_only else f"checkpoint_{epoch}"
    return os.path.join(save_dir, filename)


def unwrap_parallel_model(model):
    """Return the underlying model for plain, DDP, and FSDP execution."""

    return model.module if hasattr(model, "module") else model


def checkpoint_state_dict(model, *, uses_fsdp):
    """Use FSDP's state-dict hooks only when the model is actually sharded."""

    state_model = model if uses_fsdp else unwrap_parallel_model(model)
    return state_model, state_model.state_dict()


def main():
    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    parser.add_argument(
        "--save-best-only",
        action="store_true",
        help=(
            "save only the weights with the highest validation accuracy, "
            "overwriting best_model.pt"
        ),
    )
    args = parser.parse_args()
    # init distributed environment
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)
    if args.save_best_only:
        config_dict["save_best_only"] = True

    if rank == 0:
        print("Config:", config_dict)

    configs = Config(config_dict)
    configs.save_best_only = getattr(configs, "save_best_only", False)
    configs.save_only_improve = getattr(configs, "save_only_improve", False)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier()
    cur_ckpts = os.listdir(save_dir)
    checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
    checkpoints.sort(key=lambda x: int(x.split("_")[1]))

    # check if the job is preempted and resumed.

    if len(checkpoints) > 0 and not configs.only_eval:
        # if there are previous checkpoints, and only_eval is False
        # it means the previous run was preempted and the program is restarted.
        # need to find the latest checkpoint and resume from that.

        if rank == 0:
            print(
                f"Warning: found previous run and gonna resume from that. the inputted `resume` argument is ignored!"
            )

        # Get the last item in the sorted list
        latest_checkpoint = checkpoints[-1]
        configs.resume = int(latest_checkpoint.split("_")[1])
        load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)

        configs.load_model_path = load_dir
        print(f"Loading from previous run epoch_{configs.resume}!")

    elif configs.resume != 0:
        # by setting `resume`, we can skip a few epoches at the beginning.
        if configs.load_model_path == "None":
            print(
                f"Warning: you want to skip the first {configs.resume} but you are not loading any existing checkpoint!"
            )
            # not an intended use case at this point
        print(
            f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
        )

    
    tokenizer_id = getattr(configs, "tokenizer", "stokenizer")
    if tokenizer_id == "stokenizer":
        tokenizer = STokenizer()
        if rank == 0:
            print("Using built-in tokenizer: stokenizer")
    else:
        if rank == 0:
            print(f"Loading tokenizer from Hugging Face: {tokenizer_id}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        special_tokens = {
            "additional_special_tokens": [
                "<|start-latent|>",
                "<|end-latent|>",
                "<|latent|>",
            ]
        }
        added_tokens = tokenizer.add_special_tokens(special_tokens)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
                pad_message = f"using EOS token {tokenizer.eos_token!r} as padding"
            else:
                added_tokens += tokenizer.add_special_tokens(
                    {"pad_token": "<|pad|>"}
                )
                pad_message = "added <|pad|> as padding"
        else:
            pad_message = f"using existing padding token {tokenizer.pad_token!r}"
        tokenizer.padding_side = "right"
        if rank == 0:
            print(
                f"Registered latent special tokens; added {added_tokens} new "
                f"tokens; {pad_message}"
            )
            print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    pretrained_model_id = getattr(configs, "pretrained_model_id", None)
    uses_lora = pretrained_model_id not in {None, "None", ""}
    if not uses_lora:
        if rank == 0:
            print(
                "Initializing causal LM from model config: "
                f"{configs.model_id} (random weights)"
            )
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(configs.model_id)
        )
    else:
        if rank == 0:
            print(
                "Loading pretrained causal LM from Hugging Face: "
                f"{pretrained_model_id}"
            )
            print(f"Bypassing model_id config: {configs.model_id}")
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_id)

    old_vocab_size = model.get_input_embeddings().num_embeddings
    if old_vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        if rank == 0:
            print(
                f"Resized model token embeddings: {old_vocab_size} -> "
                f"{len(tokenizer)}"
            )

    model.config.vocab_size = len(tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    if uses_lora:
        model = add_pretrained_lora(model)
        if rank == 0:
            print(f"Attached LoRA adapters with rank r={LORA_RANK}")

    resolved_config_path = os.path.join(save_dir, "model_config.json")
    if rank == 0:
        model.config.to_json_file(resolved_config_path, use_diff=False)
        print(f"Saved resolved model config to: {resolved_config_path}")
        if tokenizer_id != "stokenizer":
            tokenizer_path = os.path.join(save_dir, "tokenizer")
            tokenizer.save_pretrained(tokenizer_path)
            print(f"Saved resolved tokenizer to: {tokenizer_path}")
    torch.distributed.barrier()

    print(model)

    if configs.load_model_path != "None":
        saved_weights = torch.load(
            configs.load_model_path, map_location=torch.device(rank)
        )

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    if configs.coconut:
        finite_state = getattr(configs, "finite_state", {}) or {}
        if finite_state.get("enabled", False):
            model = StrictFiniteStateCoconut(
                model,
                latent_id,
                start_id,
                end_id,
                tokenizer.eos_token_id,
                finite_state,
            )
        else:
            model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    if configs.load_model_path != "None":
        print(
            load_training_checkpoint(
                model,
                saved_weights,
                coconut=configs.coconut,
                uses_lora=uses_lora,
            )
        )

    if rank == 0:
        trainable, total = trainable_parameter_counts(model)
        print(
            f"Trainable parameters: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.4f}%)"
        )

    model = model.to(rank)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            # GPT2Block,       # for GPT2, we don't need to shard layers (it becomes DDP)
            LlamaDecoderLayer,
            Qwen3DecoderLayer,
        },
    )

    training_dtype = getattr(configs, "training_dtype", None)
    if training_dtype is None:
        training_dtype = "bfloat16" if configs.bf16 else "float32"
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if training_dtype not in dtype_map:
        raise ValueError(
            "training_dtype must be float16, float32, or bfloat16"
        )
    model.to(dtype_map[training_dtype])

    uses_fsdp = world_size > 1 and not configs.only_eval
    if world_size == 1:
        parallel_model = model
        if rank == 0:
            print("Running plain PyTorch on one GPU (no DDP/FSDP)")
    elif configs.only_eval:
        parallel_model = DDP(model, device_ids=[rank])
    else:
        if rank == 0:
            print(f"Running FSDP with world size {world_size}")
        parallel_model = FSDP(
            model,
            auto_wrap_policy=llama_auto_wrap_policy,
            device_id=rank,
            # Preserve parameter identities so optimizer groups can distinguish
            # LoRA and finite-state projection weights after FSDP wrapping.
            use_orig_params=True,
        )

    del model

    if rank == 0:
        print(parallel_model)

    answers_val = [
        d["target"] for d in json.load(open(configs.val_path))
    ]

    if "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 128

    total_train_steps = 0

    if not configs.debug and not configs.only_eval and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None


    optimizer = create_optimizer(
        parallel_model,
        learning_rate=configs.lr,
        weight_decay=configs.weight_decay,
    )
    lr_scheduler = None

    best_acc = float("-inf")

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    for epoch in range(configs.resume, configs.num_epochs):
        
        scheduled_stage = (
            0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
        )
        print("scheduled_stage", scheduled_stage)
        
        if True:
            if configs.cot or configs.no_cot:
                dataset_gen_val = get_graph_no_latent_question_dataset(
                    configs.val_path,
                    configs,
                    tokenizer,
                )
            else:   
                dataset_gen_val = get_graph_latent_question_dataset(
                    configs.val_path,
                    scheduled_stage,
                    configs,
                    tokenizer,
                )

            valid_gen_dataloader = torch.utils.data.DataLoader(
                dataset_gen_val,
                num_workers=1,
                pin_memory=True,
                batch_size=1,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_gen_val, shuffle=False),
            )

        if not configs.only_eval:

            if configs.cot:
                dataset_train = get_graph_cot_dataset(
                    configs.train_path,
                    configs,
                    tokenizer,
                )
            elif configs.no_cot:
                dataset_train = get_graph_no_cot_dataset(
                    configs.train_path,
                    configs,
                    tokenizer,
                )
            else:
                dataset_train = get_graph_latent_cot_dataset(
                    configs.train_path,
                    scheduled_stage,
                    configs,
                    tokenizer,
                )
            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )

            # the sampler is deterministic even if shuffle is set to True
            # so we have shuffled the dataset when it's constructed (at every epoch).
            if configs.cot:
                dataset_loss_val = get_graph_cot_dataset(
                    configs.val_path,
                    configs,
                    tokenizer,
                )
            elif configs.no_cot:
                dataset_loss_val = get_graph_no_cot_dataset(
                    configs.val_path,
                    configs,
                    tokenizer,
                )
            else:
                dataset_loss_val = get_graph_latent_cot_dataset(
                    configs.val_path,
                    scheduled_stage,
                    configs,
                    tokenizer,
                )

            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_loss_val, shuffle=False),
            )

            if configs.reset_optimizer and scheduled_stage < configs.max_latent_stage:
                del optimizer

                optimizer = create_optimizer(
                    parallel_model,
                    learning_rate=configs.lr,
                    weight_decay=configs.weight_decay,
                )
                lr_scheduler = None

            updates_per_epoch = math.ceil(
                len(train_dataloader) / configs.gradient_accumulation_steps
            )
            if lr_scheduler is None:
                lr_scheduler = create_lr_scheduler(
                    optimizer,
                    num_training_steps=(configs.num_epochs - epoch)
                    * updates_per_epoch,
                )

            unwrap_parallel_model(parallel_model).train()

            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=updates_per_epoch,
                dynamic_ncols=True,
            )

            for step, batch in enumerate(train_dataloader):

                if step == 0 and wandb_run and rank == 0:
                    print("logging training data")
                    cur_bs = len(batch["input_ids"])
                    text_str = ""
                    for data_idx in range(cur_bs):
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            text_str += (
                                str(batch["input_ids"][data_idx][token_idx].item())
                                + " "
                                + str(batch["attention_mask"][data_idx][token_idx].item())
                                + " "
                                + str(batch["labels"][data_idx][token_idx].item())
                                + " "
                                + tokenizer.decode(
                                    batch["input_ids"][data_idx][token_idx]
                                )
                                + "\n"
                            )
                        text_str += "====" * 10 + "\n"
                    text_table.add_data(total_train_steps, text_str)
                    # copy the table due to a bug in wandb
                    # https://github.com/wandb/wandb/issues/2981

                    # wandb_run.log({"data_table": copy(text_table)})
                    # this will produce larger and larger tables as the training progresses
                    # so we don't log it to wandb when the epoch number is set large
                
                total_train_steps += 1
                batch = {
                    key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                }

                outputs = parallel_model(**batch)

                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    if uses_fsdp:
                        grad_norm = parallel_model.clip_grad_norm_(MAX_GRAD_NORM)
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            parallel_model.parameters(), MAX_GRAD_NORM
                        )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                if wandb_run and rank == 0:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float()
                        * configs.gradient_accumulation_steps,
                    }
                    if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                        train_dataloader
                    ) - 1:
                        log_dict["train/grad_norm"] = grad_norm.detach().float()
                        for group in optimizer.param_groups:
                            log_dict[f"train/lr_{group['name']}"] = group["lr"]
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
            pbar.close()
            dist.barrier()

            if (
                not configs.save_only_improve
                and not configs.save_best_only
                and not configs.debug
                and not configs.only_eval
            ):
                state_model, states = checkpoint_state_dict(
                    parallel_model, uses_fsdp=uses_fsdp
                )
                if uses_lora:
                    states = trainable_state_dict(state_model, states)
                if rank == 0:
                    torch.save(
                        states,
                        checkpoint_path(save_dir, epoch + 1),
                    )
                    print("saving model.")

                dist.barrier()
                del states
                gc.collect()
                torch.cuda.empty_cache()

            # val loss
            total_loss = 0

            with torch.no_grad():
                unwrap_parallel_model(parallel_model).eval()
                for step, batch in enumerate(valid_loss_dataloader):

                    batch = {
                        key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                    }

                    outputs = parallel_model(**batch)
                    loss = outputs.loss
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    total_loss += loss.item() / world_size

                if wandb_run and rank == 0:

                    log_dict = {
                        "eval/loss": total_loss / len(valid_loss_dataloader),
                    }
                    wandb_run.log(log_dict)
                    print("eval loss", total_loss / len(valid_loss_dataloader))

        # if scheduled_stage >= configs.max_latent_stage:
        if True:
            # val generation accuracy
            total_length = len(valid_gen_dataloader)

            pbar = tqdm(
                colour="blue", desc=f"Test Accuracy", total=total_length, dynamic_ncols=True
            )
            cor, cor_cot, total = (
                torch.tensor(0, device=rank),
                torch.tensor(0, device=rank),
                torch.tensor(0, device=rank),
            )

            with torch.no_grad():
                unwrap_parallel_model(parallel_model).eval()
                for idx, batch in enumerate(valid_gen_dataloader):
                    test_idx = batch["idx"][0]

                    batch = {
                        k: v.to(rank)
                        for k, v in batch.items()
                        if v != None and k not in ["idx", "position_ids"]
                    }
                    # https://github.com/huggingface/transformers/issues/32492

                    assert len(batch["input_ids"]) == 1
                    answer = str(answers_val[test_idx.cpu().item()])
                    # answer_cot = cot_val[test_idx.cpu().item()]
                    # question = question_val[test_idx.cpu().item()]

                    total += 1

                    # synced_gpus=True in FSDP mode, as we need to keep # forward pass the same on each device
                    if configs.cot:
                        outputs = unwrap_parallel_model(parallel_model).generate(
                            **batch,
                            max_new_tokens=64,
                            synced_gpus=world_size > 1 and not configs.only_eval,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    elif configs.no_cot:
                        outputs = unwrap_parallel_model(parallel_model).generate(
                            **batch,
                            max_new_tokens=64,
                            synced_gpus=world_size > 1 and not configs.only_eval,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    else:
                        outputs = unwrap_parallel_model(parallel_model).generate(
                            **batch,
                        max_new_tokens=1,
                        synced_gpus=world_size > 1 and not configs.only_eval,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    text_output = tokenizer.decode(outputs[0], skip_special_tokens=True).replace("<eos>", "").strip()
                    answer_output = text_output.split("[A]")[-1].replace(",", "").strip()
                    cot_output = (
                        ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()
                    )

                    if idx < 5 and rank == 0:
                        # print some examples
                        print(
                            f"Question {test_idx}: Answer = '{answer}'"
                        )
                        print(f"Full output: '{tokenizer.decode(outputs[0])}'")
                        print(f"Extracted Output: '{answer_output}'")

                    cor += answer_output == answer
                    # cor_cot += cot_output == answer_cot

                    pbar.update(1)
                    pbar.set_description(
                        f"Test accuracy: {round(float(cor.detach().float() / total.detach().float()), 2)}"
                    )

                pbar.close()
                print(f"Device {rank}: Cor={cor}, Total={total}")

            dist.all_reduce(cor_cot, op=dist.ReduceOp.SUM)
            dist.all_reduce(cor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

            # cor_cot = cor_cot.item()
            cor = cor.item()
            total = total.item()
            if rank == 0:
                print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
                # print(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")
            sys.stdout.flush()

            if wandb_run:
                wandb_run.log({"eval/acc": cor / total})

            if configs.only_eval:
                break

            dist.barrier()
            if (
                cor / total > best_acc
                and (configs.save_only_improve or configs.save_best_only)
                and not configs.debug
                and not configs.only_eval
            ):
                state_model, states = checkpoint_state_dict(
                    parallel_model, uses_fsdp=uses_fsdp
                )
                if uses_lora:
                    states = trainable_state_dict(state_model, states)

                if rank == 0:
                    output_path = checkpoint_path(
                        save_dir,
                        epoch + 1,
                        save_best_only=configs.save_best_only,
                    )
                    torch.save(states, output_path)
                    print(f"saving model to {output_path}.")

                best_acc = cor / total

                dist.barrier()
                del states
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
