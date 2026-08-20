# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

import os, sys

# This training harness must never attempt an online W&B sync. Keep local
# offline logging when the package is installed, and allow training to run
# without W&B installed at all.
os.environ["WANDB_MODE"] = "offline"
os.environ.setdefault("WANDB_SILENT", "true")

from stokenizer import STokenizer
try:
    import wandb
except ImportError:
    wandb = None

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
import torch.distributed as dist
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

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
import yaml
import json
import gc
import argparse
import functools
from contextlib import nullcontext
from utils import Config, set_seed


class DistributedEvalSampler(Sampler):
    """Shard evaluation without padding/duplicating samples across ranks."""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.world_size - 1) // self.world_size)


def make_dataloader(
    dataset,
    *,
    batch_size,
    collator,
    sampler,
    configs,
):
    """Build a pinned, reusable input pipeline from config knobs."""

    num_workers = int(getattr(configs, "num_workers", 0))
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "collate_fn": collator,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(getattr(configs, "pin_memory", True)),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(getattr(configs, "prefetch_factor", 2))
    return torch.utils.data.DataLoader(**kwargs)


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key != "idx" and value is not None
    }


def state_dict_for_save(parallel_model, strategy):
    """Return a loadable checkpoint without gathering it onto every GPU."""

    if strategy == "fsdp":
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            parallel_model, StateDictType.FULL_STATE_DICT, save_policy
        ):
            return parallel_model.state_dict()
    return parallel_model.module.state_dict()

def main():
    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args = parser.parse_args()
    # init distributed environment
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", local_rank)
    )

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    torch.set_float32_matmul_precision(
        getattr(configs, "float32_matmul_precision", "high")
    )
    torch.backends.cuda.matmul.allow_tf32 = bool(
        getattr(configs, "allow_tf32", True)
    )
    torch.backends.cudnn.allow_tf32 = bool(getattr(configs, "allow_tf32", True))
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
    attention_backend = getattr(configs, "attention_backend", None)
    model_kwargs = (
        {"attn_implementation": attention_backend}
        if attention_backend not in {None, "", "auto"}
        else {}
    )
    if pretrained_model_id in {None, "None", ""}:
        if rank == 0:
            print(
                "Initializing causal LM from model config: "
                f"{configs.model_id} (random weights)"
            )
        model_config = AutoConfig.from_pretrained(configs.model_id)
        model = AutoModelForCausalLM.from_config(model_config, **model_kwargs)
    else:
        if rank == 0:
            print(
                "Loading pretrained causal LM from Hugging Face: "
                f"{pretrained_model_id}"
            )
            print(f"Bypassing model_id config: {configs.model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_id, **model_kwargs
        )

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

    resolved_config_path = os.path.join(save_dir, "model_config.json")
    if rank == 0:
        model.config.to_json_file(resolved_config_path, use_diff=False)
        print(f"Saved resolved model config to: {resolved_config_path}")
        if tokenizer_id != "stokenizer":
            tokenizer_path = os.path.join(save_dir, "tokenizer")
            tokenizer.save_pretrained(tokenizer_path)
            print(f"Saved resolved tokenizer to: {tokenizer_path}")
    torch.distributed.barrier()

    if rank == 0:
        print(model)

    loaded = False

    if configs.load_model_path != "None":
        saved_weights = torch.load(
            configs.load_model_path,
            map_location="cpu",
            weights_only=True,
        )

        if configs.coconut and not any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # we are loading a base model into coconut model
            # e.g., for GSM8k, we used a SFTed model to skip the stage 0
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif not configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            raise ValueError("Cannot load coconut model weights into a causallm model")

        elif configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # loading from preempted run
            # will handle later
            pass

        else:
            # resume or evaluate sft model
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

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

    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    distributed_strategy = str(
        getattr(configs, "distributed_strategy", "fsdp")
    ).lower()
    if distributed_strategy not in {"ddp", "fsdp"}:
        raise ValueError("distributed_strategy must be 'ddp' or 'fsdp'")

    if bool(getattr(configs, "gradient_checkpointing", False)):
        if isinstance(model, Coconut):
            raise ValueError(
                "gradient checkpointing is incompatible with Coconut's "
                "training-time latent KV cache; use FSDP or a smaller batch"
            )
        base_model = model.base_causallm if hasattr(model, "base_causallm") else model
        if not hasattr(base_model, "gradient_checkpointing_enable"):
            raise ValueError("this model does not support gradient checkpointing")
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_model.config.use_cache = False

    print(
        f"Running {distributed_strategy.upper()} on rank={rank}, "
        f"local_rank={local_rank}, world_size={world_size}"
    )
    model = model.to(local_rank)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            # GPT2Block,       # for GPT2, we don't need to shard layers (it becomes DDP)
            LlamaDecoderLayer  # only shard llama's layers.
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

    if distributed_strategy == "fsdp" and not configs.only_eval:
        parallel_model = FSDP(
            model,
            auto_wrap_policy=llama_auto_wrap_policy,
            device_id=local_rank,
            use_orig_params=True,
        )
    else:
        # The strict finite-state readonly_input path intentionally does not
        # use the sealed-prefix transition parameters. Let DDP account for
        # those unused parameters instead of failing on the next iteration.
        finite_state_config = getattr(configs, "finite_state", {}) or {}
        find_unused_parameters = bool(
            finite_state_config.get("enabled", False)
            and finite_state_config.get("access_mode", "readonly_input")
            == "readonly_input"
        )
        parallel_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            find_unused_parameters=find_unused_parameters,
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

    if (
        not configs.debug
        and not configs.only_eval
        and rank == 0
        and wandb is not None
    ):
        wandb_run = wandb.init(
            project=configs.project,
            name=configs.name,
            mode="offline",
        )
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None
        if not configs.debug and not configs.only_eval and rank == 0:
            print("W&B is unavailable; continuing without W&B logging.")


    optimizer = optim.AdamW(
        parallel_model.parameters(),
        lr=configs.lr,
        weight_decay=configs.weight_decay,
        fused=bool(getattr(configs, "fused_optimizer", True)),
    )

    best_acc = 0

    collator = MyCollator(
        tokenizer,
        latent_id=latent_id,
        label_pad_token_id=-100,
        pad_to_multiple_of=getattr(configs, "pad_to_multiple_of", None),
    )

    for epoch in range(configs.resume, configs.num_epochs):
        
        scheduled_stage = (
            0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
        )
        if rank == 0:
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

            valid_gen_sampler = DistributedEvalSampler(
                dataset_gen_val,
                rank=rank,
                world_size=world_size,
            )
            finite_state_enabled = bool(
                (getattr(configs, "finite_state", {}) or {}).get("enabled", False)
            )
            eval_batch_size = int(getattr(configs, "batch_size_eval", 1))
            # The strict reference adapter intentionally processes one example
            # at a time; the vectorized Coconut path supports real eval batches.
            if finite_state_enabled:
                eval_batch_size = 1
            valid_gen_dataloader = make_dataloader(
                dataset_gen_val,
                batch_size=eval_batch_size,
                collator=collator,
                sampler=valid_gen_sampler,
                configs=configs,
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
            train_sampler = DistributedSampler(
                dataset_train,
                shuffle=True,
                seed=configs.seed,
            )
            train_sampler.set_epoch(epoch)
            train_dataloader = make_dataloader(
                dataset_train,
                batch_size=configs.batch_size_training,
                collator=collator,
                sampler=train_sampler,
                configs=configs,
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

            valid_loss_dataloader = make_dataloader(
                dataset_loss_val,
                batch_size=configs.batch_size_training,
                collator=collator,
                sampler=DistributedEvalSampler(
                    dataset_loss_val, rank=rank, world_size=world_size
                ),
                configs=configs,
            )

            if configs.reset_optimizer and scheduled_stage < configs.max_latent_stage:
                del optimizer

                optimizer = optim.AdamW(
                    parallel_model.parameters(),
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                    fused=bool(getattr(configs, "fused_optimizer", True)),
                )

            parallel_model.module.train()
            torch.cuda.reset_peak_memory_stats(local_rank)

            total_length = (
                len(train_dataloader) + configs.gradient_accumulation_steps - 1
            ) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
                disable=rank != 0,
            )

            for step, batch in enumerate(train_dataloader):

                if step == 0 and wandb_run and rank == 0:
                    print("logging training data")
                    cur_bs = min(
                        len(batch["input_ids"]),
                        int(getattr(configs, "log_data_examples", 2)),
                    )
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
                batch = move_batch_to_device(batch, local_rank)

                should_step = (
                    (step + 1) % configs.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                )
                accumulation_group_start = (
                    step // configs.gradient_accumulation_steps
                ) * configs.gradient_accumulation_steps
                accumulation_divisor = min(
                    configs.gradient_accumulation_steps,
                    len(train_dataloader) - accumulation_group_start,
                )
                sync_context = (
                    nullcontext()
                    if should_step
                    else parallel_model.no_sync()
                )
                with sync_context:
                    outputs = parallel_model(**batch)
                    loss = outputs.loss / accumulation_divisor
                    loss.backward()

                if should_step:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pbar.update(1)
                    if rank == 0:
                        loss_value = float(
                            loss.detach().float() * accumulation_divisor
                        )
                        pbar.set_description(
                            f"Training Epoch: {epoch+1}/{configs.num_epochs}, "
                            f"batch {step}/{len(train_dataloader)} "
                            f"(loss: {loss_value:.4f})"
                        )
                        log_every = max(
                            1, int(getattr(configs, "log_every_steps", 10))
                        )
                        if wandb_run and total_train_steps % log_every == 0:
                            wandb_run.log(
                                {
                                    "train/epoch": epoch + 1,
                                    "train/step": (
                                        epoch * len(train_dataloader) + step
                                    ),
                                    "train/loss": loss_value,
                                }
                            )
            pbar.close()
            peak_vram = torch.tensor(
                torch.cuda.max_memory_allocated(local_rank),
                device=local_rank,
                dtype=torch.float64,
            )
            dist.all_reduce(peak_vram, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(f"Peak training VRAM: {peak_vram.item() / 2**30:.2f} GiB/GPU")
            dist.barrier()

            if (
                not configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
            ):
                states = state_dict_for_save(
                    parallel_model, distributed_strategy
                )
                if rank == 0:
                    torch.save(
                        states, os.path.join(save_dir, f"checkpoint_{epoch + 1}")
                    )
                    print("saving model.")

                dist.barrier()
                del states
                gc.collect()
                torch.cuda.empty_cache()

            # val loss
            total_loss = torch.zeros((), device=local_rank, dtype=torch.float64)
            total_loss_examples = torch.zeros(
                (), device=local_rank, dtype=torch.long
            )

            with torch.no_grad():
                parallel_model.module.eval()
                for step, batch in enumerate(valid_loss_dataloader):

                    batch = move_batch_to_device(batch, local_rank)

                    outputs = parallel_model(**batch)
                    local_batch_size = batch["input_ids"].shape[0]
                    total_loss += outputs.loss.double() * local_batch_size
                    total_loss_examples += local_batch_size

                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_loss_examples, op=dist.ReduceOp.SUM)
                mean_eval_loss = (
                    total_loss / total_loss_examples.clamp_min(1)
                ).item()

                if wandb_run and rank == 0:

                    log_dict = {
                        "eval/loss": mean_eval_loss,
                    }
                    wandb_run.log(log_dict)
                    print("eval loss", mean_eval_loss)

        # if scheduled_stage >= configs.max_latent_stage:
        if True:
            # val generation accuracy
            total_length = len(valid_gen_dataloader)

            pbar = tqdm(
                colour="blue",
                desc=f"Test Accuracy",
                total=total_length,
                dynamic_ncols=True,
                disable=rank != 0,
            )
            cor, cor_cot, total = (
                torch.tensor(0, device=local_rank),
                torch.tensor(0, device=local_rank),
                torch.tensor(0, device=local_rank),
            )

            with torch.no_grad():
                parallel_model.module.eval()
                for idx, batch in enumerate(valid_gen_dataloader):
                    test_indices = batch["idx"]
                    batch = {
                        k: v.to(local_rank, non_blocking=True)
                        for k, v in batch.items()
                        if v is not None and k not in ["idx", "position_ids"]
                    }
                    # https://github.com/huggingface/transformers/issues/32492

                    # FSDP requires the same number of generation forwards on
                    # every rank; DDP has no such constraint.
                    synced_generation = (
                        distributed_strategy == "fsdp" and not configs.only_eval
                    )
                    if configs.cot:
                        outputs = parallel_model.module.generate(
                            **batch,
                            max_new_tokens=64,
                            synced_gpus=synced_generation,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    elif configs.no_cot:
                        outputs = parallel_model.module.generate(
                            **batch,
                            max_new_tokens=64,
                            synced_gpus=synced_generation,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    else:
                        outputs = parallel_model.module.generate(
                            **batch,
                            max_new_tokens=1,
                            synced_gpus=synced_generation,
                            eos_token_id=tokenizer.eos_token_id,
                        )

                    for sample_offset, test_idx in enumerate(test_indices):
                        answer = str(answers_val[int(test_idx)])
                        text_output = tokenizer.decode(
                            outputs[sample_offset], skip_special_tokens=True
                        ).replace("<eos>", "").strip()
                        answer_output = (
                            text_output.split("[A]")[-1]
                            .replace(",", "")
                            .strip()
                        )
                        total += 1
                        cor += answer_output == answer

                        if idx * eval_batch_size + sample_offset < 5 and rank == 0:
                            print(f"Question {int(test_idx)}: Answer = '{answer}'")
                            print(
                                "Full output: "
                                f"'{tokenizer.decode(outputs[sample_offset])}'"
                            )
                            print(f"Extracted Output: '{answer_output}'")

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
                and configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
            ):
                states = state_dict_for_save(
                    parallel_model, distributed_strategy
                )

                if rank == 0:
                    torch.save(states, os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
                    print("saving model.")

                best_acc = cor / total

                dist.barrier()
                del states
                gc.collect()
                torch.cuda.empty_cache()

    if wandb_run:
        wandb_run.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
