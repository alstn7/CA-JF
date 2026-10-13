#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, List
import glob
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader  # ### CHANGED
import transformers
from transformers.trainer_pt_utils import LabelSmoother
import datasets
import logging

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# import deepspeed
from transformers.integrations import HfDeepSpeedConfig

from accelerate import Accelerator, DeepSpeedPlugin

# Local imports
from pathlib import Path
path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))
from soft_flexattn_cllm_trainer_multiblock_conv import CllmTrainer

logger = logging.getLogger(__name__)
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    target_model_path: Optional[str] = field(
        default="models/vicuna-7b-v1.5",
        metadata={"help": "Path or HF id for the base model"},
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path or HF id for tokenizer. If None, use target_model_path."},
    )
    teacher_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path or HF id for the AR teacher model"},
    )
    qlora: bool = field(default=False, metadata={"help": "Enable QLoRA"})
    
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data JSONL"})
    lazy_preprocess: bool = True

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024, metadata={"help": "Max seq length for tokenizer/model"}
    )
    max_new_tokens: int = field(
        default=64, metadata={"help": "N_BLOCK in the Jacobi trajectory (keep at 64 unless you changed preprocessing)"}
    )
    report_to: str = field(default="wandb")
    use_gt_labels: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing to reduce activation memory"},
    )
    bf16: bool = field(
        default=True, metadata={"help": "Train in bfloat16. Keep True for FA2/SDPA efficiency."}
    )
    fp16: bool = field(default=False)

    max_pairs_per_step: int = field(
        default=10,
        metadata={"help": "How many (k_j, last_j) pairs to use per training step (<= 0 means use all)"},
    )

    # ---- new args for adaptive gate / KD ----
    use_keep_gate: bool = field(
        default=False,
        metadata={"help": "Enable learnable keep gate"},
    )
    keep_gate_hidden_size: int = field(
        default=256,
        metadata={"help": "Hidden size for keep gate MLP"},
    )
    lambda_keep: float = field(
        default=0.0,
        metadata={"help": "Weight for keep gate regularization"},
    )
    keep_gate_detach_features: bool = field(
        default=True,
        metadata={"help": "Detach features before keep gate to stabilize early experiments"},
    )

    use_teacher_kd: bool = field(
        default=False,
        metadata={"help": "Enable teacher-student KD"},
    )
    lambda_kd: float = field(
        default=0.0,
        metadata={"help": "Weight for KD loss"},
    )
    kd_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for KD"},
    )
    kd_on_consistency_only: bool = field(
        default=True,
        metadata={"help": "Apply KD only on Jacobi future-block / consistency positions"},
    )
    lambda_anchor: float = field(
        default=0.0,
        metadata={"help": "Weight for anchor-position auxiliary CE loss"},
    )

    anchor_weight: float = field(
        default=1.0,
        metadata={"help": "Internal multiplier for anchor-position CE loss"},
    )
    anchor_cons_alpha: float = field(
        default=0.0,
        metadata={"help": "Extra weight for consistency loss on anchor target positions."},
    )
    anchor_cons_type: str = field(
        default="all",
        metadata={
            "help": "Anchor consistency type: all | syntax | ctrlsyntax | math | math_final | math_plus_final | code_completion | all_plus_completion. "
                    "all uses the original code-anchor mask; syntax uses syntax/format tokens only; "
                    "ctrlsyntax uses syntax tokens plus control-flow/program-structure tokens; "
                    "math uses math-specific numeric/operator/LaTeX/final-answer anchors."
        },
    )
    anchor_cons_noise_gate: str = field(
        default="none",
        metadata={"help": "Noise gate for anchor consistency: none | mid | late"},
    )

    anchor_cons_late_threshold: float = field(
        default=0.3,
        metadata={"help": "Apply anchor consistency only when noise ratio <= threshold if gate=late."},
    )
    # ---- distance / stage-aware consistency weighting ----
    dist_cons_mode: str = field(
        default="none",
        metadata={
            "help": "Distance/stage consistency weighting: "
                    "none | noise_linear | noise_bucket | traj_linear | traj_bucket. "
                    "noise_* uses block noise ratio as a proxy for distance from final target; "
                    "traj_* uses trajectory index j/T as a proxy."
        },
    )
    dist_cons_near_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for middle/near states in bucketed distance consistency."},
    )
    dist_cons_far_weight: float = field(
        default=0.25,
        metadata={"help": "Weight for far/high-noise states in distance consistency."},
    )
    dist_cons_far_anchor_weight: float = field(
        default=0.5,
        metadata={"help": "Anchor weight for far/high-noise states when dist_cons_mode=target_bucket."},
    )
    dist_cons_mid_nonanchor_weight: float = field(
        default=0.25,
        metadata={"help": "Non-anchor weight for mid-noise states when dist_cons_mode=target_bucket."},
    )
    dist_cons_power: float = field(
        default=1.0,
        metadata={"help": "Power for linear distance consistency weighting."},
    )
    # ---- contraction / residual shrinking loss ----
    lambda_contract: float = field(
        default=0.0,
        metadata={"help": "Weight for trajectory contraction loss."},
    )
    contract_margin: float = field(
        default=0.0,
        metadata={"help": "Margin for contraction loss: penalize D_next > D_prev - margin."},
    )
    contract_on_anchor_only: bool = field(
        default=False,
        metadata={"help": "If True, apply contraction loss only on anchor target positions."},
    )

    # ---- error-aware progressive target forcing (EPJF) ----
    prog_target_mode: str = field(
        default="none",
        metadata={"help": "Progressive target subset forcing: none | epjf | sspc | rpc | residual_progressive. "
                          "When enabled, high-noise states supervise structural anchors only, "
                          "mid-noise states supervise structural+semantic keys, and low-noise states supervise full targets."},
    )
    prog_target_high_threshold: float = field(
        default=0.66,
        metadata={"help": "Noise ratio >= this is treated as high-noise for EPJF."},
    )
    prog_target_mid_threshold: float = field(
        default=0.33,
        metadata={"help": "Noise ratio between mid_threshold and high_threshold is treated as mid-noise; below is low-noise."},
    )
    prog_target_mismatch_only: bool = field(
        default=False,
        metadata={"help": "If True, EPJF supervises only noisy-clean mismatched positions, plus fallback positions when a mask is empty."},
    )
    prog_target_low_mismatch_only: bool = field(
        default=False,
        metadata={"help": "If True, low-noise EPJF also focuses on mismatched positions instead of all valid positions."},
    )
    prog_target_min_keep: int = field(
        default=1,
        metadata={"help": "Minimum number of consistency positions kept by EPJF; falls back to valid positions if too sparse."},
    )
    prog_target_random_high_keep: float = field(
        default=0.30,
        metadata={"help": "For epjf_code_soft_random: deterministic fraction of valid target positions additionally kept in high-noise states."},
    )
    prog_target_random_mid_keep: float = field(
        default=0.60,
        metadata={"help": "For epjf_code_soft_random: deterministic fraction of valid target positions additionally kept in mid-noise states."},
    )

    # ---- convergence-aware consistency weighting ----
    conv_cons_beta: float = field(
        default=0.0,
        metadata={"help": "Extra consistency weight for slow-to-converge trajectory positions."},
    )
    conv_cons_mode: str = field(
        default="persistence",
        metadata={"help": "Convergence difficulty score: persistence | global | local."},
    )
    conv_cons_max_weight: float = field(
        default=2.0,
        metadata={"help": "Clamp final per-token consistency weight to this max if > 0."},
    )
    conv_cons_warmup_steps: int = field(
        default=500,
        metadata={"help": "Linearly warm up conv_cons_beta over this many trainer steps. <=0 disables warmup."},
    )
    # ---- SSPC / RPC: residual progressive consistency ----
    prog_target_selected_weight: float = field(
        default=1.0,
        metadata={"help": "SSPC/RPC selected progressive-target token weight."},
    )
    prog_target_residual_high: float = field(
        default=0.0,
        metadata={"help": "SSPC/RPC residual full-consistency weight for non-selected valid tokens at high noise."},
    )
    prog_target_residual_mid: float = field(
        default=0.0,
        metadata={"help": "SSPC/RPC residual full-consistency weight for non-selected valid tokens at mid noise."},
    )
    prog_target_residual_low: float = field(
        default=1.0,
        metadata={"help": "SSPC/RPC residual weight for non-selected valid tokens at low noise. Usually 1.0 because low stage is full target."},
    )
    lambda_prog_aux: float = field(
        default=0.0,
        metadata={
            "help": "Non-interfering progressive auxiliary weight. "
                    "If >0, progressive target weights add a separate auxiliary consistency loss "
                    "instead of multiplying the primary ACW/full-token consistency weights."
        },
    )

def rank0_print(local_rank, *args):
    if local_rank in (0, -1):
        print(*args)

def safe_save_model_for_hf_trainer(model, tokenizer, output_dir, accelerator=None):  # ### CHANGED
    """
    Save weights. Works with Accelerator/DeepSpeed:
    - If accelerator provided, unwrap first.
    - If ZeRO-3, make sure your ds_config has stage3_gather_16bit_weights_on_model_save=true,
      otherwise rely on engine.save_checkpoint inside the trainer.
    """
    os.makedirs(output_dir, exist_ok=True)
    to_save = model
    if accelerator is not None:
        to_save = accelerator.unwrap_model(model)
    # Some DS engines expose .module; unwrap again if needed
    if hasattr(to_save, "module"):
        to_save = to_save.module
    to_save.save_pretrained(output_dir, safe_serialization=False)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

def preprocess_training_sequence(
    prompt_ids: List[int],
    prompt_ids_len: int,
    complete_training_sequence_ids: List[int],
    tokenizer: transformers.PreTrainedTokenizer,
    model: str,
    traj_position_indices: List[int],
) -> Dict:
    prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
    complete_training_sequence_ids = torch.tensor(complete_training_sequence_ids, dtype=torch.long)
    #labels_ids = torch.tensor(labels_ids, dtype=torch.long)
    return dict(
        prompt_ids=prompt_ids,
        prompt_ids_len=prompt_ids_len,
        input_ids=complete_training_sequence_ids,
        #labels_ids=labels_ids,
        traj_position_indices=torch.tensor(traj_position_indices, dtype=torch.long).unsqueeze(0),
    )

class JacobianDataset(Dataset):
    def __init__(self, raw_data, tokenizer, model: str, do_eval: bool=False, local_rank: int=-1):
        super().__init__()
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.do_eval = do_eval
        self.model = model
        rank0_print(local_rank, "Formatting inputs (lazy)...")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        sample = self.raw_data[i]
        ret = preprocess_training_sequence(
            sample["prompt_ids"],
            sample["prompt_ids_len"],
            sample["complete_training_sequence_ids"],
            self.tokenizer,
            self.model,
            traj_position_indices=sample["traj_position_indices"],
        )

        # Debug / filtering metadata.
        # This is the index AFTER trajectory_dataset filtering.
        # Keep it tensor-only so HF/torch collators can batch it safely.
        filtered_index = sample.get("filtered_index", i)
        ret["example_index"] = torch.tensor(int(filtered_index), dtype=torch.long)
        self.cached_data_dict[i] = ret
        return ret

def make_jacobian_data_module(tokenizer, trajectory_data, data_args, model: str, local_rank: int) -> Dict:
    assert data_args.lazy_preprocess, "only support lazy preprocess"
    train_dataset = JacobianDataset(trajectory_data, tokenizer=tokenizer, model=model, local_rank=local_rank)
    return dict(train_dataset=train_dataset, eval_dataset=None)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    training_args.local_rank = local_rank

    training_args.qlora = model_args.qlora

    ds_config_path = training_args.deepspeed or "/home/irteam/data-vol1/minsu/JacobiForcing/JacobiForcing/scripts/train/ds_config_cpu_offloading.json"
    training_args.deepspeed = ds_config_path

    # Accelerate + DeepSpeed plugin
    ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_config_path)  # path or dict
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps or 1,
        mixed_precision="bf16" if training_args.bf16 else ("fp16" if training_args.fp16 else "no"),
        log_with=training_args.report_to if training_args.report_to else None,
        deepspeed_plugin=ds_plugin,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO if accelerator.is_local_main_process else logging.ERROR,  # ### ADDED
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, bf16: {training_args.bf16}, fp16: {training_args.fp16}"
    )
    logger.info(f"Training/eval parameters {training_args}")

    config = transformers.AutoConfig.from_pretrained(
        model_args.target_model_path, cache_dir=training_args.cache_dir
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        raise ValueError(
            f"Use rope scaling is not supported: "
            f"model_max_length ({training_args.model_max_length}) > orig_ctx_len ({orig_ctx_len})"
        )
    config.use_cache = False

    # Attention backend used when loading the HF model.
    # Default preserves the original behavior.
    # For dense/eager diagnostic:
    #   JF_LOAD_ATTN_IMPL=eager
    # or:
    #   JF_LOAD_ATTN_IMPL=sdpa
    # To omit the argument entirely:
    #   JF_LOAD_ATTN_IMPL=none
    attn_impl = os.environ.get("JF_LOAD_ATTN_IMPL", "flex_attention").strip()
    if attn_impl.lower() in ("", "none", "default"):
        attn_impl = None

    if accelerator.is_main_process:
        print(f"[JF ATTN CONFIG] JF_LOAD_ATTN_IMPL={attn_impl}", flush=True)

    # partitioned loading for ZeRO-3/offload
    dschf = HfDeepSpeedConfig(ds_config_path)

    # student model
    model_kwargs = dict(
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None),
        low_cpu_mem_usage=True,
    )
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.target_model_path,
        **model_kwargs,
    )
    tokenizer_path = model_args.tokenizer_path or model_args.target_model_path

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
   
    # Gradient checkpointing
    if getattr(training_args, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    if model_args.qlora:
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=16, lora_dropout=0.05)
        model = get_peft_model(model, lora_cfg)

    # Data Prep
    # Data Prep
    data_path = data_args.data_path

    if os.path.isdir(data_path):
        parquet_files = sorted(str(p) for p in Path(data_path).rglob("*.parquet"))
        jsonl_files = sorted(str(p) for p in Path(data_path).rglob("*.jsonl"))
        json_files = sorted(str(p) for p in Path(data_path).rglob("*.json"))

        if parquet_files:
            data_loader_name = "parquet"
            data_files = {"train": parquet_files}
        elif jsonl_files or json_files:
            data_loader_name = "json"
            data_files = {"train": jsonl_files + json_files}
        else:
            raise ValueError(f"No parquet/json/jsonl files found under data_path={data_path}")
    else:
        suffix = Path(data_path).suffix.lower()
        if suffix == ".parquet":
            data_loader_name = "parquet"
        elif suffix in (".jsonl", ".json"):
            data_loader_name = "json"
        else:
            raise ValueError(f"Unsupported data_path suffix: {suffix} for {data_path}")
        data_files = {"train": data_path}

    rank0_print(
        training_args.local_rank,
        f"Loading dataset with loader={data_loader_name}, data_files={data_files if isinstance(data_files['train'], str) else len(data_files['train'])}",
    )

    trajectory_dataset = load_dataset(
        data_loader_name,
        data_files=data_files,
        split="train",
        cache_dir=training_args.cache_dir,
    )
    # trajectory_dataset = load_dataset(
    #     "json", data_files={"train": data_args.data_path}, split="train", cache_dir=training_args.cache_dir
    # )
    
    # ------------------------------------------------------------------
    # Filter over-length trajectory examples before wrapping with JacobianDataset.
    # Do not filter by decoded text or EOS-like tokens here.
    # Qwen chat trajectories normally contain <|im_start|>/<|im_end|>,
    # and Jacobi trajectory blocks can decode into noisy-looking text.
    # ------------------------------------------------------------------
    def _keep_valid_trajectory(example):
        seq = example.get("complete_training_sequence_ids", None)
        if seq is None:
            return True
        return len(seq) <= training_args.model_max_length

    before_n = len(trajectory_dataset)
    trajectory_dataset = trajectory_dataset.filter(
        _keep_valid_trajectory,
        desc=f"Filtering trajectories longer than {training_args.model_max_length}",
    )
    after_n = len(trajectory_dataset)

    if accelerator.is_main_process:
        print(
            f"[DATA FILTER] model_max_length={training_args.model_max_length} "
            f"kept={after_n}/{before_n}, dropped={before_n - after_n}",
            flush=True,
        )              
    # Stable index after length filtering.
    # This index is used for NaN-trigger sample tracking and resume scan.
    if "filtered_index" not in trajectory_dataset.column_names:
        trajectory_dataset = trajectory_dataset.add_column(
            "filtered_index",
            list(range(len(trajectory_dataset))),
        )  
    # Optional scan/resume filter.
    # JF_SCAN_START_INDEX: keep filtered_index >= start
    # JF_SCAN_END_INDEX:   keep filtered_index < end
    # JF_BAD_INDEX_FILE:   drop known bad filtered_index values
    scan_start = int(os.environ.get("JF_SCAN_START_INDEX", "0"))
    scan_end_env = os.environ.get("JF_SCAN_END_INDEX", "")
    scan_end = int(scan_end_env) if scan_end_env else None

    bad_index_file = os.environ.get("JF_BAD_INDEX_FILE", "")
    bad_indices = set()
    if bad_index_file:
        try:
            with open(bad_index_file, "r", encoding="utf-8") as f:
                bad_indices = {
                    int(line.strip())
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                }
        except FileNotFoundError:
            bad_indices = set()

    if scan_start > 0 or scan_end is not None or bad_indices:
        before_scan_n = len(trajectory_dataset)

        def _keep_scan_window(example):
            idx = int(example["filtered_index"])
            if idx < scan_start:
                return False
            if scan_end is not None and idx >= scan_end:
                return False
            if idx in bad_indices:
                return False
            return True

        trajectory_dataset = trajectory_dataset.filter(
            _keep_scan_window,
            desc=(
                f"Filtering scan window: start={scan_start}, "
                f"end={scan_end}, bad={len(bad_indices)}"
            ),
        )

        after_scan_n = len(trajectory_dataset)
        if accelerator.is_main_process:
            print(
                f"[SCAN FILTER] start={scan_start} end={scan_end} "
                f"bad={len(bad_indices)} kept={after_scan_n}/{before_scan_n}, "
                f"dropped={before_scan_n - after_scan_n}",
                flush=True,
            )
    data_module = make_jacobian_data_module(
        tokenizer=tokenizer,
        trajectory_data=trajectory_dataset,
        data_args=data_args,
        model=model_args.target_model_path,
        local_rank=training_args.local_rank,
    )

    # Build DataLoader here so we can pass it through accelerator.prepare
    per_device_bs = training_args.per_device_train_batch_size or 1  # from TrainingArguments
    train_dataloader = DataLoader(
        data_module["train_dataset"],
        batch_size=per_device_bs,
        shuffle=False,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Optimizer
    # Filter trainable params
    params = (p for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(params, lr=training_args.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / (training_args.gradient_accumulation_steps or 1)
    )

    if training_args.max_steps is not None and training_args.max_steps > 0:
        max_train_steps = training_args.max_steps
    else:
        max_train_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)

    lr_scheduler = transformers.get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.get_warmup_steps(max_train_steps),
        num_training_steps=max_train_steps,
    )

    # Accelerate + DeepSpeed wrapper
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # trainer
    # We already created an external Accelerator with DeepSpeedPlugin above.
    # If TrainingArguments.deepspeed remains set, HF Trainer will try to
    # create another DeepSpeed-enabled Accelerator inside super().__init__(),
    # which causes an Accelerate/DeepSpeed double-initialization error.
    training_args.deepspeed = None

    trainer = CllmTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        accelerator=accelerator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        teacher_model_path=model_args.teacher_model_path,
        use_keep_gate=training_args.use_keep_gate,
        keep_gate_hidden_size=training_args.keep_gate_hidden_size,
        lambda_keep=training_args.lambda_keep,
        keep_gate_detach_features=training_args.keep_gate_detach_features,
        use_teacher_kd=training_args.use_teacher_kd,
        lambda_kd=training_args.lambda_kd,
        kd_temperature=training_args.kd_temperature,
        kd_on_consistency_only=training_args.kd_on_consistency_only,
    )
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if accelerator.is_main_process:
        safe_save_model_for_hf_trainer(model=model, tokenizer=tokenizer,
                                       output_dir=training_args.output_dir,
                                       accelerator=accelerator)

if __name__ == "__main__":
    train()

