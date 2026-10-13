import torch
import wandb
from torch.cuda.amp import autocast
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import transformers.integrations.deepspeed as hfds_integration
import torch.nn.functional as F

import sys
from pathlib import Path
import os
import json
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from utils.hierarchical_targets import make_batch_anchor_mask
from torch.nn.attention.flex_attention import create_block_mask
import re
from functools import lru_cache

from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class CllmTrainer(Trainer):
    def __init__(
        self,
        *args,
        accelerator=None,
        optimizer=None,
        lr_scheduler=None,
        train_dataloader=None,
        teacher_model_path=None,
        use_keep_gate=False,
        keep_gate_hidden_size=256,
        lambda_keep=0.0,
        keep_gate_detach_features=True,
        use_teacher_kd=False,
        lambda_kd=0.0,
        kd_temperature=1.0,
        kd_on_consistency_only=True,
        **kwargs
    ):
        # This project already creates an external Accelerator/DeepSpeedPlugin
        # in the train script and passes prepared model/optimizer/dataloader
        # into CllmTrainer. HF Trainer also creates its own Accelerator inside
        # super().__init__(). If args.deepspeed is still set here, HF Trainer
        # tries to create a second DeepSpeed-enabled Accelerator and crashes.
        #
        # Keep the external Accelerator for actual training, but hide DeepSpeed
        # from the internal HF Trainer initialization.
        if "args" in kwargs and kwargs["args"] is not None:
            try:
                kwargs["args"].deepspeed = None
            except Exception:
                pass
            try:
                kwargs["args"].hf_deepspeed_config = None
            except Exception:
                pass
            try:
                kwargs["args"].deepspeed_plugin = None
            except Exception:
                pass

        # Also clear HF's global weakref so Trainer does not infer DeepSpeed
        # from the earlier HfDeepSpeedConfig used for partitioned loading.
        try:
            hfds_integration._hf_deepspeed_config_weak_ref = None
        except Exception:
            pass

        super().__init__(*args, **kwargs)
        args = kwargs["args"]

        self.accelerator = accelerator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader

        self.base_model = self.accelerator.unwrap_model(self.model)
        self.cfg = self.base_model.config

        self.train_step_cnt = 0

        self.max_new_tokens = args.max_new_tokens
        self.use_gt_labels = args.use_gt_labels
        self._blockmask_cache = {}

        # Attention backend for Jacobi training forward.
        #   flex  : original BlockMask + FlexAttention path
        #   dense : dense 4D additive mask, for eager/sdpa diagnostic fallback
        self.jf_attn_backend = os.environ.get("JF_ATTN_BACKEND", "flex").strip().lower()
        if self.args.local_rank in (-1, 0):
            print(f"[JF ATTN CONFIG] JF_ATTN_BACKEND={self.jf_attn_backend}", flush=True)
        
        # Anchor-consistency type: all | syntax | ctrlsyntax
        self.anchor_cons_type = getattr(args, "anchor_cons_type", "all")
        self._anchor_type_cache = {}
        
        # new fields
        self.teacher_model = None
        self.teacher_model_path = teacher_model_path
        self.use_keep_gate = use_keep_gate
        self.lambda_keep = lambda_keep
        self.keep_gate_detach_features = keep_gate_detach_features
        self.use_teacher_kd = use_teacher_kd
        self.lambda_kd = lambda_kd
        self.kd_temperature = kd_temperature
        self.kd_on_consistency_only = kd_on_consistency_only

        self.loss_log_path = os.path.join(args.output_dir, "loss_components.tsv")
        if args.local_rank in (-1, 0):
            os.makedirs(args.output_dir, exist_ok=True)
            with open(self.loss_log_path, "w") as f:
                f.write(
                    "step\tloss_ar\tloss_consistency\tloss_hard_anchor\tloss_contract\t"
                    "n_anchor_all\tn_anchor_used\tanchor_cons_type\t"
                    "anchor_cons_noise_gate\tmean_noise_ratio\t"
                    "conv_cons_mode\tconv_beta_eff\tmean_conv_score\tmean_cons_weight\t"
                    "loss_total\n"
                )
                
        hidden_size = getattr(self.cfg, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden_size from model config")

        keep_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
        self.keep_head = None
        if self.use_keep_gate:
            self.keep_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, keep_gate_hidden_size),
                torch.nn.GELU(),
                torch.nn.Linear(keep_gate_hidden_size, 1),
            ).to(device=self.accelerator.device, dtype=keep_dtype)

        if self.use_teacher_kd:
            if self.teacher_model_path is None:
                raise ValueError("use_teacher_kd=True but teacher_model_path is not provided")

            teacher_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

            # Temporarily disable HF DeepSpeed global config so teacher is loaded as a normal full model
            old_ds_ref = getattr(hfds_integration, "_hf_deepspeed_config_weak_ref", None)
            hfds_integration._hf_deepspeed_config_weak_ref = None
            try:
                self.teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.teacher_model_path,
                    cache_dir=self.args.cache_dir if hasattr(self.args, "cache_dir") else None,
                    attn_implementation="flex_attention",
                    torch_dtype=teacher_dtype,
                    low_cpu_mem_usage=False,
                )
            finally:
                hfds_integration._hf_deepspeed_config_weak_ref = old_ds_ref

            self.teacher_model = self.teacher_model.to(self.accelerator.device)
            self.teacher_model.config.use_cache = False
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

            # optional sanity check
            emb = self.teacher_model.get_input_embeddings().weight
            if emb.dim() != 2:
                raise RuntimeError(f"Teacher embedding weight has wrong shape: {tuple(emb.shape)}")
    
    def get_train_dataloader(self):
        """
        Debug scan mode:
        Force a deterministic sequential distributed sampler.

        With JF_SCAN_SEQUENTIAL=1 and DDP world_size=W:
            rank r sees dataset indices r, r+W, r+2W, ...

        This makes example_index scanning/resume reliable.
        """
        if os.environ.get("JF_SCAN_SEQUENTIAL", "0") != "1":
            return super().get_train_dataloader()

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
                drop_last=False,
            )
        else:
            sampler = SequentialSampler(train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )    

    def _anchor_mask_from_targets(self, target_ids):
        """
        target_ids: [K], -100 ignored
        returns:
          anchor_mask: [K] bool
          n_anchor: scalar tensor
        """
        valid = target_ids.ne(-100)
        if valid.sum() == 0:
            return torch.zeros_like(target_ids, dtype=torch.bool), torch.tensor(0, device=target_ids.device)

        labels_2d = target_ids.unsqueeze(0)
        anchor_mask = make_batch_anchor_mask(
            self.processing_class,
            labels_2d,
            ignore_index=-100,
        ).squeeze(0)

        anchor_mask = anchor_mask & valid
        return anchor_mask, anchor_mask.sum()
    
    def _decode_token_for_anchor_type(self, token_id: int) -> str:
        """Cache tokenizer.decode([token_id]) because this is called many times."""
        if token_id in self._anchor_type_cache:
            return self._anchor_type_cache[token_id]

        try:
            s = self.processing_class.decode([token_id], skip_special_tokens=False)
        except Exception:
            s = ""

        self._anchor_type_cache[token_id] = s
        return s

    def _classify_anchor_token(self, token_id: int):
        """
        Return:
          is_syntax: bool
          is_control: bool

        This is intentionally conservative.
        Syntax anchors target code formatting / grammar stability.
        Control anchors target program skeleton stability.
        """
        s = self._decode_token_for_anchor_type(token_id)
        s_strip = s.strip()

        # Special tokens should not become syntax/control anchors.
        if s_strip == "":
            return False, False

        # Syntax / formatting-critical tokens.
        # Keep this broad enough for BPE tokens that include whitespace.
        syntax_chars = [
            ":", ",", ".", ";",
            "(", ")", "[", "]", "{", "}",
            "\"", "'", "`",
        ]

        has_newline = ("\n" in s)
        has_indent_like = ("\t" in s) or ("    " in s)
        has_code_fence = "```" in s
        has_syntax_char = any(ch in s for ch in syntax_chars)

        is_syntax = bool(has_newline or has_indent_like or has_code_fence or has_syntax_char)

        # Python control-flow / program-structure tokens.
        control_words = {
            "def", "class",
            "return", "yield",
            "if", "elif", "else",
            "for", "while",
            "try", "except", "finally",
            "with",
            "break", "continue", "pass",
            "raise", "assert",
            "import", "from",
            "lambda",
        }

        # Decode can include BPE whitespace markers or leading spaces.
        # strip() catches tokens like " return".
        is_control = s_strip in control_words

        return is_syntax, is_control

    def _norm_anchor_piece(self, piece: str) -> str:
        return (
            piece.replace("Ġ", "")
                 .replace("▁", "")
                 .replace("Ċ", "\n")
                 .strip()
        )

    def _is_math_anchor_token(self, token_id: int) -> bool:
        """
        Math-specific anchor predicate.

        Unlike syntax/ctrlsyntax, this should NOT merely filter the existing
        code-anchor mask. It should directly decide from the target token,
        because math-critical tokens such as frac/times/div/boxed may not be
        included in the original code-anchor rule.
        """
        s = self._decode_token_for_anchor_type(token_id)
        p = self._norm_anchor_piece(s)

        if p == "":
            return False

        # Avoid special/padding/artifact tokens.
        if p.startswith("<|") and p.endswith("|>"):
            return False
        if "using" in p and "\ufeff" in p:
            return False
        if p.startswith("using") or "\ufeffusing" in p:
            return False
        if p in {"<|endoftext|>", "<|im_end|>", "<|im_start|>"}:
            return False

        # Numeric constants: 12, -3, 30.1, .5 style variants.
        if re.fullmatch(r"-?(\d+(\.\d+)?|\.\d+)", p):
            return True

        # If the BPE piece contains any digit, it is usually math-critical:
        # e.g., " 29", "30.1", "}{6".
        if any(ch.isdigit() for ch in p):
            return True

        # Exact operators / relation symbols.
        math_exact = {
            "+", "-", "*", "/", "//", "%", "**",
            "=", "==", "!=", "<", "<=", ">", ">=",
            "≤", "≥", "≠", "≈",
            "^", "_",
        }
        if p in math_exact:
            return True

        # Delimiters important for LaTeX/math expressions and boxed answers.
        # Keep colon because many solutions introduce equations with ":\n".
        math_delims = {
            "(", ")", "[", "]", "{", "}",
            "\\(", "\\)", "\\[", "\\]",
            ":", ",", ".",
        }
        if p in math_delims:
            return True

        # BPE often merges delimiters with slashes/newlines, e.g. "}[", "}.", "):".
        if any(x in p for x in ["\\[", "\\]", "\\(", "\\)", "{", "}", "[", "]"]):
            return True

        # LaTeX/math command pieces. In Qwen tokenization, "\\frac" often appears
        # as "\\", "frac" or "d", "frac", so include bare command names too.
        math_words = {
            "frac", "dfrac", "sqrt", "boxed",
            "times", "div", "cdot", "leq", "geq", "neq", "approx",
            "hline", "begin", "end", "array",
        }
        if p in math_words:
            return True

        # Some pieces include leading backslash or merged command text.
        latex_substrings = [
            "\\frac", "\\dfrac", "\\sqrt", "\\boxed",
            "\\times", "\\div", "\\cdot",
            "\\leq", "\\geq", "\\neq", "\\approx",
            "\\begin", "\\end",
        ]
        if any(x in p for x in latex_substrings):
            return True

        # Final-answer / conclusion markers. Keep this small to avoid anchoring
        # ordinary explanation words too much.
        final_markers = {
            "answer", "Answer",
            "final", "Final",
            "Therefore", "therefore",
            "Thus", "thus",
            "Hence", "hence",
            "So", "so",
        }
        if p in final_markers:
            return True

        return False

    def _math_anchor_mask_from_targets(self, target_ids):
        """
        Build math-specific anchor mask directly from target_ids.
        This is not a filter over the original code-anchor mask.
        """
        valid = target_ids.ne(-100)
        if valid.sum() == 0:
            return torch.zeros_like(target_ids, dtype=torch.bool), torch.tensor(0, device=target_ids.device)

        anchor_mask = torch.zeros_like(target_ids, dtype=torch.bool)

        valid_positions = torch.nonzero(valid, as_tuple=False).flatten()
        for pos in valid_positions.tolist():
            tid = int(target_ids[pos].detach().cpu())
            if tid < 0:
                continue
            if self._is_math_anchor_token(tid):
                anchor_mask[pos] = True

        anchor_mask = anchor_mask & valid
        return anchor_mask, anchor_mask.sum()

    def _is_code_completion_anchor_token(self, tid: int) -> bool:
        """
        Completion-critical code anchor predicate.

        Targets tokens that often determine whether short MBPP-style
        functions are executable and complete:
          - return/yield/import/control-flow tokens
          - common library/module/function-call tokens
          - regex/math-specific pieces
          - closing delimiters / quotes
        """
        key = ("code_completion", int(tid))
        if key in self._anchor_type_cache:
            return self._anchor_type_cache[key]

        try:
            piece = self.processing_class.decode([int(tid)], skip_special_tokens=False)
        except Exception:
            self._anchor_type_cache[key] = False
            return False

        raw = piece
        s = piece.strip()
        low = s.lower()

        if not s:
            self._anchor_type_cache[key] = False
            return False

        code_keywords = {
            "return", "yield",
            "import", "from", "as",
            "def", "lambda",
            "if", "elif", "else",
            "for", "while",
            "try", "except", "finally",
            "with",
            "assert", "raise",
            "break", "continue",
        }

        common_modules = {
            "re", "math", "heapq", "hq",
            "itertools", "collections", "functools", "operator",
            "random", "bisect", "string",
            "numpy", "np",
        }

        common_calls = {
            "search", "match", "findall", "split", "sub", "compile",
            "sqrt", "pi", "ceil", "floor", "gcd", "factorial",
            "sin", "cos", "tan", "log", "exp",
            "nlargest", "nsmallest",
            "len", "range", "list", "tuple", "set", "dict",
            "str", "int", "float", "bool",
            "sum", "all", "any", "max", "min", "abs",
            "sorted", "enumerate", "zip", "map", "filter",
            "append", "extend", "insert", "remove",
            "count", "index", "find", "replace",
            "join", "lower", "upper", "strip",
            "items", "keys", "values",
        }

        # Keep this simple to avoid quote-escaping issues.
        closing_tokens = {
            ")", "]", "}",
            "'))", "\")", "']", "\"]",
            "'", "\"",
        }

        keep = False

        if low in code_keywords or low in common_modules or low in common_calls:
            keep = True

        if s in closing_tokens:
            keep = True

        # Attribute-call fragments such as ".search", ".split", ".pi", ".nlargest".
        if low.startswith(".") and low[1:] in (common_calls | common_modules):
            keep = True

        # Regex literal fragments. MBPP has many regex tasks where missing one
        # argument / quote / escape makes otherwise simple functions fail.
        if "\\" in raw and any(ch in raw for ch in ["w", "d", "s", "n", "*", "+", "^", "$", "[", "]", "(", ")", "|"]):
            keep = True

        # Power/exponent token is critical for area/volume/formula tasks.
        if s in {"**", "//"}:
            keep = True

        # Common punctuation that completes expressions.
        if s in {".", "="}:
            keep = True

        self._anchor_type_cache[key] = bool(keep)
        return bool(keep)

    def _code_completion_anchor_mask_from_targets(self, target_ids):
        """
        Build code-completion-specific anchor mask directly from target_ids.
        """
        valid = target_ids.ne(-100)
        if valid.sum() == 0:
            return torch.zeros_like(target_ids, dtype=torch.bool), torch.tensor(0, device=target_ids.device)

        anchor_mask = torch.zeros_like(target_ids, dtype=torch.bool)
        valid_positions = torch.nonzero(valid, as_tuple=False).flatten()

        for pos in valid_positions.tolist():
            tid = int(target_ids[pos].detach().cpu())
            if tid < 0:
                continue
            if self._is_code_completion_anchor_token(tid):
                anchor_mask[pos] = True

        anchor_mask = anchor_mask & valid
        return anchor_mask, anchor_mask.sum()

    def _auto_influence_anchor_mask_from_targets(self, target_ids):
        """
        Automatic anchor mask from an offline-discovered token-ID vocabulary.

        The vocabulary is discovered from training data only using
        counterfactual downstream influence. This path does NOT call
        make_batch_anchor_mask and is therefore independent of the
        hand-written semantic-anchor rules.

        target_ids: [K], where -100 marks ignored positions.
        """
        valid = target_ids.ne(-100)

        vocab_path = str(
            getattr(
                self.args,
                "anchor_auto_vocab_json",
                "",
            )
        ).strip()

        if not vocab_path:
            raise ValueError(
                "anchor_cons_type=auto_influence requires "
                "--anchor_auto_vocab_json"
            )

        cached_path = getattr(
            self,
            "_auto_anchor_vocab_cached_path",
            None,
        )

        if (
            cached_path != vocab_path
            or not hasattr(self, "_auto_anchor_token_ids_cpu")
        ):
            import json

            with open(vocab_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            token_ids = payload.get("token_ids", None)

            if token_ids is None:
                raise ValueError(
                    f"{vocab_path} does not contain 'token_ids'"
                )

            token_ids = sorted(
                {
                    int(x)
                    for x in token_ids
                }
            )

            if not token_ids:
                raise ValueError(
                    f"No token IDs found in {vocab_path}"
                )

            self._auto_anchor_vocab_cached_path = vocab_path

            self._auto_anchor_token_ids_cpu = torch.tensor(
                token_ids,
                dtype=torch.long,
                device="cpu",
            )

            print(
                "[AUTO-ANCHOR] loaded",
                len(token_ids),
                "token IDs from",
                vocab_path,
            )

        auto_ids = self._auto_anchor_token_ids_cpu.to(
            device=target_ids.device,
            non_blocking=True,
        )

        mask = (
            valid
            & torch.isin(
                target_ids,
                auto_ids,
            )
        )

        return mask, mask.sum()

    def _filter_anchor_mask_by_type(self, target_ids, anchor_mask, anchor_cons_type: str):
        """
        target_ids: [K], -100 ignored
        anchor_mask: [K] bool from original make_batch_anchor_mask
        anchor_cons_type:
          - all: original anchor_mask
          - syntax: syntax/format anchors only
          - ctrlsyntax: syntax + control-flow anchors
        """
        if anchor_cons_type == "all":
            return anchor_mask

        if anchor_cons_type not in ("syntax", "ctrlsyntax"):
            raise ValueError(
                f"Unknown anchor_cons_type={anchor_cons_type}. "
                "Expected one of: all, syntax, ctrlsyntax, math, math_final, math_plus_final, code_completion, all_plus_completion"
            )

        valid_anchor_positions = torch.nonzero(anchor_mask, as_tuple=False).flatten()
        if valid_anchor_positions.numel() == 0:
            return anchor_mask

        filtered = torch.zeros_like(anchor_mask, dtype=torch.bool)

        # Number of anchor positions is usually small, so Python loop is okay.
        for pos in valid_anchor_positions.tolist():
            tid = int(target_ids[pos].detach().cpu())
            if tid < 0:
                continue

            is_syntax, is_control = self._classify_anchor_token(tid)

            if anchor_cons_type == "syntax":
                keep = is_syntax
            elif anchor_cons_type == "ctrlsyntax":
                keep = is_syntax or is_control
            else:
                keep = True

            if keep:
                filtered[pos] = True

        return filtered
    
    def _anchor_ce_loss(self, logits, target_ids, anchor_weight: float = 1.0):
        """
        logits: [K, V]
        target_ids: [K]
        CE only on anchor target positions.
        """
        valid = target_ids.ne(-100)
        if valid.sum() == 0:
            return torch.zeros((), device=logits.device), torch.tensor(0, device=logits.device)
        
        labels_2d = target_ids.unsqueeze(0)
        anchor_mask = make_batch_anchor_mask(
            self.processing_class,
            labels_2d,
            ignore_index=-100,
        ).squeeze(0)

        anchor_mask = anchor_mask & valid

        if anchor_mask.sum() == 0:
            return torch.zeros((), device=logits.device), torch.tensor(0, device=logits.device)

        loss = F.cross_entropy(
            logits[anchor_mask].float(),
            target_ids[anchor_mask],
            reduction="mean",
            label_smoothing=0.0,
            ignore_index=-100,
        ) * anchor_weight

        return loss, anchor_mask.sum()
    # Utilities
    @staticmethod
    def _to_int(x):
        return x.item() if isinstance(x, torch.Tensor) else int(x)

    def _unpack_sample(self, inputs):
        """
        Extract a single sample. (Assumes per_device_train_batch_size == 1.)
        Required keys:
          - input_ids: [1, L]
          - prompt_ids_len: scalar or [1]
          - T: length of traj_position_indices (last uncorrupted token positions) in [1, T]
        """
        # TODO: support bsz > 1 uppacking
        input_ids = inputs["input_ids"][0]
        example_index = inputs.get("example_index", None)
        prompt_len = inputs["prompt_ids_len"]
        if isinstance(prompt_len, torch.Tensor):
            if prompt_len.dim() > 0:
                prompt_len = prompt_len[0]
        prompt_len = self._to_int(prompt_len)

        traj_position_indices = inputs["traj_position_indices"][0][0]
        traj_position_indices = [int(u) for u in traj_position_indices]
        T = len(traj_position_indices)

        return (
            input_ids.to(self.args.device),
            prompt_len,
            T,
            example_index,
        )
            
    def _duplicate_prefix_mask(self, input_ids: torch.Tensor, prompt_len: int, T: int) -> torch.Tensor:
        """
        [L] bool: True where token should be masked because it's in k_j's prefix
        identical to last_j (from left to right until first divergence).
        Only k_j tokens are masked; last_j tokens are not.
        """
        device = input_ids.device
        N = self.max_new_tokens
        L = input_ids.size(0)
        mask = torch.zeros(L, dtype=torch.bool, device=device)

        k_starts, l_starts = self._index_layout(prompt_len, T, N)
        for j in range(T):
            ks = k_starts[j]
            ls = l_starts[j]
            k_block = input_ids[ks:ks + N]
            l_block = input_ids[ls:ls + N]

            eq = (k_block == l_block)
            # find first index where they differ
            if torch.any(~eq):
                first_diff = int(torch.nonzero(~eq, as_tuple=False)[0])
            else:
                # fully identical: mask the whole k_j block
                first_diff = N

            if first_diff > 0:
                mask[ks:ks + first_diff] = True

        return mask

    def _build_padding_mask_for_loss(self, input_ids: torch.Tensor, prompt_len: int, T: int) -> torch.Tensor:
        """
        [L] bool padding mask used for losses:
        True = mask, False = keep.
        Combines: (a) PAD tokens, (b) duplicate-prefix in each k_j vs last_j.
        """
        device = input_ids.device
        pad_id = getattr(self.processing_class, "pad_token_id", None)

        mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
        if pad_id is not None:
            mask |= (input_ids == pad_id)

        mask |= self._duplicate_prefix_mask(input_ids, prompt_len, T)
        return mask

    def _block_keep_mask_divergence_and_eos(
        self,
        input_ids: torch.Tensor,
        k_start: int,
        l_start: int,
        N: int,
        eos_id: int | None,
        drop_last_offset: bool = False,
    ) -> torch.Tensor:
        """
        Returns [N-1] bool if drop_last_offset else [N] bool.
        True => keep offset t for logits at position (start+t) predicting next token (t+1).
        We: start computing loss after first divergence.
        """
        device = input_ids.device
        size = N - 1 if drop_last_offset else N
        offs = torch.arange(size, device=device)

        k_block = input_ids[k_start : k_start + N]
        l_block = input_ids[l_start : l_start + N]

        # Divergence mask (pairwise): keep from first differing offset onward
        diff = (k_block[:size] != l_block[:size])
        if diff.any():
            first_diff = int(torch.nonzero(diff, as_tuple=False)[0])
            div_keep = offs >= first_diff
        else:
            div_keep = torch.zeros(size, dtype=torch.bool, device=device)

        # UNUSED — EOS mask: keep offsets t such that (t+1) is BEFORE EOS in both blocks
        #def keep_next_before_eos(block):
        #    if eos_id is None:
        #        return torch.ones(size, dtype=torch.bool, device=device)
        #    pos = torch.nonzero(block == eos_id, as_tuple=False)
        #    if pos.numel() == 0:
        #        return torch.ones(size, dtype=torch.bool, device=device)
        #    e = int(pos[0])          # EOS index within [0..N-1]
        #    return offs < e

        #eos_keep = keep_next_before_eos(k_block) & keep_next_before_eos(l_block)

        return div_keep

    @staticmethod
    def _index_layout(prompt_len: int, T: int, N):
        """Return lists of start indices for all k_j and last_j blocks in flattened sequence."""
        k_starts = [prompt_len + 2 * j * N for j in range(T)]
        l_starts = [prompt_len + (2 * j + 1) * N for j in range(T)]
        return k_starts, l_starts

    def _build_shared_position_ids(self, L: int, prompt_len: int, T: int):
        """
        Build [L] position_ids so each (k_j, last_j) share the same positions.
        Prompt: 0...prompt_len-1
        For each j: both blocks use prompt_len + j*N .. prompt_len + j*N + (N-1)
        """
        device = self.args.device
        N = self.max_new_tokens
        pos = torch.empty(L, dtype=torch.long, device=device)

        # Prompt positions
        pos[:prompt_len] = torch.arange(prompt_len, device=device)

        # Pair-shared positions
        k_starts, l_starts = self._index_layout(prompt_len, T, N)
        rel = torch.arange(N, device=device)
        for j in range(T):
            base = prompt_len + j * N
            ks = k_starts[j]
            ls = l_starts[j]
            pos[ks:ks + N] = base + rel
            pos[ls:ls + N] = base + rel

        return pos
    
    def _flip_block_after_eos_to_pad(self, input_ids: torch.Tensor, start: int, N: int, eos_id: int | None, pad_id: int | None) -> int:
        """Mutate input_ids[start:start+N] so that tokens AFTER first EOS become PAD.
        Returns number of tokens flipped."""
        if eos_id is None or pad_id is None:
            return 0
        block = input_ids[start:start+N]
        pos = (block == eos_id).nonzero(as_tuple=False)
        if pos.numel() == 0:
            return 0
        k = int(pos[0])                  # eos offset inside block
        flip_start = start + k + 1
        flip_end   = start + N
        if flip_start < flip_end:
            input_ids[flip_start:flip_end] = pad_id
            return flip_end - flip_start
        return 0
    # ------------------------------------------------------------------
    # NaN / Inf diagnostic helpers
    # Enable with:
    #   export JF_DEBUG_NAN=1
    # Optional:
    #   export JF_DEBUG_PARAM_SCAN=1
    # ------------------------------------------------------------------
    def _jf_debug_enabled(self):
        return os.environ.get("JF_DEBUG_NAN", "0") == "1"

    def _jf_debug_dir(self):
        d = os.path.join(str(self.args.output_dir), "nonfinite_debug")
        os.makedirs(d, exist_ok=True)
        return d

    def _jf_rank_info(self):
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        step = int(getattr(self.state, "global_step", -1))
        return rank, local_rank, step

    def _jf_tensor_stats(self, x):
        if not torch.is_tensor(x):
            return {"type": str(type(x))}
        obj = {
            "shape": tuple(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
            "numel": int(x.numel()),
        }
        if x.numel() == 0:
            return obj

        if torch.is_floating_point(x):
            finite = torch.isfinite(x)
            obj["finite"] = int(finite.sum().detach().cpu())
            obj["nonfinite"] = int((~finite).sum().detach().cpu())
            if finite.any():
                xf = x[finite].detach().float()
                obj["min_finite"] = float(xf.min().cpu())
                obj["max_finite"] = float(xf.max().cpu())
                obj["mean_finite"] = float(xf.mean().cpu())
        else:
            obj["min"] = int(x.detach().min().cpu())
            obj["max"] = int(x.detach().max().cpu())
        return obj

    def _jf_save_debug(self, where, tensors=None, scalars=None, extra=None, raise_error=True):
        if tensors is None:
            tensors = {}
        if scalars is None:
            scalars = {}
        if extra is None:
            extra = {}

        rank, local_rank, step = self._jf_rank_info()
        debug_dir = self._jf_debug_dir()
        base = f"{where}_rank{rank}_local{local_rank}_step{step}"

        meta = {
            "where": where,
            "rank": rank,
            "local_rank": local_rank,
            "step": step,
            "scalars": {},
            "tensor_stats": {},
            "extra": extra,
        }

        save_obj = {
            "where": where,
            "rank": rank,
            "local_rank": local_rank,
            "step": step,
            "scalars": {},
            "tensors": {},
            "extra": extra,
        }

        for k, v in scalars.items():
            try:
                if torch.is_tensor(v):
                    vv = v.detach().float()
                    meta["scalars"][k] = float(vv.cpu()) if vv.numel() == 1 else str(tuple(vv.shape))
                    save_obj["scalars"][k] = vv.cpu()
                else:
                    meta["scalars"][k] = float(v) if isinstance(v, (int, float)) else str(v)
                    save_obj["scalars"][k] = v
            except Exception as e:
                meta["scalars"][k] = f"<failed: {e}>"
                save_obj["scalars"][k] = str(v)

        for k, v in tensors.items():
            try:
                meta["tensor_stats"][k] = self._jf_tensor_stats(v)
                if torch.is_tensor(v):
                    save_obj["tensors"][k] = v.detach().cpu()
                else:
                    save_obj["tensors"][k] = v
            except Exception as e:
                meta["tensor_stats"][k] = {"error": str(e)}
                save_obj["tensors"][k] = str(type(v))

        pt_path = os.path.join(debug_dir, base + ".pt")
        json_path = os.path.join(debug_dir, base + ".json")

        torch.save(save_obj, pt_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(
            f"[FATAL/DEBUG] {where} at step={step}, rank={rank}, local_rank={local_rank}. "
            f"Saved {pt_path} and {json_path}",
            flush=True,
        )

        if raise_error:
            raise FloatingPointError(f"{where} at step={step}. Debug saved to {pt_path}")

        return pt_path, json_path

    def _jf_check_finite(self, name, x, tensors=None, scalars=None, extra=None):
        """
        Returns:
            True  if finite or debug disabled
            False if non-finite was detected and saved in scan/skip mode
        """
        if not self._jf_debug_enabled():
            return True
        if not torch.is_tensor(x):
            return True
        if not torch.is_floating_point(x):
            return True

        if not torch.isfinite(x).all():
            dbg_tensors = {} if tensors is None else dict(tensors)
            dbg_tensors[name] = x

            skip_mode = os.environ.get("JF_SKIP_NONFINITE_FORWARD", "0") == "1"

            self._jf_save_debug(
                where=f"nonfinite_{name}",
                tensors=dbg_tensors,
                scalars=scalars,
                extra={
                    **(extra or {}),
                    "skip_mode": skip_mode,
                },
                raise_error=not skip_mode,
            )

            return False

        return True
    
    def _jf_check_int_range(self, name, x, low, high, tensors=None, scalars=None, extra=None):
        if not self._jf_debug_enabled():
            return
        if not torch.is_tensor(x):
            return
        bad = (x < low) | (x >= high)
        if bad.any():
            dbg_tensors = {} if tensors is None else dict(tensors)
            dbg_tensors[name] = x
            dbg_tensors[f"{name}_bad_mask"] = bad
            self._jf_save_debug(
                where=f"bad_range_{name}",
                tensors=dbg_tensors,
                scalars=scalars,
                extra={**(extra or {}), "low": low, "high": high},
                raise_error=True,
            )

    def _jf_check_some_params(self, model, input_ids=None):
        if not self._jf_debug_enabled():
            return
        # Full param scan is expensive. Use JF_DEBUG_PARAM_SCAN=1 only when needed.
        full_scan = os.environ.get("JF_DEBUG_PARAM_SCAN", "0") == "1"
        step = int(getattr(self.state, "global_step", 0))
        if (not full_scan) and step % 10 != 0:
            return

        bad = []
        checked = 0
        max_check = 999999999 if full_scan else 16

        for name, p in model.named_parameters():
            if p is None or p.data is None:
                continue
            if not p.requires_grad:
                continue
            checked += 1
            if torch.is_floating_point(p.data) and not torch.isfinite(p.data).all():
                bad.append(name)
                if len(bad) >= 10:
                    break
            if checked >= max_check:
                break

        if bad:
            tensors = {}
            if input_ids is not None:
                tensors["input_ids"] = input_ids
            self._jf_save_debug(
                where="nonfinite_params_before_forward",
                tensors=tensors,
                scalars={"checked_params": checked},
                extra={"bad_params": bad, "full_scan": full_scan},
                raise_error=True,
            )
    def soft_cross_entropy(self, predicts, targets, padding_mask):
        predicts = predicts.float()
        targets = targets.float()

        if (~padding_mask).sum() == 0:
            return 0 * predicts[0][0]

        if self._jf_debug_enabled():
            self._jf_check_finite(
                "soft_ce_predicts_input",
                predicts,
                tensors={"predicts": predicts, "targets": targets, "padding_mask": padding_mask},
                extra={"stage": "soft_cross_entropy_input"},
            )
            self._jf_check_finite(
                "soft_ce_targets_input",
                targets,
                tensors={"predicts": predicts, "targets": targets, "padding_mask": padding_mask},
                extra={"stage": "soft_cross_entropy_input"},
            )

        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)

        if self._jf_debug_enabled():
            self._jf_check_finite(
                "soft_ce_predict_log_prob",
                predict_log_prob,
                tensors={
                    "predicts": predicts,
                    "targets": targets,
                    "predict_log_prob": predict_log_prob,
                    "targets_prob": targets_prob,
                    "padding_mask": padding_mask,
                },
                extra={"stage": "soft_cross_entropy_after_log_softmax"},
            )
            self._jf_check_finite(
                "soft_ce_targets_prob",
                targets_prob,
                tensors={
                    "predicts": predicts,
                    "targets": targets,
                    "predict_log_prob": predict_log_prob,
                    "targets_prob": targets_prob,
                    "padding_mask": padding_mask,
                },
                extra={"stage": "soft_cross_entropy_after_softmax"},
            )

        entropy = -targets_prob * predict_log_prob
        expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
        entropy = entropy.masked_fill(expand_mask, 0)
        mean_entropy = entropy.sum() / (~padding_mask).sum()

        if self._jf_debug_enabled():
            self._jf_check_finite(
                "soft_ce_mean_entropy",
                mean_entropy,
                tensors={"entropy": entropy, "padding_mask": padding_mask},
                scalars={"valid_count": (~padding_mask).sum()},
                extra={"stage": "soft_cross_entropy_output"},
            )

        return mean_entropy

    # FlexAttention BlockMask
    # - prompt queries: causal within prompt
    # - k_j queries: causal within *their own* k_j block + prompt
    # - last_j queries: causal within *their own* last_j block + prompt
    def soft_cross_entropy_weighted(self, predicts, targets, padding_mask, token_weights=None):
        predicts = predicts.float()
        targets = targets.float()

        valid = ~padding_mask
        if valid.sum() == 0:
            return 0 * predicts[0][0]

        if self._jf_debug_enabled():
            self._jf_check_finite(
                "weighted_soft_ce_predicts_input",
                predicts,
                tensors={"predicts": predicts, "targets": targets, "padding_mask": padding_mask},
                extra={"stage": "weighted_soft_cross_entropy_input"},
            )
            self._jf_check_finite(
                "weighted_soft_ce_targets_input",
                targets,
                tensors={"predicts": predicts, "targets": targets, "padding_mask": padding_mask},
                extra={"stage": "weighted_soft_cross_entropy_input"},
            )

        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)

        if self._jf_debug_enabled():
            self._jf_check_finite(
                "weighted_soft_ce_predict_log_prob",
                predict_log_prob,
                tensors={
                    "predicts": predicts,
                    "targets": targets,
                    "predict_log_prob": predict_log_prob,
                    "targets_prob": targets_prob,
                    "padding_mask": padding_mask,
                    "token_weights": token_weights if token_weights is not None else torch.empty(0, device=predicts.device),
                },
                extra={"stage": "weighted_soft_cross_entropy_after_log_softmax"},
            )
            self._jf_check_finite(
                "weighted_soft_ce_targets_prob",
                targets_prob,
                tensors={
                    "predicts": predicts,
                    "targets": targets,
                    "predict_log_prob": predict_log_prob,
                    "targets_prob": targets_prob,
                    "padding_mask": padding_mask,
                    "token_weights": token_weights if token_weights is not None else torch.empty(0, device=predicts.device),
                },
                extra={"stage": "weighted_soft_cross_entropy_after_softmax"},
            )

        per_token_entropy = (-targets_prob * predict_log_prob).sum(dim=-1)

        if token_weights is None:
            token_weights = torch.ones_like(per_token_entropy, device=per_token_entropy.device)

        token_weights = token_weights.to(per_token_entropy.device).float()
        token_weights = token_weights.masked_fill(padding_mask, 0.0)

        denom = token_weights.sum().clamp_min(1.0)
        out = (per_token_entropy.float() * token_weights).sum() / denom

        if self._jf_debug_enabled():
            self._jf_check_finite(
                "weighted_soft_ce_output",
                out,
                tensors={
                    "per_token_entropy": per_token_entropy,
                    "token_weights": token_weights,
                    "padding_mask": padding_mask,
                },
                scalars={"denom": denom, "valid_count": valid.sum()},
                extra={"stage": "weighted_soft_cross_entropy_output"},
            )

        return out

    def _find_last_braced_content_span(self, text: str, command: str):
        """
        Find the content span of the last LaTeX command like \\boxed{...} or \\fbox{...}.

        Returns:
            (content_start, content_end) in character offsets, excluding braces,
            or None if not found / malformed.
        """
        idx = text.rfind(command)
        if idx < 0:
            return None

        brace = text.find("{", idx + len(command))
        if brace < 0:
            return None

        depth = 0
        for j in range(brace, len(text)):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return (brace + 1, j)

        return None

    def _find_math_final_char_span(self, text: str):
        """
        Heuristic final-answer span detector for math trajectories.

        Priority:
          1. Last \\boxed{...} / \\fbox{...} content.
          2. Last answer marker followed by a compact expression.
          3. Last numeric / simple fraction fallback.

        Returns:
            (start, end) char span in text, or None.
        """
        if not text:
            return None

        # 1) Boxed answer, aligned with MATH500-style evaluation.
        candidates = []
        for cmd in ("\\\\boxed", "\\\\fbox"):
            span = self._find_last_braced_content_span(text, cmd)
            if span is not None:
                candidates.append(span)
        if candidates:
            # choose the latest span
            return max(candidates, key=lambda x: x[0])

        # 2) Answer markers. Search from the end and then extract a compact answer expression.
        marker_patterns = [
            r"####",
            r'"answer"\s*:\s*"?',
            r"(?i)final\s+answer\s*(?:is|=|:)?\s*",
            r"(?i)answer\s*(?:is|=|:)\s*",
            r"(?i)therefore[,]?\s*(?:the\s*)?answer\s*(?:is|=|:)?\s*",
            r"(?i)thus[,]?\s*(?:the\s*)?answer\s*(?:is|=|:)?\s*",
            r"(?i)hence[,]?\s*(?:the\s*)?answer\s*(?:is|=|:)?\s*",
        ]

        best_marker_end = -1
        for pat in marker_patterns:
            try:
                matches = list(re.finditer(pat, text, flags=re.DOTALL))
            except TypeError:
                matches = list(re.finditer(pat, text))
            if matches:
                m = matches[-1]
                if m.end() > best_marker_end:
                    best_marker_end = m.end()

        if best_marker_end >= 0:
            tail = text[best_marker_end: best_marker_end + 256]

            # Prefer explicit LaTeX-ish or numeric expressions after the marker.
            expr_patterns = [
                r"\$([^$]{1,120})\$",
                r"(-?\d+(?:,\d{3})*(?:\.\d+)?\s*/\s*-?\d+(?:,\d{3})*(?:\.\d+)?)",
                r"(-?\d+(?:,\d{3})*(?:\.\d+)?)",
                r"([A-Za-z]?\s*=\s*-?\d+(?:\.\d+)?)",
            ]

            for ep in expr_patterns:
                m = re.search(ep, tail)
                if m:
                    # If regex has a capture group, use group 1 span.
                    if m.lastindex:
                        s, e = m.span(1)
                    else:
                        s, e = m.span(0)
                    return (best_marker_end + s, best_marker_end + e)

        # 3) Last numeric/fraction fallback over the full text.
        num_pat = r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*/\s*-?\d+(?:,\d{3})*(?:\.\d+)?)?"
        nums = list(re.finditer(num_pat, text))
        if nums:
            m = nums[-1]
            return m.span(0)

        return None

    def _math_final_anchor_mask_from_context(
        self,
        input_ids,
        target_positions,
        padding_mask=None,
        buffer_tokens: int = 1,
    ):
        """
        Sequence-level final-answer-aware math anchor.

        input_ids:
            Full flattened training sequence, shape [L].
        target_positions:
            Consistency teacher target positions, shape [K].
        padding_mask:
            Bool mask over target positions, shape [K], True = ignore.

        Returns:
            anchor_mask: [K] bool
            n_anchor: scalar tensor

        This intentionally differs from broad math anchors:
        it anchors only target positions whose decoded token pieces overlap
        the final answer span, e.g., \\boxed{...}, Answer: ..., #### ..., or
        the last numeric fallback.
        """
        device = target_positions.device
        K = int(target_positions.numel())

        if K == 0:
            z = torch.zeros((0,), dtype=torch.bool, device=device)
            return z, torch.tensor(0, device=device)

        anchor_mask = torch.zeros((K,), dtype=torch.bool, device=device)

        try:
            ids_cpu = input_ids.detach().cpu().tolist()
            tp_cpu = target_positions.detach().cpu().tolist()
        except Exception:
            return anchor_mask, anchor_mask.sum()

        # Build full decoded text and char spans per original token position.
        pieces = []
        spans = []
        cursor = 0

        for tid in ids_cpu:
            try:
                piece = self.processing_class.decode([int(tid)], skip_special_tokens=False)
            except Exception:
                piece = ""
            start = cursor
            cursor += len(piece)
            end = cursor
            pieces.append(piece)
            spans.append((start, end))

        text = "".join(pieces)
        final_span = self._find_math_final_char_span(text)

        if final_span is None:
            return anchor_mask, anchor_mask.sum()

        fs, fe = final_span

        # Mark target positions whose token char-span overlaps final-answer span.
        for i, pos in enumerate(tp_cpu):
            if pos is None:
                continue
            try:
                pos_i = int(pos)
            except Exception:
                continue

            if pos_i < 0 or pos_i >= len(spans):
                continue

            ts, te = spans[pos_i]
            # Overlap check. Include zero-length guard.
            if te > fs and ts < fe:
                anchor_mask[i] = True

        # Optional small buffer in target-index space.
        # This helps include surrounding BPE fragments / quote / punctuation around final answer.
        if buffer_tokens > 0 and anchor_mask.any():
            expanded = anchor_mask.clone()
            idxs = torch.nonzero(anchor_mask, as_tuple=False).flatten()
            for idx in idxs.tolist():
                lo = max(0, idx - int(buffer_tokens))
                hi = min(K, idx + int(buffer_tokens) + 1)
                expanded[lo:hi] = True
            anchor_mask = expanded

        if padding_mask is not None:
            anchor_mask = anchor_mask & (~padding_mask.bool())

        return anchor_mask, anchor_mask.sum()

    def _per_token_kl_from_logits(self, student_logits, teacher_logits):
        """
        student_logits: [K, V]
        teacher_logits: [K, V], detached outside or inside caller
        returns: [K] KL(teacher || student)
        """
        student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
        teacher_probs = F.softmax(teacher_logits.float(), dim=-1)

        return F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)

    def _trajectory_contraction_loss(
        self,
        residual_vec,
        traj_j_vec,
        traj_off_vec,
        padding_mask,
        anchor_mask=None,
        margin: float = 0.0,
        anchor_only: bool = False,
    ):
        """
        Encourage residual to shrink across adjacent Jacobi refinement steps.

        residual_vec: [K], e.g. KL(teacher || student) per consistency pair
        traj_j_vec:   [K], trajectory index j for each pair
        traj_off_vec: [K], token offset inside block for each pair
        padding_mask: [K], True = ignore
        anchor_mask:  [K], optional anchor target mask

        Penalty:
          max(0, D_{j+1, offset} - D_{j, offset} + margin)
        """
        valid = ~padding_mask

        if anchor_only:
            if anchor_mask is None:
                return torch.zeros((), device=residual_vec.device)
            valid = valid & anchor_mask.bool()

        if valid.sum() == 0:
            return torch.zeros((), device=residual_vec.device)

        terms = []

        max_j = int(traj_j_vec[valid].max().detach().cpu())
        for j in range(max_j):
            cur_mask = valid & (traj_j_vec == j)
            nxt_mask = valid & (traj_j_vec == (j + 1))

            if cur_mask.sum() == 0 or nxt_mask.sum() == 0:
                continue

            cur_offsets = torch.unique(traj_off_vec[cur_mask])
            nxt_offsets = torch.unique(traj_off_vec[nxt_mask])

            # small tensors, Python set intersection is fine
            common_offsets = set(int(x.detach().cpu()) for x in cur_offsets) & \
                             set(int(x.detach().cpu()) for x in nxt_offsets)

            for off in common_offsets:
                cur_off_mask = cur_mask & (traj_off_vec == off)
                nxt_off_mask = nxt_mask & (traj_off_vec == off)

                if cur_off_mask.sum() == 0 or nxt_off_mask.sum() == 0:
                    continue

                d_cur = residual_vec[cur_off_mask].mean()
                d_nxt = residual_vec[nxt_off_mask].mean()

                terms.append(F.relu(d_nxt - d_cur + margin))

        if len(terms) == 0:
            return torch.zeros((), device=residual_vec.device)

        return torch.stack(terms).mean()
     

    def _convergence_difficulty_score(
        self,
        input_ids: torch.Tensor,
        k_starts,
        l_starts,
        N: int,
        T: int,
        traj_j_vec: torch.Tensor,
        traj_off_vec: torch.Tensor,
        padding_mask: torch.Tensor,
        mode: str = "persistence",
    ) -> torch.Tensor:
        """
        Return [K] difficulty scores in [0, 1] for consistency pairs.

        Intuition:
          score(j, off) is high when the same offset remains different between
          k_u and last_u across the Jacobi trajectory.

        Modes:
          - local:       1[k_j[off] != last_j[off]]
          - global:      average mismatch over all u for this offset
          - persistence: average mismatch over future u >= j for this offset
                         This emphasizes positions that are still not converged
                         at and after the current trajectory step.
        """
        device = input_ids.device
        pad_id = getattr(self.processing_class, "pad_token_id", None)

        mismatch_rows = []
        valid_rows = []
        for jj in range(T):
            kb = input_ids[k_starts[jj] : k_starts[jj] + N]
            lb = input_ids[l_starts[jj] : l_starts[jj] + N]

            valid = torch.ones((N,), dtype=torch.bool, device=device)
            if pad_id is not None:
                valid = valid & kb.ne(pad_id) & lb.ne(pad_id)

            mismatch_rows.append((kb != lb) & valid)
            valid_rows.append(valid)

        mismatch = torch.stack(mismatch_rows, dim=0).float()  # [T, N]
        valid = torch.stack(valid_rows, dim=0).float()        # [T, N]

        mode = str(mode).lower()
        if mode == "local":
            score_grid = mismatch
        elif mode == "global":
            denom = valid.sum(dim=0, keepdim=True).clamp_min(1.0)
            score_grid = mismatch.sum(dim=0, keepdim=True) / denom
            score_grid = score_grid.expand_as(mismatch)
        elif mode == "persistence":
            # Future persistence from current j to the end.
            rev_mismatch = torch.flip(mismatch, dims=[0])
            rev_valid = torch.flip(valid, dims=[0])
            future_mismatch = torch.flip(torch.cumsum(rev_mismatch, dim=0), dims=[0])
            future_valid = torch.flip(torch.cumsum(rev_valid, dim=0), dims=[0]).clamp_min(1.0)
            score_grid = future_mismatch / future_valid
        else:
            raise ValueError(
                f"Unknown conv_cons_mode={mode}. Expected one of: persistence, global, local"
            )

        score = score_grid[traj_j_vec.long(), traj_off_vec.long()].float()
        score = score.masked_fill(padding_mask, 0.0)
        return score.clamp(0.0, 1.0)

    def _build_block_mask(self, L: int, prompt_len: int, T: int, heads: int):
        key = (L, prompt_len, T, heads, self.max_new_tokens)
        if key in self._blockmask_cache:
            return self._blockmask_cache[key]

        N = self.max_new_tokens
        k_starts, l_starts = self._index_layout(prompt_len, T, N)

        ks = torch.tensor(k_starts, device=self.args.device)
        ls = torch.tensor(l_starts, device=self.args.device)

        def mask_mod(b, h, q, k):
            rel_q = q - prompt_len
            rel_k = k - prompt_len
            block_idx_q = torch.div(rel_q, N, rounding_mode="floor")
            block_idx_k = torch.div(rel_k, N, rounding_mode="floor")

            is_prompt_q = q < prompt_len
            is_prompt_k = k < prompt_len

            is_kj_q    = (q >= prompt_len) & (block_idx_q % 2 == 0)
            is_lastj_q = (q >= prompt_len) & (block_idx_q % 2 == 1)
            is_kj_k    = (k >= prompt_len) & (block_idx_k % 2 == 0)
            is_lastj_k = (k >= prompt_len) & (block_idx_k % 2 == 1)

            # j index for q, clamped to [0, T-1]
            j_q = torch.clamp(block_idx_q // 2, min=0, max=T - 1)

            ks_per_q = ks[j_q]
            ls_per_q = ls[j_q]

            # for k_j queries, allow attending to ALL PREVIOUS k_* blocks
            # (k blocks have even block indices: 0, 2, 4, ... 2*(j_q-1))
            k_in_prev_k   = is_kj_k    & (block_idx_k < 2 * j_q)
            # keep old behavior for last_j (still sees previous last_*)
            last_in_prev_last = is_lastj_k & (block_idx_k < 2 * j_q)

            # prompt is always causal
            mask_prompt = is_prompt_q & (k <= q)

            # k_j queries:
            same_kj_block = is_kj_q & is_kj_k & (block_idx_q == block_idx_k)
            mask_kj = is_kj_q & (
                is_prompt_k |
                k_in_prev_k |  # attends to "previous k_*"
                (same_kj_block & (k >= ks_per_q) & (k <= q))
            )

            # last_j queries (UNCHANGED):
            same_lastj_block = is_lastj_q & is_lastj_k & (block_idx_q == block_idx_k)
            mask_lastj = is_lastj_q & (
                is_prompt_k |
                last_in_prev_last |  # attends to all previous last_{<j}
                (same_lastj_block & (k >= ls_per_q) & (k <= q))
            )

            return mask_prompt | mask_kj | mask_lastj

        block_mask = create_block_mask(
            mask_mod, B=1, H=heads, Q_LEN=L, KV_LEN=L, device=self.args.device, _compile=True,
        )
        self._blockmask_cache[key] = block_mask
        return block_mask

    def _build_dense_attention_mask(self, L: int, prompt_len: int, T: int, dtype=torch.float32):
        """
        Dense additive 4D attention mask equivalent to _build_block_mask().
        Shape: [1, 1, L, L].
        Allowed positions get 0.0; disallowed positions get a large negative value.

        This is for debugging/stabilizing math training when FlexAttention BlockMask
        produces non-finite logits. It preserves the intended visibility pattern:
          - prompt queries: causal prompt
          - k_j queries: prompt + previous k_* + causal current k_j
          - last_j queries: prompt + previous last_* + causal current last_j
        """
        device = self.args.device
        N = self.max_new_tokens

        q = torch.arange(L, device=device)[:, None]   # [L, 1]
        k = torch.arange(L, device=device)[None, :]   # [1, L]

        rel_q = q - prompt_len
        rel_k = k - prompt_len

        block_idx_q = torch.div(rel_q, N, rounding_mode="floor")
        block_idx_k = torch.div(rel_k, N, rounding_mode="floor")

        is_prompt_q = q < prompt_len
        is_prompt_k = k < prompt_len

        is_kj_q    = (q >= prompt_len) & (block_idx_q % 2 == 0)
        is_lastj_q = (q >= prompt_len) & (block_idx_q % 2 == 1)
        is_kj_k    = (k >= prompt_len) & (block_idx_k % 2 == 0)
        is_lastj_k = (k >= prompt_len) & (block_idx_k % 2 == 1)

        j_q = torch.clamp(block_idx_q // 2, min=0, max=T - 1)

        k_starts, l_starts = self._index_layout(prompt_len, T, N)
        ks = torch.tensor(k_starts, device=device)
        ls = torch.tensor(l_starts, device=device)

        ks_per_q = ks[j_q.squeeze(-1)].unsqueeze(-1)
        ls_per_q = ls[j_q.squeeze(-1)].unsqueeze(-1)

        k_in_prev_k = is_kj_k & (block_idx_k < 2 * j_q)
        last_in_prev_last = is_lastj_k & (block_idx_k < 2 * j_q)

        mask_prompt = is_prompt_q & (k <= q)

        same_kj_block = is_kj_q & is_kj_k & (block_idx_q == block_idx_k)
        mask_kj = is_kj_q & (
            is_prompt_k |
            k_in_prev_k |
            (same_kj_block & (k >= ks_per_q) & (k <= q))
        )

        same_lastj_block = is_lastj_q & is_lastj_k & (block_idx_q == block_idx_k)
        mask_lastj = is_lastj_q & (
            is_prompt_k |
            last_in_prev_last |
            (same_lastj_block & (k >= ls_per_q) & (k <= q))
        )

        allowed = mask_prompt | mask_kj | mask_lastj

        # Extra safety: guarantee no all-masked query row.
        # For the intended layout this should already be true, but this protects
        # against edge cases in malformed trajectories.
        allowed = allowed | torch.eye(L, dtype=torch.bool, device=device)

        neg = -1e9
        attn = torch.zeros((1, 1, L, L), device=device, dtype=dtype)
        attn = attn.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), neg)

        if self._jf_debug_enabled() and os.environ.get("JF_DEBUG_DENSE_MASK", "0") == "1":
            row_ok = allowed.any(dim=-1)
            if not row_ok.all():
                self._jf_save_debug(
                    where="dense_mask_all_masked_row",
                    tensors={"allowed": allowed, "input_dummy": torch.empty(0, device=device)},
                    scalars={"num_bad_rows": (~row_ok).sum()},
                    extra={"L": L, "prompt_len": prompt_len, "T": T, "N": N},
                    raise_error=True,
                )

        return attn



    def _semantic_key_mask_from_targets(self, target_ids):
        """
        Semantic/key-token mask for EPJF mid-noise states.

        This is intentionally broader than structural anchors but still far
        narrower than all tokens. It captures positions that usually determine
        program/math semantics: identifiers, literals, operators, common API
        names, and math/final-answer tokens.
        """
        valid = target_ids.ne(-100)
        if valid.sum() == 0:
            return torch.zeros_like(target_ids, dtype=torch.bool)

        mask = torch.zeros_like(target_ids, dtype=torch.bool)
        valid_positions = torch.nonzero(valid, as_tuple=False).flatten()

        operator_pieces = {
            "+", "-", "*", "/", "//", "%", "**",
            "=", "==", "!=", "<", "<=", ">", ">=",
            "and", "or", "not", "in", "is",
        }
        literal_words = {"true", "false", "none", "null"}
        common_semantic_calls = {
            "len", "range", "sum", "max", "min", "abs", "sorted", "enumerate", "zip",
            "map", "filter", "list", "tuple", "set", "dict", "str", "int", "float", "bool",
            "append", "extend", "join", "split", "replace", "find", "count", "items", "keys", "values",
            "sqrt", "gcd", "factorial", "ceil", "floor", "nlargest", "nsmallest",
        }

        for pos in valid_positions.tolist():
            tid = int(target_ids[pos].detach().cpu())
            if tid < 0:
                continue
            try:
                piece = self._decode_token_for_anchor_type(tid)
            except Exception:
                piece = ""
            p = self._norm_anchor_piece(piece)
            if not p or (p.startswith("<|") and p.endswith("|>")):
                continue

            low = p.lower()

            keep = False
            try:
                if self._is_math_anchor_token(tid):
                    keep = True
            except Exception:
                pass

            if any(ch.isdigit() for ch in p):
                keep = True
            if low in operator_pieces or low in literal_words or low in common_semantic_calls:
                keep = True
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,24}", p):
                keep = True
            if low.startswith(".") and len(low) > 1:
                keep = True
            if any(q in p for q in ['"', "'"]):
                keep = True

            if keep:
                mask[pos] = True

        return mask & valid

    def _code_returnaware_mask_from_targets(self, target_ids):
        """
        Code-specific expression mask for EPJF-Code-ReturnAware.

        This is a tokenizer-level heuristic:
          - keep tokens after return/yield/assert until newline/semicolon
          - keep RHS-like spans after '=' / '+=' / '-=' / etc. until newline/semicolon
          - keep import/from spans
        It is intentionally conservative and only used for codegen rescue modes.
        """
        valid = target_ids.ne(-100)
        if valid.sum() == 0:
            return torch.zeros_like(target_ids, dtype=torch.bool)

        mask = torch.zeros_like(target_ids, dtype=torch.bool)
        valid_positions = torch.nonzero(valid, as_tuple=False).flatten().tolist()

        in_expr = False
        budget = 0

        starters = {"return", "yield", "assert", "raise", "import", "from"}
        assign_ops = {"=", "+=", "-=", "*=", "/=", "//=", "%=", "**="}
        hard_stops = {";", "\n", "\r"}

        for pos in valid_positions:
            tid = int(target_ids[pos].detach().cpu())
            if tid < 0:
                continue

            try:
                piece = self._decode_token_for_anchor_type(tid)
            except Exception:
                piece = ""

            p = self._norm_anchor_piece(piece)
            low = p.lower()

            # Newlines or semicolons terminate the current expression span.
            if ("\n" in piece) or (p in hard_stops) or (";" in p):
                in_expr = False
                budget = 0
                continue

            # Start expression-aware spans.
            if low in starters or low in assign_ops:
                mask[pos] = True
                in_expr = True
                budget = 48
                continue

            # Also start after common Python punctuation that often begins an expression body.
            if low in {"lambda"}:
                mask[pos] = True
                in_expr = True
                budget = 48
                continue

            if in_expr and budget > 0:
                mask[pos] = True
                budget -= 1

                # Stop at block delimiters if the span has run for a few tokens.
                if low in {":", ","} and budget < 40:
                    in_expr = False
                    budget = 0

        return mask & valid

    def _progressive_error_target_weights(
        self,
        input_ids: torch.Tensor,
        sp: torch.Tensor,
        anchor_targets: torch.Tensor,
        anchor_mask: torch.Tensor,
        noise_ratio_vec: torch.Tensor,
        padding_mask: torch.Tensor,
    ):
        """
        Error-aware Progressive Jacobi Forcing (EPJF).

        Unlike prior ACW/dist_cons variants that keep the full target and only
        change weights, EPJF changes the supervised target subset by noise stage:
          high noise -> structural anchor positions only
          mid noise  -> structural anchors + semantic/key positions
          low noise  -> full target positions

        Returned tensor is a hard 0/1 token-weight mask, combined multiplicatively
        with existing consistency weights.
        """
        mode = str(getattr(self.args, "prog_target_mode", "none")).lower()
        if mode in ("none", "", "off", "false", "0"):
            return None
        allowed_modes = (
            "epjf",
            "progressive",
            "error_progressive",
            "progressive_error",
            "sspc",
            "rpc",
            "residual_progressive",
            "residual_epjf",
            "epjf_code_soft",
            "code_soft",
            "epjf_code_fasterfull",
            "code_fasterfull",
            "epjf_code_soft_random",
            "code_soft_random",
            "epjf_code_returnaware",
            "code_returnaware",
        )
        if mode not in allowed_modes:
            raise ValueError(f"Unknown prog_target_mode: {mode}")

        valid = (~padding_mask.bool()) & anchor_targets.ne(-100)
        if valid.sum() == 0:
            return torch.zeros_like(padding_mask, dtype=torch.float, device=padding_mask.device)

        high_thr = float(getattr(self.args, "prog_target_high_threshold", 0.66))
        mid_thr = float(getattr(self.args, "prog_target_mid_threshold", 0.33))

        # Codegen rescue variant:
        # use a shorter high-noise phase and start full-target supervision earlier.
        if mode in ("epjf_code_fasterfull", "code_fasterfull"):
            high_thr = 0.60
            mid_thr = 0.40

        if mid_thr > high_thr:
            mid_thr, high_thr = high_thr, mid_thr

        nr = noise_ratio_vec.float().to(padding_mask.device)
        high = nr >= high_thr
        mid = (nr >= mid_thr) & (nr < high_thr)
        low = nr < mid_thr

        structural = anchor_mask.bool().to(padding_mask.device) & valid
        semantic = self._semantic_key_mask_from_targets(anchor_targets).to(padding_mask.device) & valid

        try:
            noisy_tokens = input_ids.index_select(0, sp).detach().to(anchor_targets.device)
            mismatch = noisy_tokens.ne(anchor_targets) & valid
        except Exception:
            mismatch = valid

        # Default EPJF: effective for math, but too sparse for codegen.
        high_mask = structural
        mid_mask = (structural | semantic) & valid
        low_mask = valid

        # Code-Soft:
        # expose semantic/API/literal/operator/identifier tokens already in high-noise states.
        if mode in (
            "epjf_code_soft",
            "code_soft",
            "epjf_code_fasterfull",
            "code_fasterfull",
            "epjf_code_soft_random",
            "code_soft_random",
            "epjf_code_returnaware",
            "code_returnaware",
        ):
            high_mask = (structural | semantic) & valid
            mid_mask = (structural | semantic | mismatch) & valid
            low_mask = valid

        # Code-Soft-Random:
        # deterministic position-based random keep to reduce over-sparse supervision on MBPP-style short expressions.
        if mode in ("epjf_code_soft_random", "code_soft_random"):
            high_keep = float(getattr(self.args, "prog_target_random_high_keep", 0.30))
            mid_keep = float(getattr(self.args, "prog_target_random_mid_keep", 0.60))
            high_keep = max(0.0, min(1.0, high_keep))
            mid_keep = max(0.0, min(1.0, mid_keep))

            pos = torch.arange(anchor_targets.numel(), device=padding_mask.device, dtype=torch.long)
            # Deterministic pseudo-random score in [0, 1), independent of RNG state/DDP rank.
            rand_score = ((pos * 1103515245 + 12345) % 1000).float() / 1000.0
            high_rand = (rand_score < high_keep) & valid
            mid_rand = (rand_score < mid_keep) & valid

            high_mask = (high_mask | high_rand) & valid
            mid_mask = (mid_mask | mid_rand) & valid

        # Code-ReturnAware:
        # add tokens in return/yield/assert expressions and RHS assignment-like spans.
        if mode in ("epjf_code_returnaware", "code_returnaware"):
            try:
                returnaware = self._code_returnaware_mask_from_targets(anchor_targets).to(padding_mask.device) & valid
            except Exception:
                returnaware = torch.zeros_like(valid, dtype=torch.bool, device=padding_mask.device)
            high_mask = (high_mask | returnaware) & valid
            mid_mask = (mid_mask | returnaware | mismatch) & valid

        if bool(getattr(self.args, "prog_target_low_mismatch_only", False)):
            low_mask = (mismatch | structural | semantic) & valid

        target_mask = torch.where(high, high_mask, torch.where(mid, mid_mask, low_mask))

        if bool(getattr(self.args, "prog_target_mismatch_only", False)):
            target_mask = target_mask & (mismatch | structural | semantic)

        min_keep = int(getattr(self.args, "prog_target_min_keep", 1))
        if min_keep > 0 and int((target_mask & valid).sum().detach().cpu()) < min_keep:
            fallback = (structural | semantic) & valid
            if int(fallback.sum().detach().cpu()) >= min_keep:
                target_mask = fallback
            else:
                target_mask = valid

        # SSPC/RPC: residual progressive consistency.
        # EPJF keeps the original hard 0/1 progressive target mask.
        # SSPC/RPC preserves weak full-token consistency on non-selected valid tokens,
        # preventing exactness collapse on structure-fragile outputs such as code.
        if mode in ("sspc", "rpc", "residual_progressive", "residual_epjf"):
            selected_weight = float(getattr(self.args, "prog_target_selected_weight", 1.0))
            residual_high = float(getattr(self.args, "prog_target_residual_high", 0.0))
            residual_mid = float(getattr(self.args, "prog_target_residual_mid", 0.0))
            residual_low = float(getattr(self.args, "prog_target_residual_low", 1.0))

            stage_residual = torch.where(
                high,
                torch.full_like(noise_ratio_vec.float(), residual_high, device=padding_mask.device),
                torch.where(
                    mid,
                    torch.full_like(noise_ratio_vec.float(), residual_mid, device=padding_mask.device),
                    torch.full_like(noise_ratio_vec.float(), residual_low, device=padding_mask.device),
                ),
            )

            # Low stage is already full-target in the default EPJF definition, but keep this
            # explicit so low_mismatch_only experiments can still use residual preservation.
            selected = target_mask.bool() & valid
            weights = torch.where(
                selected,
                torch.full_like(stage_residual, selected_weight, device=padding_mask.device),
                stage_residual,
            )
            return weights.float().masked_fill(~valid, 0.0)

        return target_mask.float().masked_fill(~valid, 0.0)

    def _distance_consistency_weight(
        self,
        noise_ratio_vec: torch.Tensor,
        traj_j_vec: torch.Tensor,
        T: int,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build per-token distance/stage weights for consistency loss.

        Interpretation:
          - noise_ratio is a proxy for distance from the final target.
          - high noise / early trajectory states are treated as far states.
          - low noise / late trajectory states are treated as near-final states.

        Returns:
          [K] float tensor. Padding positions get 0.
        """
        mode = str(getattr(self.args, "dist_cons_mode", "none")).lower()
        device = padding_mask.device

        w = torch.ones_like(padding_mask, dtype=torch.float, device=device)
        if mode in ("none", "", "off"):
            return w.masked_fill(padding_mask, 0.0)

        near_w = float(getattr(self.args, "dist_cons_near_weight", 0.5))
        far_w = float(getattr(self.args, "dist_cons_far_weight", 0.25))
        power = float(getattr(self.args, "dist_cons_power", 1.0))

        # Clamp for safety.
        near_w = max(0.0, min(1.0, near_w))
        far_w = max(0.0, min(1.0, far_w))
        power = max(1e-6, power)

        if mode in ("noise_linear", "noise"):
            # low noise -> close to final -> high weight
            closeness = (1.0 - noise_ratio_vec.float()).clamp(0.0, 1.0)
            w = far_w + (1.0 - far_w) * torch.pow(closeness, power)

        elif mode in ("noise_bucket", "bucket", "near_far"):
            # high-noise far states get far_w, mid states get near_w,
            # low-noise near-final states get 1.0.
            nr = noise_ratio_vec.float()
            w = torch.where(
                nr <= 0.33,
                torch.ones_like(nr),
                torch.where(
                    nr <= 0.66,
                    torch.full_like(nr, near_w),
                    torch.full_like(nr, far_w),
                ),
            )

        elif mode in ("traj_linear", "traj"):
            # j = 0 is early/far; j = T-1 is late/near-final.
            if T <= 1:
                progress = torch.ones_like(traj_j_vec, dtype=torch.float, device=device)
            else:
                progress = traj_j_vec.float() / float(max(T - 1, 1))
                progress = progress.clamp(0.0, 1.0)
            w = far_w + (1.0 - far_w) * torch.pow(progress, power)

        elif mode in ("traj_bucket",):
            if T <= 1:
                progress = torch.ones_like(traj_j_vec, dtype=torch.float, device=device)
            else:
                progress = (traj_j_vec.float() / float(max(T - 1, 1))).clamp(0.0, 1.0)

            w = torch.where(
                progress >= 0.67,
                torch.ones_like(progress),
                torch.where(
                    progress >= 0.33,
                    torch.full_like(progress, near_w),
                    torch.full_like(progress, far_w),
                ),
            )

        else:
            raise ValueError(
                f"Unknown dist_cons_mode={mode}. "
                "Expected one of: none, noise_linear, noise_bucket, traj_linear, traj_bucket"
            )

        w = w.to(device=device, dtype=torch.float)
        w = w.masked_fill(padding_mask, 0.0)
        return w


    def _distance_consistency_weight(
        self,
        noise_ratio_vec: torch.Tensor,
        traj_j_vec: torch.Tensor,
        T: int,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-token distance/stage weight for consistency loss.

        This is the safe version:
          - high-noise / far states get smaller weights
          - low-noise / near-final states get weight 1.0
        """
        mode = str(getattr(self.args, "dist_cons_mode", "none")).lower()
        device = padding_mask.device

        w = torch.ones_like(padding_mask, dtype=torch.float, device=device)
        if mode in ("none", "", "off", "target_bucket"):
            return w.masked_fill(padding_mask, 0.0)

        near_w = float(getattr(self.args, "dist_cons_near_weight", 0.5))
        far_w = float(getattr(self.args, "dist_cons_far_weight", 0.25))
        power = float(getattr(self.args, "dist_cons_power", 1.0))

        near_w = max(0.0, min(1.0, near_w))
        far_w = max(0.0, min(1.0, far_w))
        power = max(1e-6, power)

        if mode in ("noise_bucket", "bucket", "near_far"):
            nr = noise_ratio_vec.float()
            w = torch.where(
                nr <= 0.33,
                torch.ones_like(nr),
                torch.where(
                    nr <= 0.66,
                    torch.full_like(nr, near_w),
                    torch.full_like(nr, far_w),
                ),
            )

        elif mode in ("noise_linear", "noise"):
            closeness = (1.0 - noise_ratio_vec.float()).clamp(0.0, 1.0)
            w = far_w + (1.0 - far_w) * torch.pow(closeness, power)

        elif mode in ("traj_bucket",):
            if T <= 1:
                progress = torch.ones_like(traj_j_vec, dtype=torch.float, device=device)
            else:
                progress = (traj_j_vec.float() / float(max(T - 1, 1))).clamp(0.0, 1.0)

            w = torch.where(
                progress >= 0.67,
                torch.ones_like(progress),
                torch.where(
                    progress >= 0.33,
                    torch.full_like(progress, near_w),
                    torch.full_like(progress, far_w),
                ),
            )

        elif mode in ("traj_linear", "traj"):
            if T <= 1:
                progress = torch.ones_like(traj_j_vec, dtype=torch.float, device=device)
            else:
                progress = (traj_j_vec.float() / float(max(T - 1, 1))).clamp(0.0, 1.0)
            w = far_w + (1.0 - far_w) * torch.pow(progress, power)

        else:
            raise ValueError(
                f"Unknown dist_cons_mode={mode}. "
                "Expected: none, noise_bucket, noise_linear, traj_bucket, traj_linear, target_bucket"
            )

        return w.to(device=device, dtype=torch.float).masked_fill(padding_mask, 0.0)

    def _distance_target_granularity_weight(
        self,
        noise_ratio_vec: torch.Tensor,
        anchor_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Distance-aware target granularity.

        Near / low-noise:
          supervise all consistency positions.

        Mid:
          keep anchors strong, downweight non-anchors.

        Far / high-noise:
          keep only anchors, suppress non-anchors.

        This approximates:
          far state  -> anchor/sketch target
          mid state  -> anchor-focused target
          near state -> exact/all-token target
        without changing the trajectory data layout.
        """
        device = padding_mask.device
        nr = noise_ratio_vec.float().to(device)

        far_anchor_w = float(getattr(self.args, "dist_cons_far_anchor_weight", 0.5))
        mid_nonanchor_w = float(getattr(self.args, "dist_cons_mid_nonanchor_weight", 0.25))

        far_anchor_w = max(0.0, min(1.0, far_anchor_w))
        mid_nonanchor_w = max(0.0, min(1.0, mid_nonanchor_w))

        anchor = anchor_mask.bool().to(device)
        valid = ~padding_mask.bool()

        near = nr <= 0.33
        mid = (nr > 0.33) & (nr <= 0.66)
        far = nr > 0.66

        w = torch.zeros_like(nr, dtype=torch.float, device=device)

        # Near: all tokens.
        w = torch.where(near, torch.ones_like(w), w)

        # Mid: anchors = 1, non-anchors = mid_nonanchor_w.
        mid_w = torch.where(
            anchor,
            torch.ones_like(w),
            torch.full_like(w, mid_nonanchor_w),
        )
        w = torch.where(mid, mid_w, w)

        # Far: anchors only. Non-anchors are effectively masked out.
        far_w = torch.where(
            anchor,
            torch.full_like(w, far_anchor_w),
            torch.zeros_like(w),
        )
        w = torch.where(far, far_w, w)

        w = w.masked_fill(~valid, 0.0)
        return w

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.train_step_cnt += 1
        return self._one_pass_losses_step(model, inputs)

    def _one_pass_losses_step(self, model, inputs):
        input_ids, prompt_len, T, example_index = self._unpack_sample(inputs)
        if self._jf_debug_enabled():
            self._jf_check_some_params(model, input_ids=input_ids)

            self._jf_check_int_range(
                "input_ids",
                input_ids,
                low=0,
                high=getattr(model.config, "vocab_size", input_ids.max().item() + 1),
                tensors={"input_ids": input_ids},
                extra={"stage": "begin_one_pass"},
            )
        L = input_ids.size(0)

        eos_id = getattr(self.processing_class, "eos_token_id")
        pad_id = getattr(self.processing_class, "pad_token_id")
        N = self.max_new_tokens

        expected_len = prompt_len + 2 * T * N
        if L != expected_len:
            raise ValueError(
                f"Length mismatch: L={L}, expected {expected_len} (prompt_len={prompt_len}, T={T}, n_token_sequence_size={N})"
            )

        # ===== Debug printing =====
        # detok the AR input sequence
        #prompt_ids_block = input_ids[:prompt_len]
        #l_blocks_concat = torch.cat([input_ids[ls : ls + N] for ls in l_starts], dim=0)
        #ar_concat_ids = torch.cat([prompt_ids_block, l_blocks_concat], dim=0)

        #print("\n=== AR INPUTS (prompt + concatenated l_j blocks) ===")
        
        # Decode
        #ar_text = self.processing_class.decode(ar_concat_ids, skip_special_tokens=False)

        #print("\n[Decoded text]")
        #print(ar_text)

        # Print all k_j blocks separately
        #print("\n=== k_j BLOCKS ===")
        #for j, ks in enumerate(k_starts):
        #    block_ids = input_ids[ks : ks + N]
        #    block_text = self.processing_class.decode(block_ids, skip_special_tokens=False)
        #    print(f"[k_{j}]")
        #    print(block_text)
        #    print()
        # ==========================

        attn_mask = torch.ones(L, dtype=torch.long, device=input_ids.device)
        k_starts, l_starts = self._index_layout(prompt_len, T, N)

        ### METHOD 1: PAD post-EOS TOKENS on last_N
        self._flip_block_after_eos_to_pad(input_ids, l_starts[-1], N, eos_id, pad_id)
        
        ###-- METHOD 2: cut post-EOS inside k_N & last_N block
        #--for j in range(T):
        #--    starting_pos_k = k_starts[j]
        #--    starting_pos_l = l_starts[j]
        
        #--    block_k = input_ids[starting_pos_k : starting_pos_k + N]
        #--    block_l = input_ids[starting_pos_l : starting_pos_l + N]

        #--    pos = (block_l == eos_id).nonzero(as_tuple=False)
        #--    if pos.numel():
        #--        first_eos_pos_k = starting_pos_k + int(pos[0])
        #--        first_eos_pos_l = starting_pos_l + int(pos[0])
        #--        attn_mask[first_eos_pos_k + 1 : starting_pos_k + N] = 0
        #--        attn_mask[first_eos_pos_l + 1 : starting_pos_l + N] = 0

        # Shared position ids for each (k_j, last_j) pair.
        position_ids = self._build_shared_position_ids(L, prompt_len, T)

        # Build attention mask according to backend.
        # flex  : original FlexAttention BlockMask path.
        # dense : dense additive 4D mask path, intended for eager/sdpa fallback.
        num_heads = getattr(self.cfg, 'num_attention_heads', 28)
        blk_mask = None
        dense_mask = None
        if self.jf_attn_backend == "flex":
            blk_mask = self._build_block_mask(L, prompt_len, T, num_heads)
        elif self.jf_attn_backend in ("dense", "eager", "sdpa"):
            dense_mask = self._build_dense_attention_mask(
                L=L,
                prompt_len=prompt_len,
                T=T,
                dtype=torch.float32,
            )
        else:
            raise ValueError(
                f"Unknown JF_ATTN_BACKEND={self.jf_attn_backend}. "
                "Expected one of: flex, dense, eager, sdpa"
            )

        need_hidden_states = self.use_keep_gate
        # ========== Debug: check model parameters before forward ==========
        # This distinguishes:
        #   A) current batch/loss creates NaN
        #   B) previous optimizer update already corrupted model weights
        #
        # To keep overhead low, check only a few representative trainable params.
        if int(getattr(self.state, "global_step", 0)) % 10 == 0:
            bad_params = []
            checked_params = 0

            for name, p in model.named_parameters():
                if p.requires_grad and p.data is not None:
                    checked_params += 1
                    if not torch.isfinite(p.data).all():
                        bad_params.append(name)
                        if len(bad_params) >= 5:
                            break

                    # Do not scan the whole 7B model every step.
                    # Checking first few trainable params is enough to detect
                    # global parameter corruption in most NaN-collapse cases.
                    if checked_params >= 8:
                        break

            if bad_params:
                print(
                    f"[FATAL] non-finite model parameters before forward "
                    f"at step={self.state.global_step}: {bad_params}",
                    flush=True,
                )
                raise FloatingPointError(
                    f"Non-finite model parameters before forward "
                    f"at step={self.state.global_step}: {bad_params}"
                )
        if self.jf_attn_backend == "flex":
            outputs = model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=blk_mask,
                # block_mask=blk_mask,
                position_ids=position_ids.unsqueeze(0),
                attn_implementation="flex_attention",
                output_hidden_states=need_hidden_states,
                return_dict=True,
            )
        else:
            outputs = model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=dense_mask,
                position_ids=position_ids.unsqueeze(0),
                output_hidden_states=need_hidden_states,
                return_dict=True,
            )

        last_hidden = outputs.hidden_states[-1] if need_hidden_states else None   # [1, T, H]
        logits = outputs.logits                                                   # [1, T, V]
        if self._jf_debug_enabled():
            dbg_tensors = {
                "input_ids": input_ids,
                "logits": logits,
            }
            if example_index is not None:
                dbg_tensors["example_index"] = example_index

            ok_logits = self._jf_check_finite(
                "logits_after_forward",
                logits,
                tensors=dbg_tensors,
                extra={
                    "stage": "after_model_forward",
                    "vocab_size": getattr(model.config, "vocab_size", None),
                },
            )

            if not ok_logits and os.environ.get("JF_SKIP_NONFINITE_FORWARD", "0") == "1":
                # Diagnostic scan mode only.
                # Return a zero loss connected to model params so backward/DDP can proceed.
                zero = None
                for p in model.parameters():
                    if p.requires_grad:
                        zero = p.float().sum() * 0.0
                        break
                if zero is None:
                    zero = logits.float().nan_to_num(0.0).sum() * 0.0

                return zero
        block_len = self.max_new_tokens

        loss_keep = torch.zeros((), device=logits.device)
        loss_kd = torch.zeros((), device=logits.device)
        loss_anchor = torch.zeros((), device=logits.device)          # old hard CE, keep for compatibility
        loss_anchor_cons = torch.zeros((), device=logits.device)     # new weighted consistency component
        loss_contract = torch.zeros((), device=logits.device)
        n_anchor = torch.tensor(0, device=logits.device)
        n_anchor_all = torch.tensor(0, device=logits.device)
        mean_noise_ratio = torch.tensor(0.0, device=logits.device)
        mean_conv_score = torch.tensor(0.0, device=logits.device)
        mean_cons_weight = torch.tensor(1.0, device=logits.device)
        conv_beta_eff = torch.tensor(0.0, device=logits.device)
        keep_probs = None
        
        # keep gate
        if self.use_keep_gate and self.keep_head is not None:
            future_hidden = last_hidden[:, -block_len:, :]   # [1, block_len, H]
            gate_features = future_hidden.detach() if self.keep_gate_detach_features else future_hidden
            gate_features = gate_features.to(next(self.keep_head.parameters()).dtype)
            keep_logits = self.keep_head(gate_features).squeeze(-1)   # [1, block_len]
            keep_probs = torch.sigmoid(keep_logits)                   # [1, block_len]
            loss_keep = keep_probs.mean()

        # ========== AR loss ==========
        # gather (logit_pos --> target_pos) pairs
        # run soft CE.
        pair_logit_positions = []
        pair_target_positions = []

        def add_forward_pairs(seg_start: int, seg_end: int):
            """
            Add all in-block next-token pairs for a segment [seg_start, seg_end).
            Produces pairs (p --> p+1) for p in [seg_start .. seg_end-2].
            """
            p = torch.arange(seg_start, seg_end - 1, device=self.args.device, dtype=torch.long)
            t = p + 1
            pair_logit_positions.append(p)
            pair_target_positions.append(t)

        # Prompt segment
        end_prompt = prompt_len
        add_forward_pairs(0, end_prompt)

        # for each j, we need an effective length ('end' from j-1) to compute first-token loss in last_j
        for j in range(T):
            ls = l_starts[j]

            # first append bridging token
            if j == 0:
                # map last token from prompt to first token in last_0
                logit_pos = end_prompt - 1
                target_pos = ls
            elif j > 0:
                prev_ls = l_starts[j - 1]
                logit_pos = prev_ls + (N - 1)
                target_pos = ls

            pair_logit_positions.append(torch.tensor([logit_pos], device=self.args.device))
            pair_target_positions.append(torch.tensor([target_pos], device=self.args.device))

            # handle (k_j, last_j) block
            block = input_ids[ls : ls + N]

            # respect EOS inside the block
            eos_pos = None
            if eos_id is not None:
                epos = torch.nonzero(block == eos_id, as_tuple=False)
                eos_pos = int(epos[0]) if epos.numel() > 0 else None

            end = N
            if eos_pos is not None:
                end = min(end, eos_pos + 1)  # include EOS in segment
                
                # mark PAD as 0 for attn_mask
                mask_block = attn_mask[ls : ls + N]
                mask_block[block == pad_id] = 0
                attn_mask[ls : ls + N] = mask_block

            #print(f"ending position for block {j}: {end}")

            # in-block forward pairs
            add_forward_pairs(ls, ls + end)

            #--if eos_pos is not None:
            #--    break

        # Bridges: (last_{j-1} last token logits) -> (first token of last_j)
        # Note: This skips the intervening k_j block
        #for j in range(1, T):
            # length of the 
        #    prev_end = last_effective_ends[j - 1]

        #    prev_ls = l_starts[j - 1]
            # last token logit from last_{j-1}
        #    logit_pos = prev_ls + (prev_end - 1)
            # first token target from last_j
        #    target_pos = l_starts[j]

            # edge case: skip if target is PAD
        #    if pad_id is not None and input_ids[target_pos].item() == pad_id:
        #        continue

        #    pair_logit_positions.append(torch.tensor([logit_pos], device=self.args.device, dtype=torch.long))
        #    pair_target_positions.append(torch.tensor([target_pos], device=self.args.device, dtype=torch.long))

        # Compute CE over all pairs
        if len(pair_logit_positions) == 0:
            loss_ar = torch.zeros((), device=self.args.device)
        else:
            p_all = torch.cat(pair_logit_positions, dim=0)
            t_all = torch.cat(pair_target_positions, dim=0)

            ar_logits  = logits[0, p_all, :].clone()                          # [K, V]
            ar_targets = input_ids.index_select(0, t_all).clone().detach()    # [K]
            
            # ===== DEBUG PRINTING ===== #
            # ===== DEBUG PRINTING ===== #

            ar_targets[ar_targets == pad_id] = -100  # respect ignore_index

            # max_values, max_indices = ar_logits.max(dim=-1)
            #print(f"\ninput ids: {input_ids.tolist()}")

            #print(f"\nlabels length: {len(max_values)}")
            #print(f"\nargmax length: {len(max_indices)}")
            #print("\nmax logits:", max_values.tolist())
            #print("\nargmax indices:", max_indices.tolist())

            #print(f"\nattention mask length: {len(attn_mask.tolist())}")
            #print(f"\nattention mask: {attn_mask.tolist()}")
            #print(f"block attention mask blk_mask: {blk_mask}")

            #print(f"\nprompt length: {prompt_len}")
            #print(f"\nEOS position offset: {eos_pos}")
            #print(f"\ngeneration max logits: {max_values[prompt_len:].tolist()}")
            #print(f"generation indices: {max_indices[prompt_len:].tolist()}")
            if self._jf_debug_enabled():
                self._jf_check_finite(
                    "ar_logits_before_ce",
                    ar_logits,
                    tensors={
                        "input_ids": input_ids,
                        "ar_logits": ar_logits,
                        "ar_targets": ar_targets,
                        "p_all": p_all if "p_all" in locals() else torch.empty(0),
                        "t_all": t_all if "t_all" in locals() else torch.empty(0),
                    },
                    extra={"stage": "before_ar_cross_entropy"},
                )

                vocab_size = ar_logits.shape[-1]
                valid_targets = ar_targets[ar_targets != -100]
                if valid_targets.numel() > 0:
                    self._jf_check_int_range(
                        "ar_targets",
                        valid_targets,
                        low=0,
                        high=vocab_size,
                        tensors={
                            "input_ids": input_ids,
                            "ar_targets": ar_targets,
                            "valid_ar_targets": valid_targets,
                        },
                        extra={"stage": "before_ar_cross_entropy", "vocab_size": vocab_size},
                    )
            loss_ar = F.cross_entropy(
                ar_logits.float(),
                ar_targets,
                reduction="mean",
                label_smoothing=0.0,
                ignore_index=-100,
            ) * 10

        # ========== Consistency loss ==========
        T_soft = getattr(self.args, "distill_temperature", 1.0)

        # learn to predict first token in next block (last_{j+1})  from the last token in k_j
        drop_last_offset = False
        if drop_last_offset:
            offs = torch.arange(N - 1, device=self.args.device)
        else:
            offs = torch.arange(N, device=self.args.device)

        student_positions, teacher_positions, noise_ratio_positions = [], [], []
        traj_j_positions, traj_off_positions = [], []
        
        for j in range(T):
            ks, ls = k_starts[j], l_starts[j]
            pair_keep = self._block_keep_mask_divergence_and_eos(
                input_ids, ks, ls, N, eos_id=eos_id, drop_last_offset=drop_last_offset
            )

            if pair_keep.any():
                sp_j = ks + offs[pair_keep]
                tp_j = ls + offs[pair_keep]

                k_block = input_ids[ks:ks + N]
                l_block = input_ids[ls:ls + N]

                # simple block-level noise ratio
                valid_block = torch.ones(N, dtype=torch.bool, device=input_ids.device)
                if pad_id is not None:
                    valid_block = valid_block & (l_block != pad_id)

                if valid_block.sum() > 0:
                    r_j = ((k_block != l_block) & valid_block).float().sum() / valid_block.float().sum()
                else:
                    r_j = torch.tensor(0.0, device=input_ids.device)

                student_positions.append(sp_j)
                teacher_positions.append(tp_j)
                noise_ratio_positions.append(
                    torch.full((sp_j.numel(),), float(r_j.detach().cpu()), device=input_ids.device)
                )
                traj_j_positions.append(
                    torch.full((sp_j.numel(),), j, dtype=torch.long, device=input_ids.device)
                )
                traj_off_positions.append(
                    offs[pair_keep].long()
                )

        if len(student_positions) == 0:
            loss_consistency = torch.zeros((), device=self.args.device)
        else:
            sp = torch.cat(student_positions, dim=0)  # [K]
            tp = torch.cat(teacher_positions, dim=0)  # [K]
            traj_j_vec = torch.cat(traj_j_positions, dim=0)      # [K]
            traj_off_vec = torch.cat(traj_off_positions, dim=0)  # [K]
            
            # build global [L] padding mask: PADs and duplicate k_j prefixes
            global_pad_and_dup_mask = self._build_padding_mask_for_loss(input_ids, prompt_len, T)
            
            # per-pair padding mask for soft CE (True = masking out)
            padding_mask = global_pad_and_dup_mask.index_select(0, sp)

            # Build logits for all candidate pairs
            cons_student_logits = logits[0, sp, :]
            cons_teacher_logits = logits[0, tp, :].detach()

            cons_student_logits = cons_student_logits / T_soft
            cons_teacher_logits = cons_teacher_logits / T_soft

            # ========== Anchor-weighted consistency loss ==========
            anchor_cons_alpha = getattr(self.args, "anchor_cons_alpha", 0.0)
            anchor_cons_noise_gate = getattr(self.args, "anchor_cons_noise_gate", "none")
            anchor_cons_late_threshold = getattr(self.args, "anchor_cons_late_threshold", 0.3)
            n_anchor_all = torch.tensor(0, device=padding_mask.device)
            n_anchor = torch.tensor(0, device=padding_mask.device)

            noise_ratio_vec = torch.cat(noise_ratio_positions, dim=0)  # [K]
            mean_noise_ratio = noise_ratio_vec.mean().detach()

            # target ids only used to decide whether each target position is an anchor token
            anchor_targets = input_ids.index_select(0, tp).clone().detach()
            anchor_targets[anchor_targets == pad_id] = -100
            anchor_targets[padding_mask] = -100

            anchor_cons_type = getattr(self.args, "anchor_cons_type", "all")
            if self._jf_debug_enabled():
                self._jf_check_finite(
                    "cons_student_logits_before_soft_ce",
                    cons_student_logits,
                    tensors={
                        "input_ids": input_ids,
                        "cons_student_logits": cons_student_logits,
                        "cons_teacher_logits": cons_teacher_logits,
                        "padding_mask": padding_mask,
                        "anchor_targets": anchor_targets if "anchor_targets" in locals() else torch.empty(0),
                        "noise_ratio_vec": noise_ratio_vec if "noise_ratio_vec" in locals() else torch.empty(0),
                        "sp": sp if "sp" in locals() else torch.empty(0),
                        "tp": tp if "tp" in locals() else torch.empty(0),
                    },
                    extra={"stage": "before_consistency_soft_ce"},
                )

                self._jf_check_finite(
                    "cons_teacher_logits_before_soft_ce",
                    cons_teacher_logits,
                    tensors={
                        "input_ids": input_ids,
                        "cons_student_logits": cons_student_logits,
                        "cons_teacher_logits": cons_teacher_logits,
                        "padding_mask": padding_mask,
                        "anchor_targets": anchor_targets if "anchor_targets" in locals() else torch.empty(0),
                        "noise_ratio_vec": noise_ratio_vec if "noise_ratio_vec" in locals() else torch.empty(0),
                        "sp": sp if "sp" in locals() else torch.empty(0),
                        "tp": tp if "tp" in locals() else torch.empty(0),
                    },
                    extra={"stage": "before_consistency_soft_ce"},
                )

                if torch.is_tensor(padding_mask):
                    if padding_mask.dtype != torch.bool:
                        self._jf_save_debug(
                            where="bad_padding_mask_dtype",
                            tensors={"input_ids": input_ids, "padding_mask": padding_mask},
                            extra={"padding_mask_dtype": str(padding_mask.dtype)},
                            raise_error=True,
                        )
            if anchor_cons_type == "auto_influence":
                anchor_mask, n_anchor = self._auto_influence_anchor_mask_from_targets(
                    anchor_targets
                )
                anchor_mask_all = anchor_mask
                n_anchor_all = n_anchor
            elif anchor_cons_type == "math":
                # Math-specific anchors are constructed directly, not as a subset of
                # the original code-anchor mask.
                anchor_mask, n_anchor = self._math_anchor_mask_from_targets(anchor_targets)

                # For logging: in math mode, "all" means all math anchors.
                anchor_mask_all = anchor_mask
                n_anchor_all = n_anchor
            elif anchor_cons_type in ("math_plus_final", "math_final_plus_math"):
                # Union of broad math anchors and sequence-level final-answer anchors.
                # This preserves math-anchor coverage while adding answer-bearing final tokens.
                math_mask, n_math_anchor = self._math_anchor_mask_from_targets(anchor_targets)
                final_mask, n_final_anchor = self._math_final_anchor_mask_from_context(
                    input_ids,
                    tp,
                    padding_mask,
                    buffer_tokens=1,
                )
                anchor_mask = (math_mask | final_mask) & anchor_targets.ne(-100)
                anchor_mask_all = anchor_mask
                n_anchor_all = anchor_mask.sum()
                n_anchor = n_anchor_all
            elif anchor_cons_type == "math_final":
                anchor_mask, n_anchor = self._math_final_anchor_mask_from_context(
                    input_ids,
                    tp,
                    padding_mask,
                )
                anchor_mask_all = anchor_mask
                n_anchor_all = n_anchor
            elif anchor_cons_type in ("all_plus_completion", "all_completion"):
                # Union of the original broad code anchor and completion-critical anchors.
                # This preserves ACW-Mid/all-anchor behavior while adding return/import/call/closing-token anchors.
                anchor_mask_all, n_anchor_all = self._anchor_mask_from_targets(anchor_targets)
                completion_mask, n_completion_anchor = self._code_completion_anchor_mask_from_targets(anchor_targets)
                anchor_mask = (anchor_mask_all | completion_mask) & anchor_targets.ne(-100)
                n_anchor = anchor_mask.sum()
            elif anchor_cons_type == "code_completion":
                anchor_mask, n_anchor = self._code_completion_anchor_mask_from_targets(anchor_targets)
                anchor_mask_all = anchor_mask
                n_anchor_all = n_anchor
            else:
                anchor_mask_all, n_anchor_all = self._anchor_mask_from_targets(anchor_targets)

                anchor_mask = self._filter_anchor_mask_by_type(
                    anchor_targets,
                    anchor_mask_all,
                    anchor_cons_type,
                )
                n_anchor = anchor_mask.sum()
            
            # ========== Anchor + convergence-aware weighted consistency loss ==========
            conv_cons_beta_raw = float(getattr(self.args, "conv_cons_beta", 0.0))
            conv_cons_mode = getattr(self.args, "conv_cons_mode", "persistence")
            conv_cons_max_weight = float(getattr(self.args, "conv_cons_max_weight", 2.0))
            conv_cons_warmup_steps = int(getattr(self.args, "conv_cons_warmup_steps", 500))

            if conv_cons_beta_raw > 0:
                if conv_cons_warmup_steps > 0:
                    warm = min(1.0, float(self.train_step_cnt) / float(conv_cons_warmup_steps))
                else:
                    warm = 1.0
                conv_beta_value = conv_cons_beta_raw * warm
                conv_beta_eff = torch.tensor(conv_beta_value, device=padding_mask.device)

                conv_score = self._convergence_difficulty_score(
                    input_ids=input_ids,
                    k_starts=k_starts,
                    l_starts=l_starts,
                    N=N,
                    T=T,
                    traj_j_vec=traj_j_vec,
                    traj_off_vec=traj_off_vec,
                    padding_mask=padding_mask,
                    mode=conv_cons_mode,
                )
            else:
                conv_score = torch.zeros_like(padding_mask, dtype=torch.float, device=padding_mask.device)
                conv_beta_eff = torch.tensor(0.0, device=padding_mask.device)

            valid_cons = ~padding_mask
            if valid_cons.any():
                mean_conv_score = conv_score[valid_cons].mean().detach()
            else:
                mean_conv_score = torch.tensor(0.0, device=padding_mask.device)

            dist_cons_mode = str(getattr(self.args, "dist_cons_mode", "none")).lower()
            use_dist_cons = dist_cons_mode not in ("none", "", "off")
            dist_cons_weight = self._distance_consistency_weight(
                noise_ratio_vec=noise_ratio_vec,
                traj_j_vec=traj_j_vec,
                T=T,
                padding_mask=padding_mask,
            )

            prog_target_mode = str(getattr(self.args, "prog_target_mode", "none")).lower()
            use_prog_target = prog_target_mode not in ("none", "", "off", "false", "0")

            use_weighted_cons = (
                (anchor_cons_alpha > 0)
                or (float(conv_beta_eff.detach().cpu()) > 0.0)
                or use_dist_cons
                or use_prog_target
            )

            if not use_weighted_cons:
                # Official vanilla Jacobi Forcing path.
                loss_consistency = self.soft_cross_entropy(
                    cons_student_logits.float(),
                    cons_teacher_logits.float(),
                    padding_mask,
                )
                mean_cons_weight = torch.tensor(1.0, device=padding_mask.device)
            else:
                token_weights = torch.ones_like(padding_mask, dtype=torch.float, device=padding_mask.device)

                if use_dist_cons and dist_cons_mode != "target_bucket":
                    token_weights = token_weights * dist_cons_weight

                if anchor_cons_alpha > 0:
                    anchor_cons_type = getattr(self.args, "anchor_cons_type", "all")
                    anchor_cons_noise_gate = getattr(self.args, "anchor_cons_noise_gate", "none")
                    anchor_cons_late_threshold = getattr(self.args, "anchor_cons_late_threshold", 0.3)

                    if anchor_cons_type == "auto_influence":
                        anchor_mask, n_anchor = self._auto_influence_anchor_mask_from_targets(
                            anchor_targets
                        )
                        anchor_mask_all = anchor_mask
                        n_anchor_all = n_anchor
                    elif anchor_cons_type == "math":
                        anchor_mask, n_anchor = self._math_anchor_mask_from_targets(anchor_targets)
                        anchor_mask_all = anchor_mask
                        n_anchor_all = n_anchor
                    elif anchor_cons_type in ("math_plus_final", "math_final_plus_math"):
                        # Union of broad math anchors and sequence-level final-answer anchors.
                        # This preserves math-anchor coverage while adding answer-bearing final tokens.
                        math_mask, n_math_anchor = self._math_anchor_mask_from_targets(anchor_targets)
                        final_mask, n_final_anchor = self._math_final_anchor_mask_from_context(
                            input_ids,
                            tp,
                            padding_mask,
                            buffer_tokens=1,
                        )
                        anchor_mask = (math_mask | final_mask) & anchor_targets.ne(-100)
                        anchor_mask_all = anchor_mask
                        n_anchor_all = anchor_mask.sum()
                        n_anchor = n_anchor_all
                    elif anchor_cons_type == "math_final":
                        anchor_mask, n_anchor = self._math_final_anchor_mask_from_context(
                            input_ids,
                            tp,
                            padding_mask,
                        )
                        anchor_mask_all = anchor_mask
                        n_anchor_all = n_anchor
                    elif anchor_cons_type in ("all_plus_completion", "all_completion"):
                        # Union of the original broad code anchor and completion-critical anchors.
                        # This preserves ACW-Mid/all-anchor behavior while adding return/import/call/closing-token anchors.
                        anchor_mask_all, n_anchor_all = self._anchor_mask_from_targets(anchor_targets)
                        completion_mask, n_completion_anchor = self._code_completion_anchor_mask_from_targets(anchor_targets)
                        anchor_mask = (anchor_mask_all | completion_mask) & anchor_targets.ne(-100)
                        n_anchor = anchor_mask.sum()
                    elif anchor_cons_type == "code_completion":
                        anchor_mask, n_anchor = self._code_completion_anchor_mask_from_targets(anchor_targets)
                        anchor_mask_all = anchor_mask
                        n_anchor_all = n_anchor
                    else:
                        anchor_mask_all, n_anchor_all = self._anchor_mask_from_targets(anchor_targets)
                        anchor_mask = self._filter_anchor_mask_by_type(
                            anchor_targets,
                            anchor_mask_all,
                            anchor_cons_type,
                        )
                        n_anchor = anchor_mask.sum()

                    if anchor_cons_noise_gate == "none":
                        gate_w = torch.ones_like(noise_ratio_vec, dtype=torch.float, device=padding_mask.device)
                    elif anchor_cons_noise_gate == "mid":
                        gate_w = 4.0 * noise_ratio_vec.float() * (1.0 - noise_ratio_vec.float())
                        gate_w = gate_w.clamp_min(0.0)
                    elif anchor_cons_noise_gate == "late":
                        gate_w = (noise_ratio_vec.float() <= anchor_cons_late_threshold).float()
                    else:
                        raise ValueError(f"Unknown anchor_cons_noise_gate: {anchor_cons_noise_gate}")

                    token_weights = token_weights + anchor_cons_alpha * anchor_mask.float() * gate_w

                if use_dist_cons and dist_cons_mode == "target_bucket":
                    token_weights = self._distance_target_granularity_weight(
                        noise_ratio_vec=noise_ratio_vec,
                        anchor_mask=anchor_mask,
                        padding_mask=padding_mask,
                    )

                if float(conv_beta_eff.detach().cpu()) > 0.0:
                    token_weights = token_weights + conv_beta_eff.float() * conv_score.float()

                # Safety: avoid runaway weights when ACW and convergence weighting overlap.
                if conv_cons_max_weight > 0:
                    clamped = token_weights.clamp(max=conv_cons_max_weight)
                    token_weights = torch.where(padding_mask, token_weights, clamped)


                # Progressive target weights used to multiply token_weights before:
                #   token_weights = token_weights * prog_weights
                # That lets progressive masking interfere with ACW/anchor/conv weights.
                #
                # Non-interfering version:
                #   - primary exact-consistency path uses base token_weights unchanged
                #   - progressive target weights are added only as an auxiliary term
                prog_weights = self._progressive_error_target_weights(
                    input_ids=input_ids,
                    sp=sp,
                    anchor_targets=anchor_targets,
                    anchor_mask=anchor_mask,
                    noise_ratio_vec=noise_ratio_vec,
                    padding_mask=padding_mask,
                )

                base_token_weights = token_weights

                if valid_cons.any():
                    mean_cons_weight = base_token_weights[valid_cons].mean().detach()
                else:
                    mean_cons_weight = torch.tensor(1.0, device=padding_mask.device)

                loss_consistency = self.soft_cross_entropy_weighted(
                    cons_student_logits.float(),
                    cons_teacher_logits.float(),
                    padding_mask,
                    token_weights=base_token_weights,
                )

                lambda_prog_aux = float(getattr(self.args, "lambda_prog_aux", 0.0))
                if prog_weights is not None and lambda_prog_aux > 0.0:
                    loss_prog_aux = self.soft_cross_entropy_weighted(
                        cons_student_logits.float(),
                        cons_teacher_logits.float(),
                        padding_mask,
                        token_weights=prog_weights,
                    )
                    loss_consistency = loss_consistency + lambda_prog_aux * loss_prog_aux

            loss_consistency = loss_consistency * (T_soft * T_soft) / T
            
            # ========== Trajectory contraction / residual shrinking loss ==========
            lambda_contract = getattr(self.args, "lambda_contract", 0.0)
            contract_margin = getattr(self.args, "contract_margin", 0.0)
            contract_on_anchor_only = getattr(self.args, "contract_on_anchor_only", False)

            if lambda_contract > 0:
                # Per-token residual for the same consistency pairs.
                # Teacher logits are already detached above.
                residual_vec = self._per_token_kl_from_logits(
                    cons_student_logits.float(),
                    cons_teacher_logits.float(),
                )
                residual_vec = residual_vec * (T_soft * T_soft)

                loss_contract = self._trajectory_contraction_loss(
                    residual_vec=residual_vec,
                    traj_j_vec=traj_j_vec,
                    traj_off_vec=traj_off_vec,
                    padding_mask=padding_mask,
                    anchor_mask=anchor_mask,
                    margin=contract_margin,
                    anchor_only=contract_on_anchor_only,
                )
                
            # ========== Old hard anchor CE, default should be off ==========
            lambda_anchor = getattr(self.args, "lambda_anchor", 0.0)
            lambda_contract = getattr(self.args, "lambda_contract", 0.0)

            # total_loss = (
            #     loss_ar
            #     + loss_consistency
            #     + lambda_anchor * loss_anchor
            #     + lambda_contract * loss_contract
            # )
            
            if lambda_anchor > 0:
                loss_anchor, _n_anchor_ce = self._anchor_ce_loss(
                    logits[0, sp, :],
                    anchor_targets,
                    anchor_weight=anchor_weight,
                )
        if self.use_teacher_kd and self.teacher_model is not None:
            with torch.inference_mode():
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=blk_mask,
                    position_ids=position_ids.unsqueeze(0),
                    return_dict=True,
                )

            teacher_logits_full = teacher_outputs.logits   # [1, L, V]
            temp_kd = self.kd_temperature

            if self.kd_on_consistency_only:
                if len(student_positions) == 0:
                    loss_kd = torch.zeros((), device=self.args.device)
                else:
                    kd_student_logits = logits[:, sp, :]               # [1, K, V]
                    kd_teacher_logits = teacher_logits_full[:, tp, :]  # [1, K, V]

                    student_log_probs = torch.log_softmax(kd_student_logits / temp_kd, dim=-1)
                    teacher_probs = torch.softmax(kd_teacher_logits / temp_kd, dim=-1)

                    kd_per_token = F.kl_div(
                        student_log_probs,
                        teacher_probs,
                        reduction="none",
                    ).sum(dim=-1)   # [1, K]

                    if self.use_keep_gate and keep_probs is not None:
                        loss_kd = kd_per_token.mean() * (temp_kd * temp_kd)
                    else:
                        loss_kd = kd_per_token.mean() * (temp_kd * temp_kd)
            else:
                kd_student_logits = logits[:, -block_len:, :]
                kd_teacher_logits = teacher_logits_full[:, -block_len:, :]

                student_log_probs = torch.log_softmax(kd_student_logits / temp_kd, dim=-1)
                teacher_probs = torch.softmax(kd_teacher_logits / temp_kd, dim=-1)

                kd_per_token = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="none",
                ).sum(dim=-1)   # [1, block_len]

                if self.use_keep_gate and keep_probs is not None:
                    loss_kd = (keep_probs * kd_per_token).mean() * (temp_kd * temp_kd)
                else:
                    loss_kd = kd_per_token.mean() * (temp_kd * temp_kd)

        lambda_anchor = getattr(self.args, "lambda_anchor", 0.0)
        lambda_contract = getattr(self.args, "lambda_contract", 0.0)

        total_loss = (
            loss_ar
            + loss_consistency
            + lambda_anchor * loss_anchor
            + lambda_contract * loss_contract
        )
        
        if self.use_keep_gate and self.keep_head is not None:
            total_loss = total_loss + self.lambda_keep * loss_keep

        if self.use_teacher_kd and self.teacher_model is not None:
            total_loss = total_loss + self.lambda_kd * loss_kd

        # print every step for debug
        if self.args.local_rank == 0 and self.train_step_cnt % 10 == 0:
            print(
                f"[loss step={self.train_step_cnt}] "
                f"ar={float(loss_ar.detach().cpu()):.4f} "
                f"cons={float(loss_consistency.detach().cpu()):.4f} "
                f"hard_anchor={float(loss_anchor.detach().cpu()):.4f} "
                f"n_anchor_all={int(n_anchor_all.detach().cpu())} "
                f"n_anchor_used={int(n_anchor.detach().cpu())} "
                f"anchor_type={getattr(self.args, 'anchor_cons_type', 'all')} "
                f"noise={float(mean_noise_ratio.detach().cpu()):.4f} "
                f"conv={float(mean_conv_score.detach().cpu()):.4f} "
                f"conv_beta={float(conv_beta_eff.detach().cpu()):.4f} "
                f"w={float(mean_cons_weight.detach().cpu()):.4f} "
                f"contract={float(loss_contract.detach().cpu()):.4f} "
                f"total={float(total_loss.detach().cpu()):.4f}",
                flush=True,
            )

        if self.args.local_rank == 0:
            with open(self.loss_log_path, "a") as f:
                f.write(
                    f"{self.train_step_cnt}\t"
                    f"{float(loss_ar.detach().cpu()):.6f}\t"
                    f"{float(loss_consistency.detach().cpu()):.6f}\t"
                    f"{float(loss_anchor.detach().cpu()):.6f}\t"
                    f"{float(loss_contract.detach().cpu()):.6f}\t"
                    f"{int(n_anchor_all.detach().cpu())}\t"
                    f"{int(n_anchor.detach().cpu())}\t"
                    f"{getattr(self.args, 'anchor_cons_type', 'all')}\t"
                    f"{getattr(self.args, 'anchor_cons_noise_gate', 'none')}\t"
                    f"{float(mean_noise_ratio.detach().cpu()):.6f}\t"
                    f"{getattr(self.args, 'conv_cons_mode', 'persistence')}\t"
                    f"{float(conv_beta_eff.detach().cpu()):.6f}\t"
                    f"{float(mean_conv_score.detach().cpu()):.6f}\t"
                    f"{float(mean_cons_weight.detach().cpu()):.6f}\t"
                    f"{float(total_loss.detach().cpu()):.6f}\n"
                )

        if self.args.qlora:
            total_loss.requires_grad = True

        # if self.args.local_rank == 0 and wandb.run is not None:
        #     log_dict = {
        #         "ar loss": float(loss_ar.detach().cpu()),
        #         "consistency loss": float(loss_consistency.detach().cpu()),
        #     }

        #     if self.use_keep_gate:
        #         log_dict["keep loss"] = float(loss_keep.detach().cpu())
        #         log_dict["mean keep prob"] = float(keep_probs.mean().detach().cpu())

        #     if self.use_teacher_kd:
        #         log_dict["kd loss"] = float(loss_kd.detach().cpu())

        #     wandb.log(log_dict)

        #del outputs, logits
        # torch.cuda.empty_cache()
        # ========== Non-finite loss debug guard ==========
        if self._jf_debug_enabled() and not torch.isfinite(total_loss):
            self._jf_save_debug(
                where="nonfinite_total_loss",
                tensors={
                    "input_ids": input_ids,
                    "ar_logits": ar_logits if "ar_logits" in locals() else torch.empty(0, device=input_ids.device),
                    "ar_targets": ar_targets if "ar_targets" in locals() else torch.empty(0, device=input_ids.device),
                    "cons_student_logits": cons_student_logits if "cons_student_logits" in locals() else torch.empty(0, device=input_ids.device),
                    "cons_teacher_logits": cons_teacher_logits if "cons_teacher_logits" in locals() else torch.empty(0, device=input_ids.device),
                    "padding_mask": padding_mask if "padding_mask" in locals() else torch.empty(0, device=input_ids.device),
                    "anchor_targets": anchor_targets if "anchor_targets" in locals() else torch.empty(0, device=input_ids.device),
                    "noise_ratio_vec": noise_ratio_vec if "noise_ratio_vec" in locals() else torch.empty(0, device=input_ids.device),
                },
                scalars={
                    "loss_ar": loss_ar,
                    "loss_consistency": loss_consistency,
                    "loss_anchor": loss_anchor,
                    "loss_contract": loss_contract,
                    "total_loss": total_loss,
                },
                extra={"stage": "after_total_loss"},
                raise_error=True,
            )
        with self.accelerator.accumulate(model):
            self.accelerator.backward(total_loss)

        # if torch.distributed.is_available() and torch.distributed.is_initialized():
        #     torch.distributed.barrier()

        return total_loss.detach()

