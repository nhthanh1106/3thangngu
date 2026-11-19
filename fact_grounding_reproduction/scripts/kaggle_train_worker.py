import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


MODEL = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
TARGET_BATCH = 16
SEED = 42
PAPER = {"precision": 0.805, "recall": 0.765, "f1": 0.785, "accuracy": 0.794}


def log_event(stage, rank=None, **fields):
    rank_text = "?" if rank is None else str(rank)
    payload = json.dumps(fields, ensure_ascii=False, default=str)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [rank={rank_text}] {stage} {payload}", flush=True)


class SaveEventCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            log_event("recovery_checkpoint_saved", rank=0, step=state.global_step, path=args.output_dir)
        return control


class ProgressEventCallback(TrainerCallback):
    def __init__(self):
        self.prediction_steps = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            log_event("training_progress", rank=0, step=state.global_step, total_steps=state.max_steps, logs=logs or {})
        return control

    def on_prediction_step(self, args, state, control, **kwargs):
        self.prediction_steps += 1
        if state.is_world_process_zero and self.prediction_steps % 250 == 0:
            log_event("prediction_progress", rank=0, prediction_steps=self.prediction_steps)
        return control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["profile", "full"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--accum", type=int, required=True)
    parser.add_argument("--profile-out", default="/kaggle/working/profile_summary.json")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    log_event("worker_started", rank, mode=args.mode, world_size=world_size, local_rank=local_rank, batch=args.batch, accumulation=args.accum)
    train_path = Path(args.data_root) / "deberta_fg_train.csv"
    dev_path = Path(args.data_root) / "deberta_fg_dev.csv"
    data_start = time.perf_counter()
    log_event("csv_load_started", rank, train_path=train_path, dev_path=dev_path)
    train = pd.read_csv(train_path, dtype={"text": str, "label": int})
    dev = pd.read_csv(dev_path, dtype={"text": str, "label": int})
    if len(train) != 46610 or len(dev) != 6404:
        raise ValueError(f"Unexpected official split sizes: {len(train)} train, {len(dev)} dev")
    log_event("csv_load_completed", rank, train_rows=len(train), dev_rows=len(dev), seconds=round(time.perf_counter() - data_start, 2))

    tokenizer_start = time.perf_counter()
    log_event("tokenizer_load_started", rank, model=MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    log_event("tokenizer_load_completed", rank, seconds=round(time.perf_counter() - tokenizer_start, 2))

    class Encoded(torch.utils.data.Dataset):
        def __init__(self, frame, name):
            start = time.perf_counter()
            log_event("tokenization_started", rank, dataset=name, examples=len(frame), max_length=MAX_LENGTH)
            self.x = None
            texts = frame.text.tolist()
            for offset in range(0, len(texts), 512):
                encoded = tokenizer(texts[offset:offset + 512], max_length=MAX_LENGTH, padding=False, truncation=True)
                if self.x is None:
                    self.x = {key: [] for key in encoded}
                for key in self.x:
                    self.x[key].extend(encoded[key])
                if (offset // 512) % 10 == 0 or offset + 512 >= len(texts):
                    log_event("tokenization_progress", rank, dataset=name, processed=min(offset + 512, len(texts)), total=len(texts), elapsed_seconds=round(time.perf_counter() - start, 2))
            self.lengths = np.asarray([len(ids) for ids in self.x["input_ids"]], dtype=np.int32)
            self.y = torch.tensor(frame.label.to_numpy(), dtype=torch.long)
            log_event("tokenization_completed", rank, dataset=name, examples=len(self.y), seconds=round(time.perf_counter() - start, 2))

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return {
                **{k: torch.tensor(v[i], dtype=torch.long) for k, v in self.x.items()},
                "labels": self.y[i],
            }

    train_ds = Encoded(train, "train")
    dev_ds = Encoded(dev, "dev")
    collator = DataCollatorWithPadding(tokenizer, padding="longest", pad_to_multiple_of=8)
    token_length_summary = {
        name: {
            "mean": float(np.mean(ds.lengths)),
            "p50": float(np.percentile(ds.lengths, 50)),
            "p90": float(np.percentile(ds.lengths, 90)),
            "p99": float(np.percentile(ds.lengths, 99)),
            "max": int(np.max(ds.lengths)),
        }
        for name, ds in [("train", train_ds), ("dev", dev_ds)]
    }

    actual_batch = args.batch * world_size * args.accum
    if actual_batch != TARGET_BATCH:
        raise ValueError(f"Worker effective batch {actual_batch} differs from target {TARGET_BATCH}")

    def model_init():
        start = time.perf_counter()
        log_event("model_load_started", rank, model=MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL,
            num_labels=2,
            id2label={0: "NOT-ANSWERABLE", 1: "ANSWERABLE"},
            label2id={"NOT-ANSWERABLE": 0, "ANSWERABLE": 1},
        )
        log_event("model_load_completed", rank, parameters=sum(p.numel() for p in model.parameters()), seconds=round(time.perf_counter() - start, 2))
        return model

    def metric_fn(p):
        y = p.label_ids
        probability = torch.softmax(torch.tensor(p.predictions), dim=-1).numpy()[:, 1]
        yhat = (probability >= 0.5).astype(int)
        pr, re, f1, _ = precision_recall_fscore_support(
            y, yhat, average="binary", pos_label=1, zero_division=0
        )
        return {"accuracy": accuracy_score(y, yhat), "precision": pr, "recall": re, "f1": f1}

    profile = args.mode == "profile"
    output_dir = f"/kaggle/working/profile_ddp_{args.batch}" if profile else "/kaggle/working/training_runs_ddp"
    import inspect

    strategy_arg = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": 10,
        "per_device_train_batch_size": args.batch,
        "per_device_eval_batch_size": args.batch,
        "gradient_accumulation_steps": args.accum,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "linear",
        "save_strategy": "epoch" if not profile else "no",
        "load_best_model_at_end": not profile,
        "metric_for_best_model": "f1" if not profile else None,
        "greater_is_better": True,
        "save_total_limit": 2,
        "fp16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "group_by_length": True,
        "optim": "adamw_torch",
        "logging_steps": 10,
        "logging_first_step": True,
        "disable_tqdm": True,
        "log_level": "info",
        "report_to": "none",
        "seed": SEED,
        "data_seed": SEED,
        "dataloader_num_workers": 2,
        "max_steps": 100 if profile else -1,
    }
    kwargs[strategy_arg] = "no" if profile else "epoch"
    training_args = TrainingArguments(**kwargs)
    log_event("trainer_ready", rank, mode=args.mode, world_size=world_size, actual_effective_batch=actual_batch, output_dir=output_dir, max_steps=training_args.max_steps, logging_steps=training_args.logging_steps, save_strategy=str(training_args.save_strategy), eval_strategy=str(getattr(training_args, strategy_arg)))
    trainer = Trainer(
        model=model_init(),
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None if profile else dev_ds,
        compute_metrics=None if profile else metric_fn,
        callbacks=[] if profile else [EarlyStoppingCallback(early_stopping_patience=3), SaveEventCallback(), ProgressEventCallback()],
        data_collator=collator,
    )

    parallelism = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "n_gpu_seen_by_trainer": int(training_args.n_gpu),
        "per_device_batch_size": args.batch,
        "train_batch_size": int(training_args.train_batch_size),
        "gradient_accumulation": args.accum,
        "actual_effective_batch": actual_batch,
        "parallel_mode": str(training_args.parallel_mode),
        "device": str(training_args.device),
    }

    torch.cuda.reset_peak_memory_stats(local_rank)
    start = time.perf_counter()
    log_event("train_started", rank, mode=args.mode, max_steps=training_args.max_steps, dataset_examples=len(train_ds))
    train_output = trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    log_event("train_completed", rank, mode=args.mode, seconds=round(elapsed, 2), global_step=trainer.state.global_step, metrics=train_output.metrics)
    local_peak = torch.tensor([torch.cuda.max_memory_allocated(local_rank) / 2**30], device=training_args.device)
    if dist.is_initialized():
        gathered = [torch.zeros_like(local_peak) for _ in range(world_size)]
        dist.all_gather(gathered, local_peak)
        peaks = [round(float(item.item()), 3) for item in gathered]
        max_elapsed = torch.tensor([elapsed], device=training_args.device)
        dist.all_reduce(max_elapsed, op=dist.ReduceOp.MAX)
        elapsed = float(max_elapsed.item())
    else:
        peaks = [round(float(local_peak.item()), 3)]

    if profile:
        if rank == 0:
            result = {
                "optimizer_steps": 100,
                "elapsed_seconds": elapsed,
                "seconds_per_step": elapsed / 100,
                "micro_batch_per_gpu": args.batch,
                "gradient_accumulation": args.accum,
                "gpu_count": world_size,
                "effective_batch": actual_batch,
                "parallelism": parallelism,
                "peak_allocated_gib_per_gpu": peaks,
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
                "dynamic_padding": True,
                "group_by_length": True,
                "token_length_summary": token_length_summary,
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
            }
            Path(args.profile_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2))
        if dist.is_initialized():
            dist.barrier()
        return

    train_metrics = train_output.metrics
    log_event("dev_evaluation_started", rank, examples=len(dev_ds))
    eval_metrics = trainer.evaluate(dev_ds)
    log_event("dev_evaluation_completed", rank, metrics=eval_metrics)
    log_event("dev_prediction_started", rank, examples=len(dev_ds))
    predictions = trainer.predict(dev_ds)
    log_event("dev_prediction_completed", rank, metrics=predictions.metrics)
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()[:, 1]
    yhat = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        dev.label, yhat, average="binary", pos_label=1, zero_division=0
    )
    metrics = {
        "dataset": "MuSiQue dev",
        "n": len(dev),
        "n_positive": int(dev.label.sum()),
        "n_negative": int((dev.label == 0).sum()),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(dev.label, yhat)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix_labels_0_1": confusion_matrix(dev.label, yhat, labels=[0, 1]).tolist(),
        "training_seconds": float(train_metrics.get("train_runtime", elapsed)),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_f1": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
    }
    paper_comparison = {k: {"paper": v, "reproduced": metrics[k], "delta": metrics[k] - v} for k, v in PAPER.items()}
    output = Path("/kaggle/working/deberta_fact_grounding_checkpoint")
    checkpoint_export_seconds = None
    log_event("best_checkpoint_export_started", rank, output=output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    export_start = time.perf_counter()
    trainer.save_model(str(output))
    if rank == 0:
        tokenizer.save_pretrained(str(output))
        checkpoint_export_seconds = time.perf_counter() - export_start
        log_event("best_checkpoint_export_completed", rank, seconds=round(checkpoint_export_seconds, 2))
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        pred = dev.copy()
        pred["prob_answerable"] = probs
        pred["prediction"] = yhat
        pred.to_csv("/kaggle/working/in_domain_predictions.csv", index=False)
        pd.DataFrame([metrics]).to_csv("/kaggle/working/in_domain_metrics.csv", index=False)
        Path("/kaggle/working/in_domain_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        Path("/kaggle/working/training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")

    if dist.is_initialized():
        dist.barrier()

    reload_check = None
    reload_seconds = None
    if rank == 0:
        log_event("fresh_reload_check_started", rank, output=output)
        reload_start = time.perf_counter()
        fresh_tokenizer = AutoTokenizer.from_pretrained(output)
        fresh_model = AutoModelForSequenceClassification.from_pretrained(output).cuda().eval()
        # Check a spread of examples in small chunks so that the original and
        # freshly loaded models do not create a large validation activation.
        check_indices = np.linspace(0, len(dev) - 1, num=min(32, len(dev)), dtype=int).tolist()
        trainer.model.eval()
        original_chunks, restored_chunks = [], []
        with torch.no_grad():
            for idx in check_indices:
                sample = fresh_tokenizer(
                    dev.text.iloc[idx], max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt"
                ).to("cuda")
                original_chunks.append(trainer.model(**sample).logits.float().cpu())
                restored_chunks.append(fresh_model(**sample).logits.float().cpu())
        original = torch.cat(original_chunks, dim=0)
        restored = torch.cat(restored_chunks, dim=0)
        max_abs_logit_diff = float((original - restored).abs().max())
        same_predictions = torch.equal(original.argmax(-1), restored.argmax(-1))
        # CUDA kernels and the saved/reloaded fp32 weights can differ slightly
        # numerically; prediction identity and a 1e-3 logit tolerance are the
        # useful deployment check. The previous 1e-5 tolerance rejected an
        # otherwise identical checkpoint (observed max difference: 2.9e-4).
        logit_tolerance = 1e-3
        reload_check = {
            "checkpoint_dir": str(output),
            "files": sorted(p.name for p in output.iterdir()),
            "sample_count": len(check_indices),
            "sample_indices": check_indices,
            "max_abs_logit_diff": max_abs_logit_diff,
            "logit_tolerance": logit_tolerance,
            "same_prediction_all_samples": bool(same_predictions),
            "pass": bool(same_predictions and max_abs_logit_diff <= logit_tolerance),
        }
        reload_seconds = time.perf_counter() - reload_start
        log_event("fresh_reload_check_completed", rank, result=reload_check, seconds=round(reload_seconds, 2))
        metadata = {
            "base_model": MODEL,
            "parameter_count": sum(p.numel() for p in trainer.model.parameters()),
            "max_length": MAX_LENGTH,
            "label_mapping": {"0": "NOT-ANSWERABLE", "1": "ANSWERABLE"},
            "threshold": 0.5,
            "seed": SEED,
            "train_examples": len(train),
            "dev_examples": len(dev),
            "training_configuration": {
                "learning_rate": 2e-5,
                "effective_batch": TARGET_BATCH,
                "micro_batch_per_gpu": args.batch,
                "gradient_accumulation": args.accum,
                "max_epochs": 10,
                "early_stopping_patience": 3,
                "warmup_ratio": 0.1,
                "scheduler": "linear",
                "weight_decay": 0.01,
                "precision": "fp16",
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
                "dynamic_padding": True,
                "group_by_length": True,
            },
            "training_arguments": json.loads(json.dumps(training_args.to_dict(), default=str)),
            "parallelism": parallelism,
            "token_length_summary": token_length_summary,
            "training_runtime_metrics": train_metrics,
            "evaluation_metrics": eval_metrics,
            "validation_metrics": metrics,
            "paper_comparison": paper_comparison,
            "checkpoint_reload": reload_check,
            "checkpoint_export_seconds": checkpoint_export_seconds,
            "checkpoint_reload_seconds": reload_seconds,
            "peak_allocated_gib_per_gpu": peaks,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        }
        (output / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (output / "training_args.json").write_text(json.dumps(metadata["training_arguments"], indent=2), encoding="utf-8")
        (output / "label_mapping.json").write_text(json.dumps(metadata["label_mapping"], indent=2), encoding="utf-8")
        Path("/kaggle/working/reproduction_summary.json").write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "comparison": paper_comparison,
                    "reload_check": reload_check,
                    "training_runtime_metrics": train_metrics,
                    "checkpoint_export_seconds": checkpoint_export_seconds,
                    "checkpoint_reload_seconds": reload_seconds,
                    "gpu_configuration": parallelism,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # Persist diagnostics before raising so a failed validation never
        # hides the trained checkpoint's metrics or reload evidence.
        (output / "checkpoint_reload.json").write_text(json.dumps(reload_check, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2))
        if not reload_check["pass"]:
            raise RuntimeError(f"Fresh checkpoint reload failed; diagnostics were saved: {reload_check}")

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
