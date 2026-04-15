from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, get_linear_schedule_with_warmup

from post_training.logging import BaseTracker, NullTracker, build_tracker

from .datasets import TextToSelfiesCollator, TextToSelfiesDataset, move_tensor_batch_to_device
from .io_utils import dump_yaml, ensure_dir, write_json
from .tokenizer_utils import build_decoder_tokenizer, prepare_training_tokenizer


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def resolve_mixed_precision(mixed_precision: str, device: torch.device) -> str:
    if device.type != "cuda":
        return "no"
    if mixed_precision == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError(f"Unsupported mixed_precision value: {mixed_precision}")
    return mixed_precision


def autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "no":
        return nullcontext()
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_dataloader(
    dataset_path: str | Path,
    tokenizer: Any,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    dataset = TextToSelfiesDataset.from_jsonl(dataset_path)
    collator = TextToSelfiesCollator(
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )


def evaluate_loss(
    model: T5ForConditionalGeneration,
    dataloader: DataLoader,
    device: torch.device,
    mixed_precision: str,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            tensor_batch = move_tensor_batch_to_device(batch, device)
            with autocast_context(device, mixed_precision):
                outputs = model(**tensor_batch)
            batch_size = tensor_batch["input_ids"].size(0)
            total_examples += batch_size
            total_loss += outputs.loss.item() * batch_size

    return total_loss / total_examples if total_examples else float("nan")


def save_checkpoint(
    checkpoint_dir: Path,
    model: T5ForConditionalGeneration,
    training_tokenizer: Any,
    decoder_tokenizer: Any,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    ensure_dir(checkpoint_dir)
    model.save_pretrained(checkpoint_dir)
    training_tokenizer.save_pretrained(checkpoint_dir)
    decoder_tokenizer.save_pretrained(checkpoint_dir / "decoder_tokenizer")
    dump_yaml(checkpoint_dir / "config.yaml", config)
    write_json(checkpoint_dir / "metrics.json", metrics)


def _build_sft_tracking_config_payload(
    config: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    return {
        "stage_name": "sft_chebi20",
        "output_dir": str(output_dir),
        "seed": config.get("seed"),
        "resolved_config": config,
    }


def _build_sft_tracking_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "best_validation_loss": summary["best_validation_loss"],
        "device": summary["device"],
        "mixed_precision": summary["mixed_precision"],
        "tokenizer_vocab_size": summary["tokenizer_vocab_size"],
        "output_dir": summary["output_dir"],
    }


def train_model(config: dict[str, Any]) -> dict[str, Any]:
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]

    device = choose_device(training_config.get("device", "auto"))
    mixed_precision = resolve_mixed_precision(training_config.get("mixed_precision", "auto"), device)

    training_tokenizer, tokenizer_metadata = prepare_training_tokenizer(
        tokenizer_name=model_config["tokenizer_name"],
        selfies_vocab_path=model_config["selfies_vocab_path"],
    )
    decoder_tokenizer = build_decoder_tokenizer(
        base_tokenizer_name=model_config["base_tokenizer_name"],
        training_tokenizer=training_tokenizer,
    )

    model = T5ForConditionalGeneration.from_pretrained(model_config["name"])
    embedding_size = model.get_input_embeddings().weight.size(0)
    if embedding_size != len(training_tokenizer):
        model.resize_token_embeddings(len(training_tokenizer))
    model.to(device)

    train_loader = build_dataloader(
        dataset_path=data_config["train_file"],
        tokenizer=training_tokenizer,
        batch_size=int(training_config["per_device_train_batch_size"]),
        max_source_length=int(data_config["max_source_length"]),
        max_target_length=int(data_config["max_target_length"]),
        num_workers=int(data_config.get("num_workers", 0)),
        shuffle=True,
        device=device,
    )
    validation_loader = build_dataloader(
        dataset_path=data_config["validation_file"],
        tokenizer=training_tokenizer,
        batch_size=int(training_config["per_device_eval_batch_size"]),
        max_source_length=int(data_config["max_source_length"]),
        max_target_length=int(data_config["max_target_length"]),
        num_workers=int(data_config.get("num_workers", 0)),
        shuffle=False,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    gradient_accumulation_steps = int(training_config["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_training_steps = max(1, int(training_config["num_epochs"]) * steps_per_epoch)
    warmup_steps = int(total_training_steps * float(training_config.get("warmup_ratio", 0.0)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    use_grad_scaler = device.type == "cuda" and mixed_precision == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    output_dir = Path(training_config["output_dir"])
    ensure_dir(output_dir / "checkpoints")
    dump_yaml(output_dir / "resolved_config.yaml", config)
    write_json(output_dir / "tokenizer_metadata.json", tokenizer_metadata)

    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    global_step = 0

    tracker: BaseTracker = NullTracker()
    try:
        tracker = build_tracker(
            config,
            stage_name="sft_chebi20",
            output_dir=output_dir,
        )
        tracker.log_config(_build_sft_tracking_config_payload(config, output_dir=output_dir))

        for epoch in range(1, int(training_config["num_epochs"]) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            train_loss_sum = 0.0
            train_examples = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            for batch_index, batch in enumerate(progress_bar, start=1):
                tensor_batch = move_tensor_batch_to_device(batch, device)
                batch_size = tensor_batch["input_ids"].size(0)

                with autocast_context(device, mixed_precision):
                    outputs = model(**tensor_batch)
                    raw_loss = outputs.loss
                    loss = raw_loss / gradient_accumulation_steps

                if use_grad_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                train_loss_sum += raw_loss.item() * batch_size
                train_examples += batch_size

                should_step = (
                    batch_index % gradient_accumulation_steps == 0
                    or batch_index == len(train_loader)
                )

                if should_step:
                    if use_grad_scaler:
                        scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), float(training_config["max_grad_norm"]))

                    if use_grad_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % int(training_config["log_every"]) == 0:
                        average_train_loss = train_loss_sum / max(train_examples, 1)
                        progress_bar.set_postfix(
                            train_loss=f"{average_train_loss:.4f}",
                            lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        )

            average_train_loss = train_loss_sum / max(train_examples, 1)
            validation_loss = evaluate_loss(model, validation_loader, device, mixed_precision)

            epoch_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": average_train_loss,
                "validation_loss": validation_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            history.append(epoch_metrics)
            write_json(output_dir / "history.json", history)
            tracker.log_metrics(epoch_metrics, step=global_step, prefix="sft")

            save_checkpoint(
                checkpoint_dir=output_dir / "checkpoints" / "last",
                model=model,
                training_tokenizer=training_tokenizer,
                decoder_tokenizer=decoder_tokenizer,
                config=config,
                metrics=epoch_metrics,
            )

            if epoch % int(training_config["save_every_epochs"]) == 0:
                save_checkpoint(
                    checkpoint_dir=output_dir / "checkpoints" / f"epoch-{epoch:02d}",
                    model=model,
                    training_tokenizer=training_tokenizer,
                    decoder_tokenizer=decoder_tokenizer,
                    config=config,
                    metrics=epoch_metrics,
                )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                save_checkpoint(
                    checkpoint_dir=output_dir / "checkpoints" / "best",
                    model=model,
                    training_tokenizer=training_tokenizer,
                    decoder_tokenizer=decoder_tokenizer,
                    config=config,
                    metrics=epoch_metrics,
                )

        summary = {
            "device": str(device),
            "mixed_precision": mixed_precision,
            "tokenizer_vocab_size": len(training_tokenizer),
            "best_validation_loss": best_validation_loss,
            "history": history,
            "output_dir": str(output_dir),
        }
        write_json(output_dir / "run_summary.json", summary)
        tracker.log_summary(_build_sft_tracking_summary(summary), prefix="sft")
        tracker.finish(status="success")
        return summary
    except Exception:
        tracker.finish(status="failed")
        raise
