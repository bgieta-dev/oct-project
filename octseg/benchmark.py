"""Inference time and VRAM benchmarking for SegFormer and SwinUNETR 3D models."""
import argparse
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional

import albumentations as A
import numpy as np
import torch
from transformers import SegformerImageProcessor
import yaml

from octseg.config import Config, load_config, save_config
from octseg.dataset import OCTDataset
from octseg.evaluate3d import predict_volume
from octseg.hybrid_inference import HybridInference
from octseg.model import load_segformer, load_swin_unetr
from octseg.postprocess import apply_thresholds, clean_regions, predict_logits, sharpen_ped
from octseg.runs import make_run_dir, setup_logging
from octseg.splits import split_files
from octseg.volume_data import group_volumes, load_volume

log = logging.getLogger(__name__)


@torch.no_grad()
def time_calls(fn: Callable[[Any], Any], inputs: List[Any], warmup: int, device: torch.device) -> Dict[str, Any]:
    """Times one call per input after warming up on inputs[0].

    Args:
        fn: callable accepting one element from inputs.
        inputs: list of preloaded inputs.
        warmup: number of warmup iterations on inputs[0].
        device: device on which fn executes.

    Returns:
        dict with keys mean_ms, median_ms, p95_ms, n.
    """
    if not inputs:
        raise ValueError("inputs must not be empty")

    if warmup > 0:
        for _ in range(warmup):
            fn(inputs[0])
        if device.type == "cuda":
            torch.cuda.synchronize()

    times: List[float] = []
    for inp in inputs:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(inp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "n": len(times),
    }


@torch.no_grad()
def peak_vram_mb(fn: Callable[[Any], Any], x: Any, device: torch.device) -> Optional[float]:
    """Measures peak allocated VRAM in megabytes during fn(x), or None on CPU.

    Args:
        fn: callable accepting x.
        x: input element.
        device: target device.

    Returns:
        Peak allocated memory in MB as float, or None if device is not CUDA.
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        fn(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return None


def parse_args(cli_args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses benchmark CLI arguments."""
    parser = argparse.ArgumentParser(description="OCT segmentation inference time and VRAM benchmark")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--expert-config", default=None, help="Path to expert config YAML")
    parser.add_argument("--expert-weights", default=None, help="Path to expert model weights")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Device override")
    parser.add_argument("--threads", type=int, default=None, help="PyTorch CPU threads")
    parser.add_argument("--slices", type=int, default=50, help="Number of test slices for 2D models")
    parser.add_argument("--volumes", type=int, default=2, help="Number of test volumes for 3D models")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--name", default="bench", help="Run directory name suffix")
    parser.add_argument("--output", default=None, help="Explicit output directory")

    args = parser.parse_args(cli_args)
    if (args.expert_config is None) != (args.expert_weights is None):
        parser.error("--expert-config and --expert-weights must be used together")
    return args


@torch.no_grad()
def benchmark(
    cfg: Config,
    model_path: str,
    output_dir: Path,
    expert_cfg: Optional[Config] = None,
    expert_weights_path: Optional[str] = None,
    slices: int = 50,
    volumes: int = 2,
    warmup: int = 5,
) -> Dict[str, Any]:
    """Runs latency and VRAM benchmark across supported pipeline modes.

    Args:
        cfg: resolved model configuration.
        model_path: path to checkpoint weights.
        output_dir: directory to write benchmark.yaml.
        expert_cfg: optional configuration for IRF expert model.
        expert_weights_path: optional path to IRF expert weights.
        slices: number of slices to test for 2D models.
        volumes: number of volumes to test for 3D models.
        warmup: number of warmup calls.

    Returns:
        Benchmark results dictionary.
    """
    device = cfg.DEVICE
    modes_dict: Dict[str, Any] = {}
    mean_z: Optional[float] = None

    if cfg.ARCH == "segformer":
        log.info(f"Loading SegFormer model from {model_path}")
        model = load_segformer(cfg, model_path, output_attentions=False)
        params_m = round(float(sum(p.numel() for p in model.parameters()) / 1e6), 2)

        test_imgs, test_masks = split_files(cfg, "test")
        if not test_imgs:
            raise RuntimeError("No test slices found in split.")

        slice_count = min(slices, len(test_imgs))
        log.info(f"Preloading {slice_count} test slices into memory...")

        processor = SegformerImageProcessor.from_pretrained(cfg.MODEL_NAME)
        val_transform = A.Compose([A.Resize(height=cfg.AUG_SIZE[0], width=cfg.AUG_SIZE[1])])
        dataset = OCTDataset(
            test_imgs[:slice_count],
            test_masks[:slice_count],
            processor=processor,
            transform=val_transform,
            cfg=cfg,
        )

        pixel_values_list = []
        orig_imgs_list = []
        for i in range(len(dataset)):
            item = dataset[i]
            pixel_values_list.append(item["pixel_values"].unsqueeze(0).to(device))
            orig_imgs_list.append(item["orig_img"])

        # Mode forward
        modes_dict["forward"] = (lambda px: model(pixel_values=px), pixel_values_list)

        # Mode pipeline
        def run_pipeline(px):
            logits = predict_logits(model, px, cfg.AUG_SIZE, cfg)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            sharpen_ped(probs, factor=cfg.PED_SHARPEN_FACTOR)
            pred = apply_thresholds(probs, cfg.CLASS_THRESHOLDS, irf_override=True)
            if cfg.MIN_REGION_SIZE > 0:
                pred = clean_regions(pred, cfg.MIN_REGION_SIZE, irf_open=True)
            return pred

        modes_dict["pipeline"] = (run_pipeline, pixel_values_list)

        # Mode hybrid (if expert provided)
        if expert_cfg is not None and expert_weights_path is not None:
            log.info(f"Initializing HybridInference with expert weights {expert_weights_path}...")
            hybrid_engine = HybridInference(
                base_model_path=model_path,
                expert_model_path=expert_weights_path,
                cfg=cfg,
                expert_cfg=expert_cfg,
            )
            modes_dict["hybrid"] = (lambda img: hybrid_engine.segment(img), orig_imgs_list)

    elif cfg.ARCH == "swin_unetr_3d":
        if expert_cfg is not None or expert_weights_path is not None:
            raise ValueError("Expert model is not supported for ARCH=swin_unetr_3d")

        log.info(f"Loading SwinUNETR 3D model from {model_path}")
        model = load_swin_unetr(cfg, model_path)
        params_m = round(float(sum(p.numel() for p in model.parameters()) / 1e6), 2)

        test_imgs, test_masks = split_files(cfg, "test")
        records = group_volumes(test_imgs, test_masks)
        if not records:
            raise RuntimeError("No test volumes found in split.")

        vol_count = min(volumes, len(records))
        log.info(f"Preloading {vol_count} test volumes at {cfg.SWIN_VOLUME_SIZE} into memory...")
        images_u8 = [load_volume(r, cfg.SWIN_VOLUME_SIZE)[0] for r in records[:vol_count]]
        mean_z = float(np.mean([img.shape[0] for img in images_u8]))

        # Mode volume_forward
        modes_dict["volume_forward"] = (lambda img: predict_volume(model, img, cfg, tta=False), images_u8)

        # Mode volume_pipeline
        def run_volume_pipeline(img):
            probs = predict_volume(model, img, cfg, tta=cfg.USE_TTA)
            preds = []
            for z in range(len(probs)):
                p_slice = probs[z]
                sharpen_ped(p_slice, factor=cfg.PED_SHARPEN_FACTOR)
                pred = apply_thresholds(p_slice, cfg.CLASS_THRESHOLDS, irf_override=True)
                if cfg.MIN_REGION_SIZE > 0:
                    pred = clean_regions(pred, cfg.MIN_REGION_SIZE, irf_open=True)
                preds.append(pred)
            return preds

        modes_dict["volume_pipeline"] = (run_volume_pipeline, images_u8)

    else:
        raise ValueError(f"Unsupported ARCH: {cfg.ARCH}")

    modes_results: Dict[str, Any] = {}
    for mode_name, (fn, inputs) in modes_dict.items():
        log.info(f"Benchmarking mode '{mode_name}' with {len(inputs)} inputs (warmup={warmup})...")
        vram = peak_vram_mb(fn, inputs[0], device)
        stats = time_calls(fn, inputs, warmup=warmup, device=device)
        stats["peak_vram_mb"] = vram
        if mean_z is not None:
            stats["ms_per_bscan"] = float(stats["mean_ms"] / mean_z)

        modes_results[mode_name] = stats

        vram_str = f"{stats['peak_vram_mb']:.1f} MB" if stats["peak_vram_mb"] is not None else "null (CPU)"
        if "ms_per_bscan" in stats:
            log.info(
                f"Mode {mode_name:16s} | Mean: {stats['mean_ms']:.2f} ms | "
                f"Per B-scan: {stats['ms_per_bscan']:.2f} ms | "
                f"Median: {stats['median_ms']:.2f} ms | p95: {stats['p95_ms']:.2f} ms | VRAM: {vram_str}"
            )
        else:
            log.info(
                f"Mode {mode_name:16s} | Mean: {stats['mean_ms']:.2f} ms | "
                f"Median: {stats['median_ms']:.2f} ms | p95: {stats['p95_ms']:.2f} ms | VRAM: {vram_str}"
            )

    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda" and torch.cuda.is_available()
        else "CPU"
    )
    result = {
        "device": device.type,
        "device_name": device_name,
        "torch_threads": torch.get_num_threads(),
        "arch": cfg.ARCH,
        "params_m": params_m,
        "modes": modes_results,
    }

    yaml_path = output_dir / "benchmark.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(result, f, sort_keys=False)
    log.info(f"Benchmark results written to {yaml_path}")

    return result


def main(cli_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """CLI entry point for benchmark."""
    args = parse_args(cli_args)

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.DEVICE_OVERRIDE = args.device

    expert_cfg = None
    if args.expert_config is not None:
        expert_cfg = load_config(args.expert_config)
        if args.device is not None:
            expert_cfg.DEVICE_OVERRIDE = args.device

    run_dir = Path(make_run_dir("benchmark", name=args.name, output=args.output))
    setup_logging(run_dir, "benchmark.log")
    save_config(cfg, run_dir / "config.yaml")
    if expert_cfg is not None:
        save_config(expert_cfg, run_dir / "expert_config.yaml")

    log.info(f"Starting benchmark for ARCH={cfg.ARCH} on device={cfg.DEVICE}")

    return benchmark(
        cfg=cfg,
        model_path=args.model,
        output_dir=run_dir,
        expert_cfg=expert_cfg,
        expert_weights_path=args.expert_weights,
        slices=args.slices,
        volumes=args.volumes,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
