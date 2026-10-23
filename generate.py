import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from data_utils import load_dataset
from model_utils import ModelWrapper, build_model
from prompt_utils import apply_postprocessors, apply_prompt


def handle_example(
    sample: dict,
    completion_type: str,
    mode: str,
    model_wrapper: ModelWrapper,
    cache: dict,
    post_processors: Iterable[str]
):
    prompt = apply_prompt(sample, completion_type, mode, model_wrapper)
    completion = model_wrapper.invoke_cached(prompt, cache)
    completion = apply_postprocessors(completion, sample, completion_type, post_processors)
    result = {
        "task_id": sample["task_id"],
        "completion": completion
    }
    if mode in {"infilling", "reverse_infilling"}:
        result["context"] = model_wrapper.last_context
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("completion_type", type=str)
    parser.add_argument("cache_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("mode", type=str)
    parser.add_argument("--block_comments", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; each prompt gets a stable derived seed")
    parser.add_argument("--model-revision", help="Immutable native-adapter model/tokenizer commit; defaults to the registry pin")
    parser.add_argument("--context-limit", type=int, help="Native-adapter context limit, at most the model configuration limit")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Native-adapter device (default cuda)")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], help="Native-adapter weight dtype (default float16)")
    parser.add_argument("--post_processors", type=str, nargs="+")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Requested output token cap (historical default: 128)")
    parser.add_argument("--api-timeout", type=float, default=120.0, help="Remote request timeout in seconds per attempt")
    parser.add_argument("--api-max-retries", type=int, default=2, help="SDK retries after the initial remote attempt")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Atomic cache checkpoint interval; output rows flush individually")

    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.api_timeout <= 0 or args.api_max_retries < 0:
        parser.error("API timeout must be positive and retries nonnegative")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if Path(args.cache_path).resolve() == Path(args.output_path).resolve():
        parser.error("Cache and output paths must differ")

    if os.path.exists(args.cache_path):
        with open(args.cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    def save_json(path, value):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    manifest_path = args.output_path + ".run.json"
    manifest = dict(status="running", completed=0, expected=None, configuration=vars(args))
    save_json(manifest_path, manifest)
    try:
        with open(args.output_path, "w", encoding="utf-8") as f:
            samples = load_dataset(args.completion_type)
            manifest["dataset"] = getattr(samples, "provenance", None)
            manifest["expected"] = len(samples)
            model_wrapper = build_model(args)
            manifest["model"] = getattr(model_wrapper, "cache_configuration", None)
            for sample in tqdm(samples):
                manifest["active_task"] = sample["task_id"]
                item = handle_example(sample, args.completion_type, args.mode, model_wrapper, cache, args.post_processors)
                f.write(json.dumps(item) + "\n")
                f.flush()
                manifest["completed"] += 1
                save_json(manifest_path, manifest)
                if manifest["completed"] % args.checkpoint_every == 0:
                    save_json(args.cache_path, cache)
        manifest.pop("active_task", None)
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "incomplete"
        manifest["error_type"] = type(error).__name__
        parser.exit(130 if isinstance(error, KeyboardInterrupt) else 1,
                    f"Generation {manifest['status']}: {type(error).__name__}; completed rows are preserved\n")
    finally:
        save_json(args.cache_path, cache)
        save_json(manifest_path, manifest)


if __name__ == '__main__':
    main()
