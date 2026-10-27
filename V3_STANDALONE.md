# Standalone SAFIM v3 benchmarking

Both `dataset` and `fix-bench-dataset` include the `safim_v3` Python package.
No parent repository, editable parent install, or custom `PYTHONPATH` is needed.
The bundle contains the HF loader, Docker runner/worker, versioned checkers,
metadata filters, metrics, and parser helper. Its source revision and exact file
hashes are recorded in `safim_v3/VENDORED.json`. `.gitattributes` protects pinned
checker and worker bytes from checkout line-ending conversion on Windows.

## Setup

Clone the desired branch and run commands from that clone's root:

```sh
git clone --branch dataset https://github.com/averkij/safim_fixed.git safim-dataset
cd safim-dataset
python -m pip install -r requirements-v3.txt
python -c "import safim_v3.hf_dataset; print(safim_v3.hf_dataset.__file__)"
```

Use `--branch fix-bench-dataset` and a separate folder for the fixed variant.
Use Python 3.12 for the maintained v3 environment. On `fix-bench-dataset`,
`requirements-v3.txt` includes its existing modern model/parser profile. On
`dataset`, it adds only v3 runtime dependencies: install your chosen model and
parser environment separately, following the branch README. This packaging change
does not replace historical model wrappers, parser settings or sampling behavior.
Historical environments described in the original README may require a different
Python version; record the environment actually used for each benchmark.

## Generation and evaluation

Use explicit dataset flags on both branches: the `dataset` branch retains its
historical default dataset source. Replace `MODEL` with a supported identifier.

```sh
python generate.py MODEL block_v3 cache-block.json predictions-block.jsonl infilling --dataset-source averoo/safim_v3 --dataset-revision 70edeef93658085e8ab905e5688996d098b8172e --post_processors truncate_line_until_block
python evaluate.py block_v3 predictions-block.jsonl results-block.jsonl --dataset-source averoo/safim_v3 --dataset-revision 70edeef93658085e8ab905e5688996d098b8172e --runtimes configs/runtimes.v3.json
python show_detailed_results.py block_v3 results-block.jsonl --dataset-source averoo/safim_v3 --dataset-revision 70edeef93658085e8ab905e5688996d098b8172e
```

For control completion, use `control_v3`, separate filenames and
`--post_processors truncate_control`. Keep caches and outputs separate across
branches. Exact-reference automatic passing remains enabled by default. The fixed
branch also supports `--force-execution` and its existing stricter run validation.

Docker must be running and the images in `configs/runtimes.v3.json` must be
available locally. This registry is copied from the pinned HF publication and
uses the original validation image digest. Its original image build recipe is
not recorded; cloning supplies the evaluator and registry, not that image.
For another machine, transfer the original image with `docker save`/`docker load`,
or provide a separately validated registry via `--runtimes`. Changing the image is
an execution-environment change and must be recorded in the experiment.

## Isolation smoke check

After installing runtime dependencies, run:

```sh
python -I tests/standalone_v3_smoke.py
python -I tests/standalone_v3_smoke.py --docker
```

The check constructs tiny local fixtures for both configs, verifies all vendored
hashes and checker paths, loads metadata, exercises filters/reporting helpers,
and evaluates both exact and non-exact completions. The Docker option executes a
Python completion through the actual harness using the pinned registry. It does
not download models or the full dataset. Imported v3 modules must resolve inside
the clone, including when Python ignores external `PYTHONPATH` settings.

To update the bundle, copy the dependency files byte-for-byte from a reviewed
parent revision, update `VENDORED.json`, and apply the identical bundle to both
benchmark branches. Do not silently change checker versions or worker bytes.
