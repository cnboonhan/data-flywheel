# data-flywheel

Monorepo for data collection, training and evaluating **action models** (system 1, VLA policies) and **agentic models** (system 2).

| Folder | Contents |
|---|---|
| [`datasets/`](datasets/README.md) | Downloaded datasets (gitignored) |
| [`checkpoints/`](checkpoints/README.md) | Downloaded checkpoints (gitignored) |
| [`eval/`](eval/README.md) | Benchmarks for action models (system 1) and agentic models (system 2) (git submodules) |
| [`sensors/`](sensors/README.md) | Data-collection hardware (git submodules) |
| [`scripts/`](scripts/README.md) | Data/checkpoint downloader, Arena environment generation |

- **Download data and checkpoints:** see [`scripts/`](scripts/README.md#download-data-and-checkpoints).
- **Evaluation:** see [`eval/`](eval/README.md).
