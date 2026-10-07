# data-flywheel

Monorepo for data collection, training and evaluating **action models** (system 1, VLA policies) and **agentic models** (system 2).

**Architecture:** [Data Flywheel diagram](https://cnboonhan.github.io/data-flywheel/architecture.html) ([source](architecture.html))

## Clone

```bash
git -c url."https://github.com/".insteadOf=git@github.com: \
    clone --recurse-submodules --jobs 8 https://github.com/cnboonhan/data-flywheel.git
```

- IsaacLab-Arena pins two of its submodules (IsaacLab, Isaac-GR00T) with SSH URLs. The `-c` option fetches them over HTTPS, so no GitHub SSH key is needed.
- Install [Git LFS](https://git-lfs.com/) (`git lfs install`) first. Several submodules store assets in LFS; without it they check out as pointer files.
- The full recursive clone is several GB. To fetch one benchmark only, clone without `--recurse-submodules`, then run `git submodule update --init --recursive <path>`, e.g. `eval/system1/RoboDojo`. Keep the same `-c` option when that path is `eval/system2/IsaacLab-Arena`.

| Folder | Contents |
|---|---|
| [`datasets/`](datasets/README.md) | Downloaded datasets (gitignored) |
| [`checkpoints/`](checkpoints/README.md) | Downloaded checkpoints (gitignored) |
| [`eval/`](eval/README.md) | Benchmarks for action models (system 1) and agentic models (system 2) (git submodules) |
| [`sensors/`](sensors/README.md) | Data-collection hardware (git submodules) |
| [`sim/`](sim/README.md) | Building sim scenes from real spaces with NuRec |
| [`scripts/`](scripts/README.md) | Data/checkpoint downloader, Arena environment generation |

- **Download data and checkpoints:** see [`scripts/`](scripts/README.md#download-data-and-checkpoints).
- **Evaluation:** see [`eval/`](eval/README.md).
