# eval

The evaluation benchmark, a git submodule. Fetch it:

```bash
git submodule update --init --recursive eval/system1/RoboDojo
```

| Benchmark | Path | Simulator | Env |
|---|---|---|---|
| [RoboDojo](https://github.com/robodojo-benchmark/RoboDojo) | [`system1/RoboDojo`](system1/README.md) | Isaac Sim 5.1 | conda `RoboDojo` |

- [`system1/`](system1/README.md): action models (VLA policies)

**Notes**
- **Driver:** NVIDIA 580-open is required, because Isaac Sim 5.1 crashes on 595. It's held with `apt-mark hold`.
- **Upstream code:** don't commit inside the submodule. Changes belong on a fork; this repo only records its commit.
