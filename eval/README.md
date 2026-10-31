# eval

The evaluation benchmark, RoboDojo, a git submodule for action models (system 1): [`system1/`](system1/README.md). Fetch it:

```bash
git submodule update --init --recursive eval/system1/RoboDojo
```

**Notes**
- **Driver:** NVIDIA 580-open is required, because Isaac Sim 5.1 crashes on 595. It's held with `apt-mark hold`.
- **Upstream code:** don't commit inside the submodule. Changes belong on a fork; this repo only records its commit.
