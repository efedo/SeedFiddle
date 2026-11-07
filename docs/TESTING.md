# Running and maintaining the tests

The suite is plain `unittest`, run through a parallel runner in `tests/__main__.py`.
Related scenarios are grouped into *functional units* (one `test_workflow_lifecycle`
or `test_validation_and_recovery` method that runs several private scenarios). The
suite has 317 units.

## Commands

```powershell
.venv\Scripts\python.exe -m tests                         # core subset (16 units)
.venv\Scripts\python.exe -m tests tests.test_gui_layout   # one module, class or method
.venv\Scripts\python.exe -m tests --tier cpu-qt           # an explicit tier
.venv\Scripts\python.exe -m tests --full                  # everything, unfiltered
.venv\Scripts\python.exe -m tests --full --list           # show selection without running
.venv\Scripts\python.exe -m tests --full --preflight      # check prerequisites only
```

Useful options: `-v`, `-f` (stop at the first failure in a focused run),
`--durations N` (slowest N; 0 for all), `--jobs 1|2|3` (worker processes; 3 by
default), `--report artifacts/<new-file>.json` (timing and result JSON; must be a
new file under `artifacts/`), `--stall-seconds S` and `--memory-limit-gb G`.

## Memory limit per test

On Windows every test may commit at most 8 GiB beyond what its process held when
the test started (`--memory-limit-gb`, or `SEEDFIDDLE_TEST_MEMORY_GB`; 0 disables).
Going over makes the allocation fail inside that test, which reports the traceback,
and the run lists it under "Over the per-test memory limit". Windows charges CUDA
allocations to the process as well, so the budget covers GPU and host memory
together. 8 GiB is the GPU's size; beyond it the NVIDIA driver spills CUDA
allocations into system memory instead of failing. The runner starts CUDA and
empties Torch's GPU cache before each test, so neither counts against it. The
heaviest analyses need about 7 GiB: the 27-megapixel dense dish in
`tests.test_pilot_analysis`, and 12-megapixel photos without a vessel, which are
analysed whole. `tests/memory_limit.py` sets the limit with a job object;
elsewhere none is set.

Also run `scripts\generate_node_catalogue.py --check` after changing nodes or
parameters; CI runs it.

## Tiers

`tests/selection.py` defines the selections. Full discovery is always the union.

| Tier | Units | Needs |
|---|---:|---|
| `core` (default) | 16 | Overlaps the others; quick routine check. |
| `cpu-qt` | 171 | Qt offscreen, no GPU. Hosted CI runs this. |
| `cuda` | 138 | CUDA device. Every test not in `cpu-qt` lands here, so new tests cannot escape full runs. |

## How a full run works

`--full` discovers every test and assigns each unit to one of three isolated worker
processes. Each worker writes a journal as units start and finish. The controller
reconciles journals and results with the discovered list; a run passes only if
every unit is accounted for and none failed. A crashed or incomplete worker
cannot produce a pass.

- **Watchdog.** A worker that writes no journal entry for `--stall-seconds`
  (default 900) is stopped and the run fails, naming the last unit it started.
- **Time budget is advisory.** The report records whether the run stayed under
  1,000 units and 900 seconds and prints a warning if not; it does not change the
  verdict.
- **Windows are deleted between tests.** Closing a window only hides it, and
  connections to its own methods keep it alive. `tests/functional_units.py` therefore
  deletes the parentless widgets each scenario and test leaves behind. Windows a
  class fixture opens last until that class ends.
- **Workers end without interpreter shutdown.** Once its report and journal are
  on disk, a worker ends itself with its result code (`tests/worker.py`). On Windows
  it uses `TerminateProcess`, because `os._exit` still unloads DLLs and that crashes
  after a long run. As a result, a crash during worker shutdown cannot fail a run,
  and temporary directories that tests leave open stay in `artifacts\test-temp\`
  until pruned.
- **Skips.** A case-sensitive filesystem check is expected to skip on Windows.
  Tests whose committed images are absent from `images/` skip rather than fail.
- Worker logs and journals go to `artifacts\test-runs\<run-id>\`; temporary
  directories to `artifacts\test-temp\`. Clear old ones with
  `scripts\prune_artifacts.py --older-than 7 --delete` (dry run without `--delete`).

Run full suites from the main checkout and one at a time: two full runs sharing a
GPU can stall a worker or push slow units past their timing limits.

## Prerequisites

Full and `cuda` runs need a CUDA device; `--preflight` checks that without running
tests. Tests read only committed inputs (`images/`, `config/`) and write under
`artifacts\test-temp\`, so no ignored file is needed to run them.

Python 3.12 is the application baseline. Python 3.15 is permitted by the runtime
gate when compatible dependencies are already installed, but the baseline pins are
not guaranteed to provision a fresh Python 3.15 environment.

## Writing tests

- Put new desktop-only tests in `cpu-qt` by adding them to `CPU_QT_TARGETS`;
  otherwise they run in `cuda`.
- Use `tests.review_fixtures.isolated_window` for a main window with a private
  project root, and `dispose_window` in cleanup. `workspace_window` keeps the real
  application settings.
- Group related scenarios with the `@functional_units` decorator and
  `self.scenario(...)`, as in `tests/test_gui_layout.py`; each scenario keeps its own
  fixture state.
- A window that must outlive one test belongs in `setUpClass`; any other window a
  test leaves open is deleted when it ends.
- Poll for asynchronous Qt state with a bounded loop rather than a fixed sleep.
- Avoid raising from `QTimer.singleShot(0, callback)`, in tests as in application
  code, which uses `seedvision.ui.deferred.run_soon`. Inside `processEvents()` or a
  static dialog, PySide stores that exception, and graphics items painting for the
  first time can then crash the process ("pure virtual method
  'QGraphicsObject.paint'", then an access violation or abort). A window deleted
  before its timer fires raises in exactly this way.
- `tests/test_ui_text_limits.py` keeps command labels at 45 characters or fewer and
  tooltips at 450 or fewer.

## CI

[.github/workflows/tests.yml](../.github/workflows/tests.yml) runs the `cpu-qt`
tier and the catalogue check on hosted Windows for every push and pull request.
A manually dispatched job runs `--full` on a self-hosted Windows CUDA runner, which
keeps its `.venv/` between runs.

## Inventory

[testing/TEST_INVENTORY.md](testing/TEST_INVENTORY.md) is a generated,
source-derived description of every discovered unit. Regenerate it with:

```powershell
.venv\Scripts\python.exe tools\audit_test_inventory.py --collect --without-history
```

[archive/TEST_SUITE_REDESIGN_2026_10_18.md](archive/TEST_SUITE_REDESIGN_2026_10_18.md) records the
18 October 2026 consolidation from 2,823 methods to functional units.
