# Functional test consolidation — 18 October 2026

19 October workflow-repair verification: **640 units, 636 passes, four existing
skips, zero failures/errors in 806.689 seconds** (806.821 parent seconds). The
additional unit exercises real local-reference optimization. Eight existing units
from three whole modules now run on the previously underused worker, preserving
their shared fixtures and every discovered ID. The three assignments contain
159, 193 and 288 units; two-worker scheduling is unchanged. The fresh full run
passes both budgets, and the numerical baselines remain byte-identical. Evidence:
`artifacts/workflow-repairs-full-v2.json` and its `.receipt.json` companion. The
earlier V1 attempt reached its deadline with two identified assertion failures;
its incomplete report and journals remain separate from this passing run.

The 18 October consolidation checkpoint below retains its original scope.

The target is fewer than 1,000 discovered functional units and at most 900 seconds
for one fresh, unfiltered canonical full run. **Accepted: 639 units, 635 passes, four existing skips, 762.295 seconds** (762.822 parent seconds).
The previous 126.5-minute figure was recorded work across completed segments,
not uninterrupted wall time; it is not a directly comparable wall-time baseline.

## Inventory before implementation

The complete starting inventory records **2,823 methods in 248 files**, including
their functionality, assertions, helper calls, fixtures, historical outcomes and
timings. It was saved before source consolidation:

- [Every original method](consolidation-2026-10-18/before-TEST_INVENTORY.md)
- [Machine-readable original inventory](consolidation-2026-10-18/before-test_inventory.json)
- [Every original method decision](consolidation-2026-10-18/TEST_DISPOSITIONS.md)
- [Machine-readable decisions](consolidation-2026-10-18/dispositions.json)
- [Original-to-current method mapping](consolidation-2026-10-18/method_mapping.json)
- [Current discovered units and scenario evidence](TEST_INVENTORY.md)

Static call overlap helps locate duplication. It does not establish statement,
branch or semantic coverage equivalence. The register makes no measured coverage
percentage claim.

## Passes and decisions

| Pass | Finding | Implementation |
| --- | --- | --- |
| Complete inventory | Thousands of tiny methods mixed basic guards with repeated producer/export/replay lifecycles. | Inventory every method first; retain an explicit destination or retirement decision for each. |
| Functional overlap | Recipe variants repeatedly executed identical operators; several adapter suites repeated generic archive and protocol faults. | Retire repeated integration executions and group related checks within each existing functional class. Preserve independent numerical primitives, boundary checks and end-to-end witnesses. |
| Expensive fixtures | A representative public chain spent 46 of 54 profiled seconds starting or waiting for interpreters. | Reuse third-party imports in a test-only PlantCV process. Reload PlantCV itself for every request; execute the original worker and its request/source/array checks each time. Keep actual launcher/isolation tests on one-shot processes. |
| Calculation admission | The same immutable source was repeatedly compiled during real loaded-code checks. | Cache only immutable code objects, keyed by exact source bytes, filename and compiler options. Retain actual source reads, AST parsing, loaded binding/default checks and mutation rejection. Mutable ASTs bypass the cache. |
| Numerical matrix | 223 recipe permutations repeated complete cold/warm/export lifecycles. | Compare all 223 admitted recipe specifications exactly; calculate 19 representative workflows spanning every operation/port domain and input family. Compare their complete numerical records against the unchanged frozen baseline. |
| Manual family review | Cheap source-call overlap alone would have removed distinct control and boundary witnesses. | Restore even-kernel numerical decisions, range modes/endpoints, corrected-crop controls/cache, all extended-threshold control effects, image preparation recovery, thickness invalidation, and actual public backend adapters. |
| Runner review | Replacing a production launcher function conflicted with its loaded-code verification. | Substitute the standard-library process boundary in test infrastructure instead. Production functions, provenance guards and application behavior remain intact. |
| Measurement | Method names alone can conceal unchanged work. | Report real discovered unit count, actual subscenario timings, class fixtures, reference requests, compiler cache statistics, selected recipes and full wall time separately. |

## Retained obligations

Independent primitive numerical matrices retain their selected source bodies,
including exact pixels, floating-point boundaries, model/table identity and
public PlantCV comparisons. Complete pipeline witnesses retain cache reuse,
changed-node invalidation, backend choice, cancellation, mutation withholding,
private storage, archive corruption and source-absent replay checks. Full-size
memory/resource witnesses remain; repeating the same full-image calculation in
every adapter is unnecessary.

Functional units call explicit scenario helpers. Each scenario receives a fresh
case, original setup/teardown and its own cleanup stack. Mutable windows, analysis
caches, tensors, settings and temporary archives are not shared between scenarios.
Assertions and exceptions remain ordinary unittest subtest outcomes. Failure,
cleanup-error, skip and private-state behavior have a regression witness.

The original 223-way **numerical integration cross-product is deliberately
reduced**. All specifications remain exhaustive; numerical calculations now use
operation/port/input-family representatives plus the independent primitive and
control-effect tests. This preserves meaningful numerical and regression
evidence without claiming identical coverage for every former permutation.

The four frozen baseline files are byte-identical. Production-source provenance
remains byte-exact. Historical whole-test-file pins are bridged only by a checked
AST digest of their numerical fixture definitions: imports, constants, helper
functions, setup and teardown stay exact while test organization can change.
See [fixture definition manifest](consolidation-2026-10-18/fixture_definitions.json).

## Execution and budget

```powershell
.venv\Scripts\python.exe -m tests --full -v --report artifacts/my-fresh-full-run.json
```

This uses ordinary unittest discovery with every discovered unit included. The
canonical runner owns the shared import/compiler fixtures and is also used by CI.
CPU/Qt, CUDA and acceptance tiers still form a complete, nonoverlapping partition;
they do not filter a full run. The canonical full command returns failure when
its discovered count exceeds 999 or its measured wall time exceeds 900 seconds,
even if its assertions pass. Hardware and load affect the time budget.

The existing pilot-fixture, opt-in image diagnostic and filesystem capability
skips remain explicit. No new capability skips were introduced to meet the budget.
Focused checks do not establish full-suite or runtime acceptance. The final full run accounts for every discovered unit exactly once. Its immutable result is
`artifacts/test-functional-full-v5.json`; the current formatting-reviewed source receipt is
`artifacts/test-functional-current-review-v1.json`.

## Final implementation and remaining costs

The full command now runs three isolated processes: desktop contracts, primitive/public
numerical/reference acceptance, and remaining workflows. A manifest assigns every discovered
unit exactly once. The parent validates complete per-worker identities/outcomes and enforces
the 900-second deadline, including process cleanup. CPU/Qt and other focused tiers remain
explicit single-process selections. No application process model changed.

132 classes explicitly reuse immutable factory catalogues. The full-resolution saved-stage
replay uses actual Run to node preparation for its threshold input, preserving the full-size
numerical and memory checks while avoiding an unrelated recovery execution. The original
full core-recovery numerical/memory case remains in the suite.

The completed workers took 186.822, 443.273 and 756.108 seconds; total parent wall time was
762.822 seconds. The retained full-resolution core/recovery case, KDE training, watershed
matrices, real pilot analyses and 19-profile acceptance are intentional remaining costs.
Their correctness/resource assertions were preserved. The budget is measured on the local
configured CUDA/reference environment; slower or incomplete environments fail rather than
silently omit units. The preceding serial V4 exceeded the goal and remains incomplete.

The post-run EOF cleanup changes no AST node, location, executable code or line/exception
table. Its exact before/after hashes are in `artifacts/test-functional-eof-proof-v1.json`;
the subsequent 10-unit metadata gate passes. These reviewed formatting changes are explicit
in the current source receipt. All four numerical baseline files remain byte-identical.
