"""Internal isolated unittest worker; the full controller owns its manifest."""
from pathlib import Path
import json
import sys
import time
import unittest


def main(argv=None):
    manifest,report,journal=map(Path,sys.argv[1:] if argv is None else argv)
    root=Path(__file__).resolve().parents[1]
    if any(not path.resolve().is_relative_to(root/'artifacts') for path in (manifest,report,journal)):
        raise ValueError('Worker files must remain inside artifacts/')
    ids=json.loads(manifest.read_text(encoding='utf-8'))
    if (not isinstance(ids,list) or not ids or len(ids)!=len(set(ids))
            or any(not isinstance(name,str) or not name.startswith('tests.') for name in ids)):
        raise ValueError('Invalid worker unit manifest')
    selected=set(ids);owner=None
    class JournalResult(unittest.TextTestResult):
        def startTest(self,test):
            nonlocal owner
            if owner is None:owner=self
            self.started=time.perf_counter()
            if self is owner:
                # The controller's stall watchdog names this unit if it hangs.
                with journal.open('a',encoding='utf-8') as output:
                    output.write(json.dumps(dict(id=test.id(),event='start'))+'\n')
            super().startTest(test)
        def stopTest(self,test):
            if self is owner:
                with journal.open('a',encoding='utf-8') as output:
                    output.write(json.dumps(dict(id=test.id(),event='stop',seconds=time.perf_counter()-self.started,
                        failures=len(self.failures),errors=len(self.errors),skips=len(self.skipped)))+'\n')
            super().stopTest(test)
        def addSubTest(self,test,subtest,err):
            super().addSubTest(test,subtest,err)
            if err is not None and self is owner:self.stream.writeln(self._exc_info_to_string(err,subtest))
    unittest.TextTestRunner.resultclass=JournalResult
    from tests import functional_units
    class ScenarioJournal(list):
        def append(self,row):
            super().append(row)
            if row['unit'] in selected:
                with journal.with_suffix('.scenarios.jsonl').open('a',encoding='utf-8') as output:
                    output.write(json.dumps(row)+'\n')
    functional_units.SCENARIO_DURATIONS=ScenarioJournal()
    from tests.__main__ import main as run
    return run([*ids,'-v','--jobs','1','--report',str(report)])


if __name__=='__main__':raise SystemExit(main())
