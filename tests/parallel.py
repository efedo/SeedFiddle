"""Isolated full-suite workers with exhaustive assignment and fail-closed results."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from tests.selection import ACCEPTANCE_TARGETS, matches


# The 19 October full-run journals put 225s of native work on the slow
# third worker while the desktop worker finished early. Move whole modules so
# their class/catalogue fixtures remain together. Two-worker runs are unchanged.
BALANCED_NATIVE_MODULES = frozenset((
    'tests.test_native_core_recovery_recipe',
    'tests.test_native_calibrated_range_recipe',
    'tests.test_native_paper_replay',
))


def assignments(groups, jobs=3):
    cpu={test.id() for test in groups['cpu-qt']}
    buckets=[[],[],[]]
    for test in [*groups['cpu-qt'],*groups['cuda'],*groups['acceptance']]:
        name=test.id()
        numerical=(name.startswith(('tests.test_plantcv_', 'tests.test_reference_',
                    'tests.test_smpts_paper.', 'tests.test_core_recovery.'))
                    or matches(name,ACCEPTANCE_TARGETS))
        balanced = jobs == 3 and name.rsplit('.', 2)[0] in BALANCED_NATIVE_MODULES
        buckets[0 if name in cpu or balanced else 1 if numerical else 2].append(name)
    if jobs==2:buckets=[buckets[0]+buckets[1],buckets[2]]
    elif jobs!=3:raise ValueError('Parallel full runs require two or three workers')
    flat=[name for bucket in buckets for name in bucket]
    expected=[test.id() for group in groups.values() for test in group]
    if len(flat)!=len(set(flat)) or set(flat)!=set(expected):
        raise ValueError('Worker assignment must include every discovered unit exactly once')
    return [sorted(bucket) for bucket in buckets if bucket]


def merge(assignments, reports, returncodes):
    expected={name for bucket in assignments for name in bucket}
    record=dict(tests=0,passed=True,failures=[],errors=[],skips=[],discovery_errors=[],
                test_durations=[],fixture_durations=[],scenario_durations=[],
                reference_fixture={},compilation_fixture={},recipe_acceptance_scope={})
    accounted=[]
    for index,(ids,report,code) in enumerate(zip(assignments,reports,returncodes,strict=True)):
        valid=(report is not None and report.get('selection',{}).get('selected_ids')==ids
               and report.get('tests')==len(ids))
        if not valid:
            record['passed']=False
            record['errors'].append('worker '+str(index)+': incomplete or inconsistent result')
        if report is None:continue
        if valid:accounted.extend(ids)
        record['tests']+=report.get('tests',0)
        record['passed'] &= (report.get('passed',False) and code==0
            and not any(report.get(key,[]) for key in ('failures','errors','discovery_errors')))
        for key in ('failures','errors','skips','discovery_errors','test_durations','fixture_durations'):
            record[key].extend(report.get(key,[]))
        record['scenario_durations'].extend(row for row in report.get('scenario_durations',[])
            if row['unit'] in ids)
        for key in ('reference_fixture','compilation_fixture'):
            for name,value in report.get(key,{}).items():
                if isinstance(value,(int,float)):record[key][name]=record[key].get(name,0)+value
        record['recipe_acceptance_scope'].update(report.get('recipe_acceptance_scope',{}))
    record['all_discovered_units_accounted_for']=(len(accounted)==len(expected)
        and len(accounted)==len(set(accounted)) and set(accounted)==expected)
    record['passed'] &= record['all_discovered_units_accounted_for']
    return record


def stop_process_tree(child):
    if child.poll() is not None:return
    if os.name=='nt':
        subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
    else:
        import signal
        os.killpg(child.pid,signal.SIGKILL)
    child.wait(timeout=30)


def run(groups, *, root, jobs, deadline):
    buckets=assignments(groups,jobs)
    folder=Path(root)/'artifacts/test-runs'/uuid4().hex
    folder.mkdir(parents=True)
    children=[];streams=[];paths=[]
    try:
        for index,ids in enumerate(buckets):
            stem=folder/f'worker-{index}'
            manifest=stem.with_suffix('.manifest.json');manifest.write_text(json.dumps(ids))
            stdout=stem.with_suffix('.stdout.log').open('w',encoding='utf-8')
            stderr=stem.with_suffix('.stderr.log').open('w',encoding='utf-8')
            streams.extend((stdout,stderr))
            report=stem.with_suffix('.json');journal=stem.with_suffix('.journal.jsonl')
            command=[sys.executable,'-u','-X','faulthandler','-m','tests.worker',
                str(manifest),str(report),str(journal)]
            child=subprocess.Popen(command,cwd=root,stdout=stdout,stderr=stderr,
                **({'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt'
                   else {'start_new_session':True}))
            children.append(child);paths.append((report,journal,stem.with_suffix('.stderr.log')))
            print(f'Worker {index}: {len(ids)} units; {paths[-1][2]}',file=sys.stderr,flush=True)
        timed_out=False
        while any(child.poll() is None for child in children):
            if time.perf_counter()>=deadline:
                timed_out=True
                for child in children:stop_process_tree(child)
                break
            time.sleep(.1)
        reports=[]
        for report,_,_ in paths:
            try:reports.append(json.loads(report.read_text(encoding='utf-8')))
            except (OSError,ValueError):reports.append(None)
        record=merge(buckets,reports,[child.returncode for child in children])
        record['parallel_workers']=[dict(ids=ids,report=str(report),journal=str(journal),
            log=str(log),returncode=child.returncode) for ids,(report,journal,log),child
            in zip(buckets,paths,children,strict=True)]
        record['deadline_exceeded']=timed_out
        record['passed'] &= not timed_out
        return record
    finally:
        for child in children:stop_process_tree(child)
        for stream in streams:stream.close()
