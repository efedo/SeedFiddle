"""Reuse reference dependency imports in healthy functional calculation tests.

Results are never cached. Request IDs, private files, installed-source checks,
audited PlantCV imports, calculations and response validation execute each time.
Tests of interpreter startup, transport, isolation or source loading retain the
production one-shot launcher. The fixture changes no application code.
"""
from contextlib import contextmanager
import json
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
import tempfile
from threading import Event, Lock, Thread, local
import time
from unittest.mock import patch

ORIGINAL_SUBPROCESS_RUN = subprocess.run
ORIGINAL_POPEN = subprocess.Popen


class _ResponseStream:
    def __init__(self):
        self.ready=Event();self.data=b'';self.position=0;self.closed=False
    def finish(self,data):
        self.data=data;self.ready.set()
    def read(self,size=-1):
        self.ready.wait()
        end=len(self.data) if size<0 else self.position+size
        value=self.data[self.position:end];self.position+=len(value)
        return value
    def close(self):self.closed=True


class _RequestProcess:
    """Pipe-compatible request handle for the real bounded production reader."""
    def __init__(self,session,command,kwargs):
        self.stdout=_ResponseStream();self.stderr=_ResponseStream()
        self.returncode=None;self.finished=Event();self.cancelled=Event()
        def execute():
            try:
                code,out,err=session.run(list(command),cwd=kwargs['cwd'],env=kwargs['env'],
                    timeout_seconds=120,cancel_event=self.cancelled)
            except BaseException as error:
                code,out,err=2,b'',str(error).encode()
            self.stdout.finish(out);self.stderr.finish(err)
            self.returncode=code;self.finished.set()
        Thread(target=execute,daemon=True).start()
    def poll(self):return self.returncode
    def kill(self):self.cancelled.set()
    def terminate(self):self.kill()
    def wait(self,timeout=None):
        if not self.finished.wait(timeout):raise subprocess.TimeoutExpired('reference request',timeout)
        return self.returncode

class Session:
    def __init__(self, original):
        self.original = original
        self.original_run = ORIGINAL_SUBPROCESS_RUN
        self.children = {}
        self.probes = {}
        self.lock = Lock()
        self.bypass = local()
        self.stats = dict(worker_starts=0,actual_operations=0,probe_starts=0,probe_reuses=0,isolated_calls=0)
        self.temporary = tempfile.TemporaryDirectory(prefix='reference-session-')

    def close(self):
        for child, queue in self.children.values():
            child.stdin.close()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
            child.stdout.close()
        self.children.clear()
        self.temporary.cleanup()

    def isolated_witness(self):
        frame = sys._getframe(2)
        while frame is not None:
            name = frame.f_globals.get('__name__','')
            if name in ('tests.test_reference_runner','tests.test_reference_watershed_capture'):
                return True
            frame = frame.f_back
        return False

    def popen(self,command,**kwargs):
        candidate=isinstance(command,(list,tuple)) and len(command)>2
        worker=candidate and list(command[1:3])==['-m','seedvision.benchmarking.plantcv_worker']
        probe=candidate and command[1]=='-c' and 'audited source mismatch:' in command[2]
        if (getattr(self.bypass,'enabled',False) or self.isolated_witness() or not (worker or probe)):
            return ORIGINAL_POPEN(command,**kwargs)
        return _RequestProcess(self,command,kwargs)

    def run(self, command, *, cwd, env, timeout_seconds, cancel_event, script=False, script_input=None):
        from seedvision.benchmarking import reference_runner as runner
        worker = len(command)>2 and command[1:3]==['-m','seedvision.benchmarking.plantcv_worker']
        probe = len(command)>2 and command[1]=='-c' and 'audited source mismatch:' in command[2]
        if self.isolated_witness() or not (worker or probe or script):
            self.stats['isolated_calls'] += 1
            return self.original(command,cwd=cwd,env=env,timeout_seconds=timeout_seconds,cancel_event=cancel_event)
        with self.lock:
            if probe:
                key = tuple(command)
                if key not in self.probes:
                    self.stats['probe_starts'] += 1
                    self.bypass.enabled=True
                    try:
                        answer = self.original(command,cwd=cwd,env=env,timeout_seconds=timeout_seconds,cancel_event=cancel_event)
                    finally:
                        self.bypass.enabled=False
                    if answer[0] == 0: self.probes[key] = answer
                    return answer
                self.stats['probe_reuses'] += 1
                return self.probes[key]
            if runner._cancelled(cancel_event):
                raise runner.ReferenceCancelledError('PlantCV reference execution was cancelled')
            key = command[0]
            if key not in self.children:
                settings = dict(env)
                settings.update(MPLCONFIGDIR=self.temporary.name,TEMP=self.temporary.name,TMP=self.temporary.name)
                child = ORIGINAL_POPEN([command[0],'-m','tests.reference_session_worker'],
                    cwd=Path(__file__).resolve().parents[1],env=settings,
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                    text=True,encoding='utf-8',bufsize=1)
                queue = Queue()
                def read(stream=child.stdout, results=queue):
                    for line in stream: results.put(line)
                    results.put(None)
                Thread(target=read,daemon=True).start()
                self.children[key] = child,queue
                self.stats['worker_starts'] += 1
            child,queue = self.children[key]
            request = dict(arguments=command[3:],cwd=str(cwd))
            if script: request.update(kind='script',source=command[2],input=script_input or '')
            child.stdin.write(json.dumps(request)+'\n')
            child.stdin.flush()
            started = time.monotonic()
            while True:
                if runner._cancelled(cancel_event) or time.monotonic()-started >= timeout_seconds:
                    child.kill(); child.wait()
                    child.stdin.close(); child.stdout.close()
                    del self.children[key]
                    if runner._cancelled(cancel_event):
                        raise runner.ReferenceCancelledError('PlantCV reference execution was cancelled')
                    raise runner.ReferenceTimeoutError('Test reference session exceeded its request timeout')
                try: line = queue.get(timeout=.01)
                except Empty: continue
                if line is None: raise RuntimeError('Test reference session terminated without a response')
                response = json.loads(line)
                self.stats['actual_operations'] += 1
                return response['code'],response['stdout'].encode(),response['stderr'].encode()

    def subprocess_run(self, args, **kwargs):
        # Direct pinned numerical-oracle scripts also share dependency imports.
        # General application commands and all transport witnesses stay real.
        if (not isinstance(args,(list,tuple)) or len(args)<3 or args[1]!='-c'
                or 'plantcv-reference' not in str(args[0]).lower()
                or self.isolated_witness()
                or set(kwargs)-{'cwd','env','timeout','text','encoding','errors','capture_output','check','input'}):
            return self.original_run(args,**kwargs)
        import os
        payload = kwargs.get('input')
        if isinstance(payload,bytes):
            try: payload=payload.decode(kwargs.get('encoding') or 'utf-8')
            except UnicodeError: return self.original_run(args,**kwargs)
        code,out,err = self.run(list(map(str,args)),cwd=Path(kwargs.get('cwd') or Path.cwd()),
            env=kwargs.get('env') or os.environ.copy(),timeout_seconds=kwargs.get('timeout') or 120,
            cancel_event=None,script=True,script_input=payload)
        if kwargs.get('text') or kwargs.get('encoding'):
            encoding=kwargs.get('encoding') or 'utf-8'
            out,err=out.decode(encoding,kwargs.get('errors') or 'strict'),err.decode(encoding,kwargs.get('errors') or 'strict')
        result=subprocess.CompletedProcess(args,code,out,err)
        if kwargs.get('check'): result.check_returncode()
        return result


@contextmanager
def reference_session():
    from seedvision.benchmarking import reference_runner
    fixture = Session(reference_runner._run_child)
    try:
        # Keep every production callable and its code provenance intact. Only
        # the standard-library process boundary is substituted in tests.
        with patch.object(subprocess,'Popen',fixture.popen), patch.object(subprocess,'run',fixture.subprocess_run):
            yield fixture
    finally:
        fixture.close()
