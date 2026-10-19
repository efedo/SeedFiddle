"""Test-only stdio worker: fresh PlantCV imports, shared third-party imports.

The application continues using its original one-shot worker. Every request
here executes that same worker's main and all its original verification.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback


def main():
    from seedvision.benchmarking import plantcv_worker
    base = Path.cwd()
    for line in sys.stdin:
        request = json.loads(line)
        stdout, stderr = io.StringIO(), io.StringIO()
        original_stdin = sys.stdin
        try:
            os.chdir(request['cwd'])
            tempfile.tempdir = request['cwd']
            # Preserve the production rule that PlantCV loads only after the
            # request's source verification, using its audited source loader.
            for name in tuple(sys.modules):
                if name == 'plantcv' or name.startswith('plantcv.'):
                    del sys.modules[name]
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                if request.get('kind') == 'script':
                    sys.stdin = io.StringIO(request.get('input',''))
                    sys.argv = ['-c',*request['arguments']]
                    exec(compile(request['source'],'<string>','exec'),{'__name__':'__main__'})
                    code = 0
                else:
                    code = plantcv_worker.main(request['arguments'])
        except SystemExit as error:
            code = error.code if isinstance(error.code,int) else 0 if error.code is None else 1
            if isinstance(error.code,str): stderr.write(error.code+'\n')
        except BaseException:
            code = 2
            traceback.print_exc(file=stderr)
        finally:
            sys.stdin = original_stdin
            pyplot = sys.modules.get('matplotlib.pyplot')
            if pyplot is not None:
                pyplot.close('all')
            os.chdir(base)
            tempfile.tempdir = os.environ.get('TEMP')
        print(json.dumps(dict(code=code,stdout=stdout.getvalue(),stderr=stderr.getvalue())),flush=True)


if __name__ == '__main__':
    main()
