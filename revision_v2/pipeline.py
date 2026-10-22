"""Durably queue audited experiments and result sections; never edit the paper.

If a stage fails, its exit code is recorded and later stages do not execute.
Existing completed fits are resumable; another active pipeline is refused.
"""
from pathlib import Path
import argparse, ctypes, datetime, json, os, subprocess, sys, time


def active(pid):
    if not pid:
        return False
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_uint32()
        try:
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--cuda-runtime', type=Path)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--wait-main-pid', type=int)
    args = parser.parse_args()
    if args.cache_root:
        os.environ['GTS_CACHE_ROOT'] = str(args.cache_root.resolve())
    if args.cuda_runtime:
        os.environ['PYTHONPATH'] = str(args.cuda_runtime.resolve()) + os.pathsep + os.environ.get('PYTHONPATH', '')
    workspace = args.workspace.resolve()
    code = Path(__file__).resolve().parent
    root = workspace / 'experiments/revision_v2'
    prepared, runs, results = root / 'prepared', root / 'runs', root / 'results'
    multihorizon = root / 'multihorizon'
    root.mkdir(parents=True, exist_ok=True)
    lock = root / 'pipeline.lock.json'
    if lock.exists():
        old = json.loads(lock.read_text())
        if active(old['pid']):
            raise RuntimeError(f"Another revision pipeline is running: PID {old['pid']}")
    lock.write_text(json.dumps({'pid': os.getpid(), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}))
    state = root / 'pipeline_status.json'
    def status(stage, outcome, **details):
        state.write_text(json.dumps({'pipeline_pid': os.getpid(), 'stage': stage, 'state': outcome,
            'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'manuscript_modified': False, **details}, indent=2), encoding='utf-8')
        print(stage, outcome, flush=True)
    def execute(stage, command):
        status(stage, 'running')
        with (root / f'{stage}.log').open('a', encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', *map(str, command)],
                cwd=workspace, stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
            status(stage, 'running', child_pid=child.pid)
            exit_code = child.wait()
        if exit_code:
            status(stage, 'failed', exit_code=exit_code, log=str(root / f'{stage}.log'))
            raise RuntimeError(f'{stage} failed; later stages were not run.')
        status(stage, 'complete')
    try:
        status('waiting_for_existing_main_run', 'running', child_pid=args.wait_main_pid)
        while active(args.wait_main_pid):
            time.sleep(5)
        # Preserve the manifest of the already-running first main invocation.
        old_manifest = runs / 'run_manifest.json'
        if old_manifest.exists() and not (runs / 'run_manifest_initial.json').exists():
            (runs / 'run_manifest_initial.json').write_bytes(old_manifest.read_bytes())
        runtime = ['--cuda-runtime', args.cuda_runtime] if args.cuda_runtime else []
        common = ['--prepared', prepared, '--output', runs, *runtime]
        if not (runs / 'completed_main.json').exists():
            execute('main_resume', [code / 'run.py', *common, '--stage', 'main'])
        execute('main_statistics', [code / 'aggregate.py', '--prepared', prepared, '--runs', runs, '--output', results])
        execute('ablations', [code / 'run.py', *common, '--stage', 'ablations'])
        execute('multihorizon', [code / 'multihorizon.py', '--prepared', prepared,
            '--runs', runs, '--output', multihorizon, *runtime])
        execute('diagnostic_statistics', [code / 'aggregate_diagnostics.py', '--prepared', prepared,
            '--runs', runs, '--multihorizon', multihorizon, '--output', results])
        execute('saved_model_benchmarks', [code / 'benchmark.py', '--prepared', prepared,
            '--runs', runs, '--output', results, *runtime])
        execute('result_figures', [code / 'plots.py', '--results', results])
        execute('final_result_checks', [code / 'verify_results.py', '--prepared', prepared,
            '--runs', runs, '--results', results])
        status('results_ready_for_manuscript_revision', 'complete', results_directory=str(results),
               note='All numerical sections are complete. Manuscript must now be revised and visually checked against these actual results.')
    except BaseException as error:
        current = json.loads(state.read_text()) if state.exists() else {}
        if current.get('state') != 'failed':
            status(current.get('stage', 'pipeline'), 'failed', error=str(error))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
