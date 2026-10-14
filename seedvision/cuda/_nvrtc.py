"""Compile fixed application CUDA kernels using PyTorch's bundled runtime.

No toolkit, external compiler, subprocess or disk cache is required. Kernels
operate on PyTorch-owned device storage and the current PyTorch CUDA stream.
Callers must retain tensor arguments until the stream has completed their use.
"""
from ctypes import (CDLL, POINTER, byref, c_char_p, c_int, c_size_t, c_uint,
                    c_uint64, c_void_p, cast, create_string_buffer)
from functools import lru_cache
import ctypes
import os
from pathlib import Path

import torch


class CudaKernelUnavailable(RuntimeError):
    pass


class CudaKernelError(RuntimeError):
    pass


def _load_runtime():
    folder = Path(torch.__file__).resolve().parent/'lib'
    if os.name == 'nt':
        candidates = sorted(p for p in folder.glob('nvrtc64_*.dll') if '.alt.' not in p.name)
        loader, driver_name = ctypes.WinDLL, 'nvcuda.dll'
    else:
        candidates = sorted(folder.glob('libnvrtc.so*'))
        candidates += sorted((folder.parent.parent/'nvidia'/'cuda_nvrtc'/'lib').glob('libnvrtc.so*'))
        loader, driver_name = CDLL, 'libcuda.so.1'
    if not candidates:
        raise CudaKernelUnavailable('The active PyTorch installation has no bundled NVRTC runtime.')
    try:
        runtime, driver = loader(str(candidates[0])), loader(driver_name)
    except OSError as error:
        raise CudaKernelUnavailable('The bundled CUDA compiler or active CUDA driver cannot be loaded.') from error
    signatures = {
        'nvrtcVersion': [POINTER(c_int), POINTER(c_int)],
        'nvrtcCreateProgram': [POINTER(c_void_p), c_char_p, c_char_p, c_int, POINTER(c_char_p), POINTER(c_char_p)],
        'nvrtcCompileProgram': [c_void_p, c_int, POINTER(c_char_p)],
        'nvrtcGetProgramLogSize': [c_void_p, POINTER(c_size_t)],
        'nvrtcGetProgramLog': [c_void_p, c_void_p],
        'nvrtcGetPTXSize': [c_void_p, POINTER(c_size_t)],
        'nvrtcGetPTX': [c_void_p, c_void_p],
        'nvrtcDestroyProgram': [POINTER(c_void_p)],
    }
    for name, types in signatures.items():
        getattr(runtime, name).argtypes = types
        getattr(runtime, name).restype = c_int
    signatures = {
        'cuDriverGetVersion': [POINTER(c_int)],
        'cuCtxGetCurrent': [POINTER(c_void_p)],
        'cuModuleLoadData': [POINTER(c_void_p), c_void_p],
        'cuModuleGetFunction': [POINTER(c_void_p), c_void_p, c_char_p],
        'cuModuleUnload': [c_void_p],
        'cuLaunchKernel': [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
                           c_uint, c_void_p, POINTER(c_void_p), POINTER(c_void_p)],
        'cuGetErrorName': [c_int, POINTER(c_char_p)],
    }
    for name, types in signatures.items():
        getattr(driver, name).argtypes = types
        getattr(driver, name).restype = c_int
    return runtime, driver


@lru_cache(maxsize=1)
def _runtime():
    return _load_runtime()


def _driver_check(driver, code, operation):
    if code:
        name = c_char_p()
        driver.cuGetErrorName(code, byref(name))
        detail = name.value.decode('ascii', errors='replace') if name.value else str(code)
        raise CudaKernelError(f'{operation} failed: {detail}.')


def runtime_versions():
    """Record the actual loaded compiler and driver API used by fixed kernels."""
    runtime, driver = _runtime()
    major, minor, driver_version = c_int(), c_int(), c_int()
    if runtime.nvrtcVersion(byref(major), byref(minor)):
        raise CudaKernelUnavailable('The loaded CUDA compiler version is unavailable.')
    _driver_check(driver, driver.cuDriverGetVersion(byref(driver_version)), 'CUDA driver version query')
    return {'nvrtc': f'{major.value}.{minor.value}', 'cuda_driver_api': str(driver_version.value)}


class _Program:
    def __init__(self, source, device):
        runtime, self.driver = _runtime()
        self.module = c_void_p()
        self.functions = {}
        program = c_void_p()
        code = runtime.nvrtcCreateProgram(byref(program), source.encode('utf-8'),
            b'seedvision_fixed_kernel.cu', 0, None, None)
        if code:
            raise CudaKernelUnavailable(f'Bundled CUDA program creation failed ({code}).')
        try:
            major, minor = torch.cuda.get_device_capability(device)
            options = (c_char_p*3)(f'--gpu-architecture=compute_{major}{minor}'.encode('ascii'),
                b'--std=c++14', b'--fmad=false')
            code = runtime.nvrtcCompileProgram(program, len(options), options)
            if code:
                size = c_size_t()
                runtime.nvrtcGetProgramLogSize(program, byref(size))
                if size.value > 1024*1024:
                    raise CudaKernelError('CUDA compiler diagnostic exceeds its metadata limit.')
                log = create_string_buffer(max(1, size.value))
                runtime.nvrtcGetProgramLog(program, log)
                raise CudaKernelError(log.value.decode('utf-8', errors='replace'))
            size = c_size_t()
            if runtime.nvrtcGetPTXSize(program, byref(size)) or not 0 < size.value <= 8*1024*1024:
                raise CudaKernelError('CUDA compiled program exceeds its bounded size or is unavailable.')
            ptx = create_string_buffer(size.value)
            if runtime.nvrtcGetPTX(program, ptx):
                raise CudaKernelError('CUDA compiled program could not be retrieved.')
            _driver_check(self.driver, self.driver.cuModuleLoadData(byref(self.module), ptx), 'CUDA module loading')
        finally:
            runtime.nvrtcDestroyProgram(byref(program))

    def launch(self, name, tensors, integers):
        if name not in self.functions:
            function = c_void_p()
            _driver_check(self.driver, self.driver.cuModuleGetFunction(byref(function), self.module,
                name.encode('ascii')), 'CUDA kernel selection')
            self.functions[name] = function
        values = [c_uint64(tensor.data_ptr()) for tensor in tensors]
        values += [c_int(value) for value in integers]
        arguments = (c_void_p*len(values))(*(cast(byref(value), c_void_p) for value in values))
        stream = c_void_p(torch.cuda.current_stream().cuda_stream)
        _driver_check(self.driver, self.driver.cuLaunchKernel(self.functions[name], 1, 1, 1,
            1, 1, 1, 0, stream, arguments, None), 'CUDA kernel launch')

    def launch_1d(self, name, tensors, integers, work_items):
        """Launch one independent kernel thread per bounded work item."""
        if type(work_items) is not int or not 1 <= work_items <= 2**31-1:
            raise ValueError('CUDA work item count must be a positive signed32 integer.')
        if any(type(value) is not int or not -(2**31) <= value <= 2**31-1 for value in integers):
            raise ValueError('CUDA scalar kernel arguments must be signed32 integers.')
        if name not in self.functions:
            function = c_void_p()
            _driver_check(self.driver, self.driver.cuModuleGetFunction(byref(function), self.module,
                name.encode('ascii')), 'CUDA kernel selection')
            self.functions[name] = function
        values = [c_uint64(tensor.data_ptr()) for tensor in tensors]
        values += [c_int(value) for value in integers]
        arguments = (c_void_p*len(values))(*(cast(byref(value), c_void_p) for value in values))
        stream = c_void_p(torch.cuda.current_stream().cuda_stream)
        _driver_check(self.driver, self.driver.cuLaunchKernel(self.functions[name],
            (work_items+127)//128, 1, 1, 128, 1, 1, 0, stream, arguments, None), 'CUDA kernel launch')

    def __del__(self):
        if getattr(self, 'module', None) and self.module.value:
            try:
                self.driver.cuModuleUnload(self.module)
            except Exception:
                pass  # Interpreter teardown can follow driver teardown.


@lru_cache(maxsize=8)
def _program(source, device_index, context):
    return _Program(source, device_index)


def fixed_kernel_program(source, device):
    """Return a bounded in-memory program for the caller's current device context."""
    if not isinstance(source, str) or not 1 <= len(source.encode('utf-8')) <= 128*1024:
        raise ValueError('Fixed CUDA source exceeds its byte limit.')
    device = torch.device(device)
    with torch.cuda.device(device):
        # An actual allocation activates PyTorch's primary device context.
        activation = torch.empty(1, device=device)
        _, driver = _runtime()
        context = c_void_p()
        _driver_check(driver, driver.cuCtxGetCurrent(byref(context)), 'CUDA context query')
        if not context.value:
            raise CudaKernelUnavailable('PyTorch has no active CUDA device context.')
        index = torch.cuda.current_device()
        return _program(source, index, context.value)
