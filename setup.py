# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

import os
import re
import sys
import sysconfig
import platform
import subprocess

# NOTE:
# `setup.py` is a build script. Some editors/type checkers (e.g. Pylance/Pyright)
# may analyze it with an interpreter that doesn't have build-time deps installed
# (setuptools/nanobind/cmake), producing false-positive import diagnostics.
#
# We prefer a setuptools-provided distutils shim to avoid stdlib `distutils`
# removal in newer Python versions.
#
from setuptools._distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        # Always rebuild (skip check removed per user request)
        # Check CMake
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: , ".join(e.name for e in self.extensions))

        # self.debug = True

        cmake_version = LooseVersion(
            re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        if cmake_version < '3.1.0':
            raise RuntimeError("CMake >= 3.1.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # IMPORTANT:
        # Build the C++ extension as a *package submodule* so it can coexist with
        # the pure-Python package `polyfempy/`.
        #
        # The extension name is `polyfempy.polyfempy`, and its output directory
        # must be the directory that contains the extension file returned by
        # `get_ext_fullpath(ext.name)`, i.e. `<build_lib>/polyfempy/`.
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Get number of threads for parallel compilation
        # Conservative default for 16GB RAM systems
        n_threads_str = os.environ.get("N_THREADS", "")
        if n_threads_str:
            n_threads = int(n_threads_str)
        else:
            # Auto-detect CPU cores, but cap at 6 threads for 16GB RAM systems
            # Large C++ projects can consume significant memory per thread
            cpu_count = os.cpu_count() or 4
            n_threads = min(cpu_count, 6)  # Conservative cap for 16GB RAM
            print(f"Auto-detected {cpu_count} CPU cores, using {n_threads} parallel compilation threads (conservative for 16GB RAM)")

        # Use sys.executable to ensure we get the correct Python path (conda environment)
        # This is more reliable than get_python_inc() which might use wrong Python
        python_executable = os.path.abspath(sys.executable)
        # Get include directory from the same Python installation
        python_include_directory = os.path.abspath(sysconfig.get_path('include'))
        
        # Verify we're not using env directory
        if 'env' in python_executable.lower() and 'envs' not in python_executable.lower():
            raise RuntimeError(
                f"ERROR: Detected wrong Python path: {python_executable}\n"
                f"This appears to be from the project's env directory, not conda environment.\n"
                f"Please ensure you're using: python -m pip install -e .\n"
                f"and that conda environment 'polyfem' is activated."
            )

        cmake_args = [
                      '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      # On Windows, the actual extension binary may be treated as a RUNTIME artifact
                      # (DLL/PYD), so also set the runtime output directory.
                      '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + python_executable,
                      '-DPYTHON_INCLUDE_DIR=' + python_include_directory,
                      '-DPOLYSOLVE_WITH_SPECTRA=OFF',
                      '-DPOLYSOLVE_WITH_AMGCL=OFF',
                      '-DPOLYSOLVE_WITH_UMFPACK=OFF',
                      '-DCMAKE_POLICY_VERSION_MINIMUM=3.5']
        
        # Use nanobind for Python bindings (pybind11 support removed)
        cmake_args.append('-DUSE_NANOBIND=ON')
        try:
            import nanobind
            import pathlib
            nb_dir = pathlib.Path(nanobind.__file__).resolve().parent / "cmake"
            cmake_args.append(f"-Dnanobind_DIR={nb_dir.as_posix()}")
            print(f"nanobind_DIR={nb_dir}")
        except Exception as e:
            print(f"Warning: could not resolve nanobind_DIR: {e}")
            
            # Ensure CMake can find conda-installed C++ packages (e.g., tsl-robin-map)
            # This is critical when using --no-build-isolation
            conda_prefix = os.environ.get('CONDA_PREFIX')
            if conda_prefix:
                cmake_prefix_path = os.environ.get('CMAKE_PREFIX_PATH', '')
                if cmake_prefix_path:
                    cmake_prefix_path = f"{conda_prefix};{cmake_prefix_path}"
                else:
                    cmake_prefix_path = conda_prefix
                cmake_args.append(f'-DCMAKE_PREFIX_PATH={cmake_prefix_path}')
                print(f"CMAKE_PREFIX_PATH={cmake_prefix_path}")
            
        print("Building with nanobind binding library")

        # Always ship the extension as Release. Debug CMake builds produce a different
        # binary profile (asserts, no inlining, .pdb beside .pyd) and are easy to mix
        # with Release VC++ runtimes — a common source of "worked until I tried a debug build".
        cfg = "Release"
        build_args = ["--config", cfg]
        cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]

        if platform.system() == "Windows":
            cmake_args += [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir),
                '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir),
            ]
            if os.environ.get('CMAKE_GENERATOR') != "NMake Makefiles":
                if sys.maxsize > 2**32:
                    cmake_args += ['-A', 'x64']
                # Enable parallel compilation on Windows (MSVC /m flag)
                # n_threads defaults to 4 if N_THREADS not specified
                build_args += ['--', f'/m:{n_threads}']
                print(f"Windows: Using {n_threads} parallel compilation threads (/m:{n_threads})")
        else:
            build_args += ['--', '-j{}'.format(n_threads)]

        env = os.environ.copy()
        # Enable CMake parallel build (for faster incremental compilation)
        env['CMAKE_BUILD_PARALLEL_LEVEL'] = str(n_threads)
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        # Configure CMake (preserve existing build cache for incremental compilation)
        try:
            subprocess.check_call(['cmake', ext.sourcedir] +
                                  cmake_args, cwd=self.build_temp, env=env)
        except subprocess.CalledProcessError as e:
            print("\n" + "="*80)
            print("CMake configuration failed!")
            print("="*80)
            raise
        
        # Build with verbose output
        try:
            # Add verbose flag to see detailed errors
            verbose_build_args = build_args.copy()
            if platform.system() != "Windows":
                verbose_build_args += ['--', 'VERBOSE=1']
            else:
                # Windows: use --verbose or capture output
                verbose_build_args = ['--verbose'] + build_args
            
            subprocess.check_call(['cmake', '--build', '.'] +
                                  verbose_build_args, cwd=self.build_temp, env=env)
        except subprocess.CalledProcessError as e:
            print("\n" + "="*80)
            print("CMake build failed! Trying to capture error output...")
            print("="*80)
            
            # Try to get more detailed error info
            try:
                result = subprocess.run(['cmake', '--build', '.'] + build_args,
                                       cwd=self.build_temp, env=env,
                                       capture_output=True, text=True)
                if result.stderr:
                    print("\nSTDERR:")
                    print(result.stderr)
                if result.stdout:
                    # Print last 100 lines of output
                    lines = result.stdout.split('\n')
                    print("\nLast 100 lines of build output:")
                    print("\n".join(lines[-100:]))
            except Exception:
                pass
            
            raise RuntimeError(
                f"CMake build failed with exit code {e.returncode}.\n"
                f"Check the output above for details, or run:\n"
                f"  cd {self.build_temp}\n"
                f"  cmake --build . --config {cfg} --verbose"
            )

        print()  # Add an empty line for cleaner output


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="polyfempy",
    version="0.8",
    description="Polyfem Python Bindings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://polyfem.github.io/",
    # Build as `polyfempy.polyfempy` (a submodule inside the Python package).
    ext_modules=[CMakeExtension('polyfempy.polyfempy')],
    cmdclass=dict(build_ext=CMakeBuild),
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.6',
    test_suite="test"
)
