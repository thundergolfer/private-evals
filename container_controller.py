## Copyright (C) 2024, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

import time
import tarfile
import io
import signal
import fcntl

import modal


def make_tar(files):
    file_like_object = io.BytesIO()
    tar = tarfile.TarFile(fileobj=file_like_object, mode="w")

    for file_name, file_content in files.items():
        tarinfo = tarfile.TarInfo(name=file_name)
        tarinfo.size = len(file_content)
        tarinfo.mtime = time.time()
        tar.addfile(tarinfo, io.BytesIO(file_content))

    tar.close()

    file_like_object.seek(0)

    return file_like_object


def setup_container(env):
    if env.image_ref:
        add_python = "3.11" if "python" not in env.image_ref else None
        image = modal.Image.from_registry(env.image_ref, add_python=add_python)
    else:
        image = modal.Image.debian_slim(python_version="3.11")
    app = modal.App.lookup("private-llm-benchmark-sandboxes", create_if_missing=True)
    container = modal.Sandbox.create(app=app, image=image)
    env.container = container
    env.is_setup = True


def stop_and_remove_container(container_id):
    sb = modal.Sandbox.from_id(container_id)
    sb.terminate()


def async_kill_container(container):
    stop_and_remove_container(container.object_id)


def safe_run(container, files, run_cmd):
    tarfile = make_tar(files)
    path = "/root/tarfile"
    with container.open(path, "wb") as sb_f:
        sb_f.write(tarfile.getvalue())

    # sb.interact() ?
    cmd = " ".join(run_cmd)
    print(f"Running command: {cmd}")
    process = container.exec(
        "bash", "-c", f"tar -xvf /root/tarfile -C /root/ && {cmd}", workdir="/root"
    )
    exit_code = process.wait()
    output = process.stdout.read()
    if exit_code != 0:
        stderr = process.stderr.read()
        raise RuntimeError(f"execution failed: {stderr}")
    return output.encode()


def is_fd_closed(fd):
    try:
        fcntl.fcntl(fd, fcntl.F_GETFD)
        return False
    except OSError:
        return True


def invoke_container(env, files, run_cmd):
    if not env.is_setup:
        setup_container(env)

    def raise_timeout(_signum, _frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(20)
    try:
        # function call that might take too long
        out = safe_run(env.container, files, run_cmd)
    except TimeoutError:
        out = b"Timeout: function took too long to complete"

    signal.alarm(0)
    return out.decode("utf-8")
