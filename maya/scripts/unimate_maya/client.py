"""HTTP client for the UniMate server (``python -m unimate.serve``), standard
library only, plus starting that server from Maya."""

import json
import os
import subprocess
import urllib.error
import urllib.request

DEFAULT_URL = 'http://127.0.0.1:8792'
EXP_DIR = 'outputs/unimate_uniml3d_f60_v3'
RIGS_DIR = 'outputs/maya_rigs'

# Maya's own Python settings, which would break the env's interpreter
_PY_ENV = ('PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP', 'PYTHONUSERBASE', 'PYTHONNOUSERSITE')


class ServerError(RuntimeError):
    """The server answered with an error, or could not be reached."""


def repo_root():
    """The UniMate checkout: ``$UNIMATE_ROOT``, else the one this module is in
    (``<root>/maya/scripts/unimate_maya/client.py``)."""
    env = os.environ.get('UNIMATE_ROOT')
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, '..', '..', '..'))


class Client:
    def __init__(self, url=DEFAULT_URL, timeout=600):
        self.url = url.rstrip('/')
        self.timeout = timeout

    def _call(self, path, body=None, timeout=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            try:
                message = json.loads(exc.read()).get('error', str(exc))
            except ValueError:
                message = str(exc)
            raise ServerError(message) from None
        except (urllib.error.URLError, OSError) as exc:
            raise ServerError(f"no UniMate server at {self.url} ({exc})") from None

    def health(self, timeout=2):
        return self._call('/health', timeout=timeout)

    def is_up(self):
        try:
            self.health()
            return True
        except ServerError:
            return False

    def analyze(self, skeleton):
        return self._call('/rigs/analyze', {'skeleton': skeleton})

    def build(self, skeleton, include=None, labels=None, face_pair=None):
        return self._call('/rigs/build', {'skeleton': skeleton, 'include': include,
                                          'labels': labels, 'face_pair': face_pair})

    def rig(self, name):
        return self._call(f'/rigs/{name}')

    def generate(self, rig, prompt, takes=1, seed=None, cfg_scale=None, sampler='fast',
                 frames=None):
        body = {'rig': rig, 'prompt': prompt, 'takes': takes, 'sampler': sampler}
        for key, value in (('seed', seed), ('cfg_scale', cfg_scale), ('frames', frames)):
            if value is not None:
                body[key] = value
        return self._call('/generate', body)


def start_server(root=None, port=8792, exp_dir=EXP_DIR, rigs_dir=RIGS_DIR):
    """Start the server from the checkout's ``.venv`` in the background; its
    log goes to ``outputs/serve.log``. Returns the process."""
    root = root or repo_root()
    python = os.path.join(root, '.venv', 'Scripts', 'python.exe')
    if not os.path.isfile(python):
        python = os.path.join(root, '.venv', 'bin', 'python')
    if not os.path.isfile(python):
        raise ServerError(f"no Python env at {root}\\.venv (run scripts/setup_windows.sh)")
    env = {k: v for k, v in os.environ.items() if k not in _PY_ENV}
    env['PYTHONUTF8'] = '1'
    os.makedirs(os.path.join(root, 'outputs'), exist_ok=True)
    log = open(os.path.join(root, 'outputs', 'serve.log'), 'ab')
    flags = 0
    if os.name == 'nt':
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(
        [python, '-m', 'unimate.serve', '--exp_dir', exp_dir, '--rigs_dir', rigs_dir,
         '--port', str(port)],
        cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        creationflags=flags)
