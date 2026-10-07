"""Local HTTP server between a DCC client (the Maya tool in ``maya/``) and
UniMate: rig set-up and text-to-motion clips.

    python -m unimate.serve --exp_dir outputs/unimate_uniml3d_f60_v3

JSON over HTTP on 127.0.0.1 (no auth: keep it local). The server answers at
once; the model loads in the background (``GET /health`` says when it is
ready). Rig set-up needs no model.

``GET  /health``            ``{ready, error, device, model, rigs}``
``GET  /rigs``              built rigs
``GET  /rigs/<name>``       a rig's record (``rig.json``: proxy skeleton, bind pose)
``POST /rigs/analyze``      ``{skeleton}`` -> suggested labels / facing / joints
``POST /rigs/build``        ``{skeleton, include?, labels?, face_pair?}`` -> rig record
``POST /generate``          ``{rig, prompt, takes?, seed?, cfg_scale?, sampler?, frames?,
                            debug?}`` -> ``{clips, seconds}``; *sampler* ``fast``
                            (default) or ``quality`` (``unimate.serve.engine.SAMPLERS``)
``POST /clips/from_features`` ``{rig, path, frames?, debug?}`` -> a clip from a saved
                            ``.npy`` of model features (testing the retarget)
"""

import argparse
import json
import os
import random
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from unimate.serve import rig_config
from unimate.serve.retarget import clip_from_features
from unimate.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PORT = 8792


class State:
    """What the handlers share: the rigs on disk and the (loading) engine."""

    def __init__(self, exp_dir, rigs_dir, device=None, llm_model=None):
        self.exp_dir = exp_dir
        self.rigs_dir = rigs_dir
        self.device = device
        self.engine = None
        self.error = None
        self.llm_client = None
        if llm_model:
            from data_process.joint_annotation.llm import LLMClient
            self.llm_client = LLMClient(llm_model)
        self._rigs = {}                 # name -> (record, cond)
        self._lock = threading.Lock()
        for name in rig_config.list_rigs(rigs_dir):
            try:
                self._rigs[name] = rig_config.load_rig(rigs_dir, name)
            except Exception as exc:    # a broken rig dir must not stop the server
                logger.warning(f"Skipping rig {name!r}: {exc}")

    def load_engine(self):
        try:
            from unimate.serve.engine import Engine
            engine = Engine(self.exp_dir, device=self.device)
            with self._lock:
                for name, (_, cond) in self._rigs.items():
                    engine.set_rig(name, cond)
                self.engine = engine
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logger.error(f"Model failed to load: {self.error}\n{traceback.format_exc()}")

    def rig(self, name):
        with self._lock:
            if name not in self._rigs:
                raise KeyError(f"unknown rig {name!r}; build it first")
            return self._rigs[name]

    def add_rig(self, record):
        name = record['name']
        loaded = rig_config.load_rig(self.rigs_dir, name)
        with self._lock:
            self._rigs[name] = loaded
            if self.engine is not None:
                self.engine.set_rig(name, loaded[1])

    def health(self):
        engine = self.engine
        return {'ready': engine is not None, 'error': self.error,
                'device': str(engine.device) if engine else None,
                'model': engine.model_path if engine else None,
                'rigs': sorted(self._rigs)}


def _generate(state, body):
    engine = state.engine
    if engine is None:
        raise RuntimeError(state.error or "the model is still loading")
    name = body['rig']
    record, cond = state.rig(name)
    seed = body.get('seed')
    if seed is None:
        seed = random.randrange(2 ** 31)
    takes = int(body.get('takes', 1))
    sampler = body.get('sampler') or 'fast'
    motions, seconds = engine.generate(name, body.get('prompt', ''), takes=takes, seed=seed,
                                       cfg_scale=body.get('cfg_scale'), sampler=sampler)
    cfg_scale = body.get('cfg_scale') or engine.config.sampling.cfg_scale
    clips = [clip_from_features(m, record, cond, frames=body.get('frames'),
                                debug=bool(body.get('debug')),
                                meta={'prompt': body.get('prompt', '').strip(), 'seed': seed,
                                      'take': i, 'cfg_scale': cfg_scale, 'sampler': sampler,
                                      'rig': name,
                                      'generator': 'unimate',
                                      'model': os.path.basename(engine.model_path)})
             for i, m in enumerate(motions)]
    return {'clips': clips, 'seconds': seconds}


def _from_features(state, body):
    record, cond = state.rig(body['rig'])
    feats = np.load(body['path'])
    if feats.shape[1] != len(cond['parents']):
        raise ValueError(f"{body['path']} has {feats.shape[1]} joints; rig {body['rig']!r} "
                         f"has {len(cond['parents'])}")
    return clip_from_features(feats, record, cond, frames=body.get('frames'),
                              debug=bool(body.get('debug')),
                              meta={'source': os.path.basename(body['path'])})


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'UniMateServe/1'

        def log_message(self, fmt, *args):
            logger.debug(fmt % args)

        def _send(self, status, payload):
            data = json.dumps(payload, default=rig_config._json_default).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _route(self, method):
            path = self.path.split('?', 1)[0].rstrip('/')
            body = {}
            if method == 'POST':
                length = int(self.headers.get('Content-Length') or 0)
                body = json.loads(self.rfile.read(length) or b'{}')
            if method == 'GET' and path == '/health':
                return state.health()
            if method == 'GET' and path == '/rigs':
                return {'rigs': state.health()['rigs']}
            if method == 'GET' and path.startswith('/rigs/'):
                return state.rig(path[len('/rigs/'):])[0]
            if method == 'POST' and path == '/rigs/analyze':
                return rig_config.analyze(body['skeleton'], state.llm_client)
            if method == 'POST' and path == '/rigs/build':
                record = rig_config.build(body['skeleton'], state.rigs_dir,
                                          include=body.get('include'),
                                          labels=body.get('labels'),
                                          face_pair=body.get('face_pair'),
                                          llm_client=state.llm_client)
                state.add_rig(record)
                return record
            if method == 'POST' and path == '/generate':
                return _generate(state, body)
            if method == 'POST' and path == '/clips/from_features':
                return _from_features(state, body)
            raise LookupError(f"no route {method} {path}")

        def _handle(self, method):
            try:
                self._send(200, self._route(method))
            except LookupError as exc:          # KeyError is a LookupError too
                self._send(404, {'error': str(exc).strip("'\"")})
            except RuntimeError as exc:
                self._send(503, {'error': str(exc)})
            except (ValueError, TypeError) as exc:
                self._send(400, {'error': str(exc)})
            except Exception as exc:
                logger.error(traceback.format_exc())
                self._send(500, {'error': f"{type(exc).__name__}: {exc}"})

        def do_GET(self):
            self._handle('GET')

        def do_POST(self):
            self._handle('POST')

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--exp_dir', default='outputs/unimate_uniml3d_f60_v3',
                    help="Trained run: config.json, dataset_stats.npy, checkpoints/.")
    ap.add_argument('--rigs_dir', default='outputs/maya_rigs',
                    help="Where built rigs are written (one asset directory each).")
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=DEFAULT_PORT)
    ap.add_argument('--device', default=None, help="Torch device (default: cuda if available).")
    ap.add_argument('--llm_model', default=None,
                    help="Label joints with this LLM (e.g. deepseek-v4-flash with "
                         "DEEPSEEK_API_KEY, gpt-* with OPENAI_API_KEY) instead of the rules.")
    args = ap.parse_args(argv)

    os.makedirs(args.rigs_dir, exist_ok=True)
    state = State(args.exp_dir, args.rigs_dir, device=args.device, llm_model=args.llm_model)
    threading.Thread(target=state.load_engine, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    logger.info(f"UniMate server on http://{args.host}:{args.port} "
                f"({len(state.health()['rigs'])} rig(s) in {args.rigs_dir}); loading the model…")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
