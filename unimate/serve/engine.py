"""The model, text encoder and normalization stats, loaded once, for
generating motion for registered rigs on request.

The per-request path is ``unimate.inference.sample``'s: encode the prompt,
``create_sample_condition`` on a :class:`SkeletonDataset` holding the rig,
sample the flow ODE with classifier-free guidance, then undo the
normalization. Unlike ``sample.py`` the text encoder stays resident next to
the model, so a request costs one sampling run.

Samplers (:data:`SAMPLERS`): ``quality`` is ``sample.py``'s adaptive dopri5
(about 4 model calls per step, ~10 s a take on an RTX 4070 Ti SUPER);
``fast``, the default, a fixed 20-step midpoint solve (76 calls, ~2.5 s a
take) whose joint positions differ from dopri5's by ~0.4% of the body size on
average from the same noise.
"""

import threading
import time

import torch

from unimate.configs.schema import MainConfig
from unimate.dataset.conditioning import create_sample_condition
from unimate.dataset.factory import create_skeleton_dataset
from unimate.inference.generate import ClassifierFreeSampleModel
from unimate.inference.sample import (
    _build_diffusion,
    _encode_prompt,
    _load_checkpoint,
    _make_text_encoder,
    _resolve_exp_paths,
    _resolve_stats_path,
)
from unimate.models.factory import create_model
from unimate.serve.rig_config import STATS_DATASET
from unimate.utils.logger import get_logger

logger = get_logger(__name__)

# name -> (torchdiffeq method, steps: grid points, or saved points for dopri5)
SAMPLERS = {'fast': ('midpoint', 20), 'quality': ('dopri5', 50)}


class Engine:
    """One loaded run (*exp_dir*: ``config.json``, ``dataset_stats.npy``,
    ``checkpoints/``). Thread-safe: requests are serialized on :attr:`lock`."""

    def __init__(self, exp_dir, device=None, model_path=None):
        config_path, model_path = _resolve_exp_paths(exp_dir, model_path)
        self.config = MainConfig.from_json(config_path)
        self.stats_path = _resolve_stats_path(exp_dir)
        if self.stats_path is None:
            raise FileNotFoundError(f"No dataset_stats.npy in {exp_dir!r}")
        if device is None:
            device = self.config.sampling.device if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model_path = model_path
        self.lock = threading.Lock()
        if self.config.training.diff_model != 'flow':
            raise ValueError("the server samples flow-matching runs only")
        # TF32 matmuls: ~15% faster sampling, no measurable change in the motion
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        t0 = time.perf_counter()
        self.encoder = _make_text_encoder(self.config, self.device)
        self.model = create_model(dataset_config=self.config.dataset,
                                  model_config=self.config.model)
        self.diffusion, self.gen_diffusion = _build_diffusion(self.config)
        _load_checkpoint(self.model, model_path, self.config)
        self.model.to(self.device).eval()
        logger.info(f"Loaded {model_path} on {self.device} in {time.perf_counter() - t0:.1f}s")

        self._conds = {}
        self._dataset = None
        self._prompts = {}

    # -- rigs ----------------------------------------------------------------

    def set_rig(self, name, cond_entry):
        """Register (or replace) rig *name* with its cond entry."""
        with self.lock:
            replacing = name in self._conds
            self._conds[name] = cond_entry
            if replacing:
                # a SkeletonDataset refuses to re-register a name: start over
                self._dataset = None
            elif self._dataset is not None:
                self._register(self._dataset, name, cond_entry)

    def rigs(self):
        return sorted(self._conds)

    def _register(self, dataset, name, cond_entry):
        md = dataset.motion_dataset
        md.text_encoder = self.encoder          # encodes the rig's joint labels
        try:
            md.add_cond_object(name, cond_entry, STATS_DATASET)
        finally:
            md.text_encoder = None

    def _skeleton_dataset(self):
        if self._dataset is None:
            dataset = create_skeleton_dataset(self.config.dataset, self.config.model,
                                              self.stats_path)
            for name, cond_entry in self._conds.items():
                self._register(dataset, name, cond_entry)
            self._dataset = dataset
        return self._dataset

    # -- sampling ------------------------------------------------------------

    def _encode(self, prompt):
        enc = self._prompts.get(prompt)
        if enc is None:
            enc = self._prompts[prompt] = _encode_prompt(self.encoder, prompt)
        return enc

    def generate(self, name, prompt, takes=1, seed=None, cfg_scale=None, sampler='fast'):
        """``(motions, seconds)``: *takes* denormalized feature arrays
        ``(T, J, 12)`` for rig *name* (cond joint order), sampled in one batch
        with *sampler* (:data:`SAMPLERS`)."""
        prompt = (prompt or '').strip()
        if not prompt:
            raise ValueError("empty prompt")
        cfg_scale = float(cfg_scale if cfg_scale is not None else self.config.sampling.cfg_scale)
        if cfg_scale <= 1.0:
            raise ValueError("cfg_scale must be above 1 for a prompt to have an effect")
        if sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {sorted(SAMPLERS)}, got {sampler!r}")
        method, steps = SAMPLERS[sampler]
        takes = max(1, min(int(takes), 16))
        cfg = self.config
        with self.lock, torch.no_grad():
            if name not in self._conds:
                raise KeyError(f"unknown rig {name!r}")
            dataset = self._skeleton_dataset()
            enc = self._encode(prompt)
            _, cond = create_sample_condition(
                config=cfg, data=dataset,
                test_case_captions=[(name, prompt, enc)] * takes)
            cond = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in cond.items()}
            shape = (cond['n_joints'].shape[0], cfg.dataset.max_joints,
                     cfg.dataset.feature_len, cfg.dataset.max_motion_length)
            if seed is not None:
                torch.manual_seed(int(seed))
            t0 = time.perf_counter()
            # generate_samples' text path, with the sampler chosen here
            noise = torch.randn(shape, device=self.device)
            solve = self.gen_diffusion.sample_ode(sampling_method=method, num_steps=steps)
            guided = ClassifierFreeSampleModel(self.model, cfg_scale=cfg_scale)
            samples = solve(noise, guided, cond=cond)[-1]
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            seconds = time.perf_counter() - t0
            motions = []
            for b in range(shape[0]):
                nj = int(cond['n_joints'][b].item())
                motion = samples[b][:nj].permute(2, 0, 1).cpu().numpy()        # (T, J, D)
                mean = cond['mean'][b][:nj][None].cpu().numpy()
                std = cond['std'][b][:nj][None].cpu().numpy()
                motions.append(motion * std + mean)
        logger.info(f"{name}: {takes} take(s) of {prompt!r} in {seconds:.2f}s ({sampler})")
        return motions, seconds
