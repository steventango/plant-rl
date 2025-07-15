import json
import logging
import lzma
import os
import pickle
import shutil
import time
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Protocol,
    Self,
    Sequence,
    Type,
    TypeVar,
)

from PyExpUtils.FileSystemContext import FileSystemContext
from PyExpUtils.models.ExperimentDescription import ExperimentDescription

from utils.RlGlue.agent import AsyncAgentWrapper

T = TypeVar("T")
Builder = Callable[[], T]

logger = logging.getLogger("plant_rl.checkpoint")
logger.setLevel(logging.DEBUG)


class Checkpoint:
    def __init__(
        self,
        exp: ExperimentDescription,
        idx: int,
        base_path: str = "./",
        load_path: str | None = None,
        save_every: float = -1,
    ) -> None:
        self._storage: Dict[str, Any] = {}
        self._exp = exp
        self._idx = idx

        self._last_save: Optional[float] = None
        self._save_every = save_every * 60

        self._load_path = load_path
        if load_path is None:
            self._ctx = self._exp.buildSaveContext(idx, base=base_path)
        else:
            self._ctx = FileSystemContext(load_path, base_path)

        self._params = exp.getPermutation(idx)
        self._base_path = f"{idx}"
        self._params_path = f"{idx}/params.json"
        self._data_path = f"{idx}/chk.pkl.xz"

    def __getitem__(self, name: str):
        return self._storage[name]

    def __setitem__(self, name: str, v: T) -> T:
        self._storage[name] = v
        return v

    def build(self, name: str, builder: Builder[T]) -> T:
        if name in self._storage:
            return self._storage[name]

        self._storage[name] = builder()
        return self._storage[name]

    def initial_value(self, name: str, val: T) -> T:
        if name in self._storage:
            return self._storage[name]

        self._storage[name] = val
        return val

    def save(self):
        params_path = self._ctx.resolve(self._params_path)

        logging.debug("Dumping checkpoint")
        if not os.path.exists(params_path):
            params_path = self._ctx.ensureExists(self._params_path, is_file=True)
            with open(params_path, "w") as f:
                json.dump(self._params, f)

        data_path = self._ctx.ensureExists(self._data_path, is_file=True)
        with lzma.open(data_path, "wb") as f:
            pickle.dump(self._storage, f)

        logging.debug("Finished dumping checkpoint")

    def maybe_save(self):
        if self._save_every < 0:
            return

        if self._last_save is None:
            self._last_save = time.time()

        if time.time() - self._last_save > self._save_every:
            self.save()
            self._last_save = time.time()

    def delete(self):
        base_path = self._ctx.resolve(self._base_path)
        if os.path.exists(base_path):
            shutil.rmtree(base_path)

    def load(self) -> bool:
        params_path = self._ctx.resolve(self._params_path)

        try:
            with open(params_path, "r") as f:
                params = json.load(f)

            assert self._load_path or params == self._params, (
                "The idx->params mapping has changed between checkpoints!!"
            )

        except Exception:
            logger.warning("Failed to load params from checkpoint", exc_info=True)
            return False

        path = self._ctx.resolve(self._data_path)
        try:
            with lzma.open(path, "rb") as f:
                self._storage = pickle.load(f)
        except Exception:
            try:
                with open(path.removesuffix(".xz"), "rb") as f:
                    self._storage = pickle.load(f)
            except Exception:
                logger.warning("Failed to load checkpoint data", exc_info=True)
                return False
        return True

    def load_if_exists(self):
        if not self._ctx.exists(self._data_path):
            return False
        logger.debug("Found a checkpoint! Loading...")
        return self.load()

    def load_from_checkpoint(
        self,
        source: Self | dict | object,
        config: dict | None,
        target=None,
        debug_key="",
    ):
        if target is None:
            target = self._storage

        if isinstance(source, Checkpoint):
            source = source._storage

        if config is None:
            if isinstance(source, Checkpoint):
                self._storage = deepcopy(source._storage)
            elif isinstance(source, dict):
                self._storage = deepcopy(source)
            return

        for key, load_or_subconfig in config.items():
            if isinstance(load_or_subconfig, dict):
                subconfig = load_or_subconfig
                self.load_from_checkpoint(
                    get(source, key),
                    subconfig,
                    get(target, key),
                    debug_key=f"{debug_key}.{key}",
                )
            else:
                load = load_or_subconfig
                if load:
                    if isinstance(target, AsyncAgentWrapper):
                        target = target.agent
                    if hasattr(target, "__getitem__"):
                        target[key] = deepcopy(get(source, key))  # type: ignore
                    else:
                        setattr(target, key, deepcopy(get(source, key)))


class Checkpointable(Protocol):
    def __setstate__(self, state) -> None: ...
    def __getstate__(self) -> Dict[str, Any]: ...


C = TypeVar("C", bound=Type[Checkpointable])


def checkpointable(props: Sequence[str]):
    def _inner(c: C) -> C:
        o_getter = c.__getstate__
        o_setter = c.__setstate__

        def setter(self, state):
            if o_setter is not None:
                o_setter(self, state)

            for p in props:
                setattr(self, p, state[p])

        def getter(self):
            out = {}
            for p in props:
                out[p] = getattr(self, p)

            out2 = {}
            if o_getter is not None:
                out2 = o_getter(self)
            elif c.__bases__[0].__getstate__:
                _getter = c.__bases__[0].__getstate__
                out2 = _getter(self)  # type: ignore

            out2 |= out
            return out2

        c.__getstate__ = getter
        c.__setstate__ = setter

        return c

    return _inner


def get(dict_or_obj: dict | object, key: str, default: T = None) -> T:
    if isinstance(dict_or_obj, dict):
        return dict_or_obj.get(key, default)
    return getattr(dict_or_obj, key, default)
