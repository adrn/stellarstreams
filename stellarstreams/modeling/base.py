# Built in
from abc import ABCMeta

# Third-party
import numpy as np

__all__ = ['BaseModel']


class BaseModel(metaclass=ABCMeta):

    def __new__(cls, *args, **kwargs):
        if cls is BaseModel:
            raise TypeError("Can't instantiate abstract class {name} directly. "
                            "Use a subclass instead.".format(name=cls.__name__))
        return object.__new__(cls)

    def __init__(self):
        self._registry = dict()

    def _default_pack_pars(self, name, p, fill_frozen):
        vals = []

        if self.frozen.get(name, False) is not True:
            this_frozen = self.frozen.get(name, dict())  # frozen values
            this_p = p.get(name, dict())  # values passed in via `p`
            for par_name in self.param_names[name]:
                if par_name in this_frozen:
                    val = this_frozen.get(par_name, None)
                    if not fill_frozen:
                        continue
                else:
                    val = this_p.get(par_name, None)

                if val is None:
                    raise ValueError("[{0}] No value passed in for parameter "
                                     "{1}, but it isn't frozen either!"
                                     .format(name, par_name))
                vals.append(val)

        return np.array(vals)

    def _default_unpack_pars(self, name, x, fill_frozen):
        n_free = 0
        pars = {}

        if self.frozen.get(name, False) is not True:
            this_frozen = self.frozen.get(name, dict())
            for par_name in self.param_names[name]:
                if par_name in this_frozen:
                    if fill_frozen:
                        pars[name] = this_frozen[par_name]
                else:
                    pars[name] = x[n_free]
                    n_free += 1

        return pars, n_free

    def pack_pars(self, p, fill_frozen=False):
        """
        p is a dictionary of parameter values
        """
        vals = np.array([])
        for name in self._registry:
            v = self._registry[name]['pack'](name, p.get(name, None),
                                             fill_frozen)
            vals = np.concatenate((vals, v))
        return vals

    # TODO: unpack pars

    def _default_transform(self, *args, **kwargs):
        pass

    def _default_inv_transform(self, *args, **kwargs):
        pass

    def _default_ln_prior(self, *args, **kwargs):
        return 0.

    def register_param_group(self, name, param_names,
                             pack_func=None, unpack_func=None,
                             transform_func=None, inv_transform_func=None,
                             ln_prior=None):

        if name in self._registry:
            raise ValueError("Parameter group {0} already exists in registry!"
                             .format(name))

        group = {}

        group['params'] = param_names

        # TODO: validate the function arguments?
        if pack_func is None:
            pack_func = self._default_pack_pars
        group['pack'] = pack_func

        if unpack_func is None:
            unpack_func = self._default_pack_pars
        group['unpack'] = unpack_func

        if transform_func is None:
            transform_func = self._default_transform
        group['transform'] = transform_func

        if inv_transform_func is None:
            inv_transform_func = self._default_inv_transform
        group['inv_transform'] = inv_transform_func

        if ln_prior is None:
            ln_prior = self._default_ln_prior
        group['ln_prior'] = ln_prior

        self._registry[name] = group

    def ln_prior(self, pars):
        lp = 0.
        for k in self._registry:
            lp += self._registry[k]['ln_prior'](pars)
        return lp

    def ln_posterior(self, p):
        pars = self.unpack_pars(p)

        lp = self.ln_prior(pars)
        if not np.all(np.isfinite(lp)):
            return -np.inf

        ll = self.ln_likelihood(pars)
        if not np.all(np.isfinite(ll)):
            return -np.inf

        return np.sum(ll) + lp

    def __call__(self, p):
        return self.ln_posterior(p)
