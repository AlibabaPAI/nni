import logging
from typing import Any, Dict

from .literal import QuantConfigLiteral, QuantDtype, QuantType, QuantScheme
from .utils import calculate_qmin_qmax, get_bits_length

logger = logging.getLogger(__name__)

# default settings for quantization module
quant_default_settings = {
    QuantType.weight: {
        QuantConfigLiteral.quant_scheme: QuantScheme.per_tensor_affine,
        QuantConfigLiteral.quant_dtype: QuantDtype.uint,
    },
    QuantType.input: {
        QuantConfigLiteral.quant_scheme: QuantScheme.per_tensor_affine,
        QuantConfigLiteral.quant_dtype: QuantDtype.uint
    },
    QuantType.output: {
        QuantConfigLiteral.quant_scheme: QuantScheme.per_tensor_affine,
        QuantConfigLiteral.quant_dtype: QuantDtype.uint
    }
}


def change_default_quant_settings(quant_type, new_scheme=None, new_dtype=None):
    # todo: remove this if we convert string config to enum type.
    if isinstance(quant_type, str):
        assert quant_type in QuantType.valid_values()
    if isinstance(new_scheme, str):
        assert new_scheme in QuantScheme.valid_values()
    if isinstance(new_dtype, str):
        assert new_dtype in QuantDtype.valid_values()

    global quant_default_settings
    import copy
    dd = copy.deepcopy(quant_default_settings)
    if new_scheme is not None:
        quant_default_settings[quant_type][QuantConfigLiteral.quant_scheme] = new_scheme
    if new_dtype is not None:
        quant_default_settings[quant_type][QuantConfigLiteral.quant_dtype] = new_dtype
    return


class QuantSettings(object):
    def __init__(self):
        self._fields: Dict[str, Any] = {}
        for t in QuantType:
            self._fields[t] = {}

    def set(self, quant_type, target, value):
        assert quant_type in QuantType or quant_type in QuantType.valid_values()
        self._fields[quant_type][target] = value

    def get(self, quant_type, target):
        return self._fields[quant_type][target]

    def has(self, quant_type, target):
        return target in self._fields[quant_type]

    def is_per_channel(self, quant_type):
        scheme = self.get(quant_type, QuantConfigLiteral.quant_scheme)
        return scheme in [QuantScheme.per_channel_affine, QuantScheme.per_channel_symmetric]

    def update_from_config(self, config, quant_type):
        def get_config(config, quant_type, target):

            if not config.get(target):
                return None

            if isinstance(config[target], str):
                return config[target]
            else:
                return config[target].get(quant_type)

        targets = [QuantConfigLiteral.quant_scheme, QuantConfigLiteral.quant_dtype]
        for t in targets:
            val = quant_default_settings[quant_type][t]
            config_val = get_config(config, quant_type, t)
            if config_val:
                val = config_val
            self.set(quant_type, t, val)

        bits = get_bits_length(config, quant_type)
        qmin, qmax = calculate_qmin_qmax(bits, self.get(quant_type, QuantConfigLiteral.quant_dtype))
        self.set(quant_type, QuantConfigLiteral.qmin, qmin)
        self.set(quant_type, QuantConfigLiteral.qmax, qmax)

    def get_quant_shape(self, shape, quant_type):
        default_idx = 0 if quant_type == QuantType.weight else 1
        if self.has(quant_type, QuantConfigLiteral.quant_scheme):
            if self.is_per_channel(quant_type):
                quant_shape = [1 if idx != default_idx else s for idx, s in enumerate(shape)]
            else:
                quant_shape = []
        else:
            # This is for layer that will quantize weight and only record input.
            # We use global default setting. However, this setting may be wrong.
            global_scheme = quant_default_settings[quant_type][QuantConfigLiteral.quant_scheme]
            if global_scheme in [QuantScheme.per_channel_affine, QuantScheme.per_channel_symmetric]:
                quant_shape = [1 if idx != default_idx else s for idx, s in enumerate(shape)]
            else:
                quant_shape = []
        return quant_shape

    def get_qmin_qmax(self, quan_type):
        qmin = self.get(quan_type, QuantConfigLiteral.qmin)
        qmax = self.get(quan_type, QuantConfigLiteral.qmax)
        return qmin, qmax

    def get_target_dim(self, quant_type):
        # for weight: c_out x c_in x (h) * (w)
        # for feature maps: batch * channel * (t) * h * w
        # other type is not supported for now
        default_idx = 0 if quant_type == QuantType.weight else 1
        if self.has(quant_type, QuantConfigLiteral.quant_scheme) and \
                self.is_per_channel(quant_type):
            target_dim = default_idx
        else:
            target_dim = None
        return target_dim
