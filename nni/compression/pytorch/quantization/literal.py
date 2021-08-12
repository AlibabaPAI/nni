from enum import Enum


class OurEnum(Enum):
    @classmethod
    def valid_values(cls):
        return [x.value for x in cls]


class QuantScheme(str, OurEnum):
    per_tensor_affine = 'per_tensor_affine'
    per_tensor_symmetric = 'per_tensor_symmetric'
    per_channel_affine = 'per_channel_affine'
    per_channel_symmetric = 'per_channel_symmetric'


class QuantDtype(str, OurEnum):
    uint = 'uint'
    int = 'int'


class QuantType(str, OurEnum):
    input = 'input'
    weight = 'weight'
    output = 'output'

    def type_to_scale_zero_point_name(self):
        if self == QuantType.input:
            return QuantConfigLiteral.input_scale, QuantConfigLiteral.input_zero_point
        elif self == QuantType.weight:
            return QuantConfigLiteral.weight_scale, QuantConfigLiteral.weight_zero_point
        elif self == QuantType.output:
            return QuantConfigLiteral.output_scale, QuantConfigLiteral.output_zero_point
        else:
            raise TypeError


class QuantConfigLiteral(str, OurEnum):
    quant_settings = 'quant_settings'
    quant_scheme = 'quant_scheme'
    quant_dtype = 'quant_dtype'
    qmin = 'qmin'
    qmax = 'qmax'
    input_scale = 'input_scale'
    input_zero_point = 'input_zero_point'
    output_scale = 'output_scale'
    output_zero_point = 'output_zero_point'
    weight_scale = 'weight_scale'
    weight_zero_point = 'weight_zero_point'


BN_FOLD_OP = ["Conv2d"]
BN_FOLD_TAG = 'BN_FOLD_TAG'
