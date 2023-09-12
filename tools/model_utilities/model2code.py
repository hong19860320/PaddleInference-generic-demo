# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
import argparse

parser = argparse.ArgumentParser(
    description='Code Generator for converting paddle model to python code.')
parser.add_argument('--model_path_prefix', default='./simple_model/model')
parser.add_argument('--code_dir', default='./output_code')
args = parser.parse_args()
print(args)

paddle.enable_static()


def paddle_dtype2numpy_dtype(paddle_dtype):
    if paddle_dtype == core.VarDesc.VarType.FP32:
        return np.float32
    elif paddle_dtype == core.VarDesc.VarType.FP64:
        return np.float64
    elif paddle_dtype == core.VarDesc.VarType.FP16:
        return np.float16
    elif paddle_dtype == core.VarDesc.VarType.INT32:
        return np.int32
    elif paddle_dtype == core.VarDesc.VarType.INT16:
        return np.int16
    elif paddle_dtype == core.VarDesc.VarType.INT64:
        return np.int64
    elif paddle_dtype == core.VarDesc.VarType.BOOL:
        return np.bool_
    elif paddle_dtype == core.VarDesc.VarType.BF16:
        return np.uint16
    elif paddle_dtype == core.VarDesc.VarType.UINT8:
        return np.uint8
    elif paddle_dtype == core.VarDesc.VarType.INT8:
        return np.int8
    elif paddle_dtype == core.VarDesc.VarType.COMPLEX64:
        return np.complex64
    elif paddle_dtype == core.VarDesc.VarType.COMPLEX128:
        return np.complex128
    else:
        raise ValueError("Unsupport to convert paddle dtype %s to string" %
                         paddle_dtype)


def paddle_dtype2string(paddle_dtype):
    if paddle_dtype == core.VarDesc.VarType.FP32:
        return 'float32'
    elif paddle_dtype == core.VarDesc.VarType.FP64:
        return 'float64'
    elif paddle_dtype == core.VarDesc.VarType.FP16:
        return 'float16'
    elif paddle_dtype == core.VarDesc.VarType.INT32:
        return 'int32'
    elif paddle_dtype == core.VarDesc.VarType.INT16:
        return 'int16'
    elif paddle_dtype == core.VarDesc.VarType.INT64:
        return 'int64'
    elif paddle_dtype == core.VarDesc.VarType.BOOL:
        return 'bool'
    elif paddle_dtype == core.VarDesc.VarType.BF16:
        return 'uint16'
    elif paddle_dtype == core.VarDesc.VarType.UINT8:
        return 'uint8'
    elif paddle_dtype == core.VarDesc.VarType.INT8:
        return 'int8'
    elif paddle_dtype == core.VarDesc.VarType.COMPLEX64:
        return 'complex64'
    elif paddle_dtype == core.VarDesc.VarType.COMPLEX128:
        return 'complex128'
    else:
        raise ValueError("Unsupport to convert paddle dtype %s to string" %
                         paddle_dtype)


def numpy_dtype2string(numpy_dtype):
    if numpy_dtype == np.float32:
        return 'float32'
    elif numpy_dtype == np.float64:
        return 'float64'
    elif numpy_dtype == np.float16:
        return 'float16'
    elif numpy_dtype == np.int32:
        return 'int32'
    elif numpy_dtype == np.int16:
        return 'int16'
    elif numpy_dtype == np.int64:
        return 'int64'
    elif numpy_dtype == np.bool_:
        return 'bool'
    elif numpy_dtype == np.uint16:
        return 'uint16'
    elif numpy_dtype == np.uint8:
        return 'uint8'
    elif numpy_dtype == np.int8:
        return 'int8'
    elif numpy_dtype == np.complex64:
        return 'complex64'
    elif numpy_dtype == np.complex128:
        return 'complex128'
    else:
        raise ValueError("Unsupport to convert numpy dtype %s to string" %
                         numpy_dtype)


def python_array2string(python_array, quote=True, separator=','):
    if type(python_array) is tuple:
        python_array = np.array(python_array)
    if type(python_array) is list:
        python_array = np.array(python_array)
    if type(python_array) is np.ndarray:
        if quote:
            return np.array2string(
                python_array,
                max_line_width=2048,
                separator=separator,
                prefix='',
                suffix='')
        else:
            return np.array2string(
                python_array,
                max_line_width=2048,
                separator=separator,
                prefix='',
                suffix='',
                formatter={'str_kind': lambda x: x})
    else:
        raise ValueError("Unsupport to convert python array to string")


def check_broadcast(x_shape, y_shape, axis=-1):
    x_len = len(x_shape)
    y_len = len(y_shape)
    max_len = max(x_len, y_len)
    axis = (axis + max_len) % max_len
    if x_len == max_len:
        if y_len + axis != max_len:
            return False
        for x_idx in range(max_len - y_len, max_len):
            y_idx = x_idx + y_len - max_len
            if x_shape[x_idx] != y_shape[y_idx] and x_shape[
                    x_idx] != 1 and y_shape[y_idx] != 1:
                return False
    else:
        if x_len + axis != max_len:
            return False
        for y_idx in range(max_len - x_len, max_len):
            x_idx = y_idx + x_len - max_len
            if x_shape[x_idx] != y_shape[y_idx] and x_shape[
                    x_idx] != 1 and y_shape[y_idx] != 1:
                return False
    return True


class CodeGenerator:
    def gen_name(self, name):
        syms = ['.', ':', '/']
        for sym in syms:
            if sym in name:
                name = name.replace(sym, "_")
        return name

    def gen_names(self, names):
        renames = []
        for name in names:
            renames.append(self.gen_name(name))
        return renames

    def gen_indent(self):
        return '    ' * self.cur_indent_size

    def gen_return(self):
        return '\n'

    def gen_head(self):
        self.generated_code += self.gen_indent(
        ) + '# Declare placeholders' + self.gen_return()
        for feed_target_name in self.feed_target_names:
            feed_target_var = self.program.global_block().var(feed_target_name)
            feed_target_shape = python_array2string(feed_target_var.shape)
            feed_target_dtype = paddle_dtype2string(feed_target_var.dtype)
            self.generated_code += self.gen_indent() + self.gen_name(
                feed_target_name
            ) + ' = paddle.static.data(name=\'' + feed_target_name + '\', shape=' + feed_target_shape + ', dtype=\'' + feed_target_dtype + '\')' + self.gen_return(
            )

    def gen_tail(self):
        self.generated_code += self.gen_indent(
        ) + '# Compile and output an inference model' + self.gen_return()
        self.generated_code += self.gen_indent(
        ) + 'exe = paddle.static.Executor(place)' + self.gen_return()
        self.generated_code += self.gen_indent(
        ) + 'startup_program = paddle.static.default_startup_program()' + self.gen_return(
        )
        self.generated_code += self.gen_indent(
        ) + 'exe.run(startup_program)' + self.gen_return()
        fetch_target_names = self.gen_name(self.fetch_targets[0].name)
        for fetch_target_idx in range(1, len(self.fetch_targets)):
            fetch_target_names += ',' + self.gen_name(self.fetch_targets[
                fetch_target_idx].name)
        self.generated_code += self.gen_indent(
        ) + 'paddle.static.save_inference_model(\'./model\', ' + python_array2string(
            self.gen_names(self.feed_target_names),
            False) + ', [' + fetch_target_names + '], exe)' + self.gen_return()
        self.generated_code += self.gen_indent(
        ) + '# Prepare the input data, reload and run the inference model' + self.gen_return(
        )
        self.generated_code += self.gen_indent(
        ) + '# [inference_program, feed_target_names, fetch_targets] = paddle.static.load_inference_model(\'./model\', exe)' + self.gen_return(
        )
        for feed_target_name in self.feed_target_names:
            feed_target_var = self.program.global_block().var(feed_target_name)
            feed_target_shape = python_array2string(
                [1 if i == -1 else i for i in feed_target_var.shape])
            feed_target_dtype = paddle_dtype2string(feed_target_var.dtype)
            self.generated_code += self.gen_indent() + '# ' + self.gen_name(
                feed_target_name
            ) + '_tensor = np.zeros(shape=' + feed_target_shape + ', dtype=\'' + feed_target_dtype + '\')' + self.gen_return(
            )
        feed_target_dict = '\'' + self.feed_target_names[
            0] + '\': ' + self.gen_name(self.feed_target_names[0]) + '_tensor'
        for feed_target_idx in range(1, len(self.feed_target_names)):
            feed_target_dict += ', \'' + self.feed_target_names[
                feed_target_idx] + '\': ' + self.gen_name(
                    self.feed_target_names[feed_target_idx]) + '_tensor'
        self.generated_code += self.gen_indent(
        ) + '# output_tensors = exe.run(inference_program, feed={' + feed_target_dict + '}, fetch_list=fetch_targets, return_numpy=True)' + self.gen_return(
        )
        self.generated_code += self.gen_indent(
        ) + '# print(output_tensors)' + self.gen_return()
        self.generated_code += self.gen_indent(
        ) + 'print(\'Done.\')' + self.gen_return() + self.gen_return()
        self.cur_indent_size -= 1
        self.generated_code += self.gen_indent(
        ) + 'if __name__ == \'__main__\':' + self.gen_return()
        self.cur_indent_size += 1
        self.generated_code += self.gen_indent() + 'main()' + self.gen_return()

    def gen_param(self, name, mode=0):
        # mode = 0: persitable variable
        # mode = 1: parameter variable
        # mode = 2: parameter attribute
        if name in self.generated_params:
            return None
        if self.global_scope.find_var(name) is None:
            return None
        self.generated_params.append(name)
        param = np.array(self.global_scope.var(name).get_tensor())
        path = self.param_dir + os.sep + name + ".npy"
        shape = python_array2string(param.shape)
        dtype = numpy_dtype2string(param.dtype)
        np.save(path, param)
        cur_indent_size = self.cur_indent_size
        self.cur_indent_size = self.init_indent_size
        if mode == 1:
            self.generated_vars += self.gen_indent() + self.gen_name(
                name
            ) + ' = paddle.static.create_parameter(name=\'' + name + '\', shape=' + shape + ', dtype=\'' + dtype + '\', attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(np.load(\'./' + self.param_name + os.sep + name + '.npy\')), trainable=False))' + self.gen_return(
            )
        elif mode == 2:
            self.generated_vars += self.gen_indent() + self.gen_name(
                name
            ) + ' = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(np.load(\'./' + self.param_name + os.sep + name + '.npy\')), trainable=False)' + self.gen_return(
            )
        else:
            self.generated_vars += self.gen_indent() + self.gen_name(
                name
            ) + ' = main_program.global_block().create_var(name=\'' + name + '\', shape=' + shape + ', dtype=\'' + dtype + '\', persistable=True)' + self.gen_return(
            )
            self.generated_vars += self.gen_indent(
            ) + 'global_scope.var(\'' + name + '\').get_tensor().set(np.load(\'./' + self.param_name + os.sep + name + '.npy\'), place)' + self.gen_return(
            )
        self.cur_indent_size = cur_indent_size
        return param

    def gen_params(self, names, mode=0):
        params = []
        for name in names:
            params.append(self.gen_param(name, mode))
        return params

    def gen_arg_max(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        axis = op_desc.attr('axis')
        axis = str(axis) if axis else 'None'
        keepdim = str(op_desc.attr('keepdims'))
        dtype = paddle_dtype2string(op_desc.attr('dtype'))
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ' = paddle.argmax(' + self.gen_name(
            x_name
        ) + ', axis=' + axis + ', keepdim=' + keepdim + ', dtype=\'' + dtype + '\')' + self.gen_return(
        )

    def gen_assign_value(self, block_idx, op_desc):
        out_name = op_desc.output('Out')[0]
        dtype = op_desc.attr('dtype')
        if dtype == core.VarDesc.VarType.BOOL:
            values = op_desc.attr('bool_values')
        elif dtype == core.VarDesc.VarType.FP32:
            values = op_desc.attr('fp32_values')
        elif dtype == core.VarDesc.VarType.INT32:
            values = op_desc.attr('int32_values')
        elif dtype == core.VarDesc.VarType.INT64:
            values = op_desc.attr('int64_values')
        else:
            raise ValueError('Unsupport to get values for dtype \'%d\'' %
                             dtype)
        dtype = paddle_dtype2string(dtype)
        values = python_array2string(values)
        shape = python_array2string(op_desc.attr('shape'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name
        ) + ' = paddle.assign(np.array(' + values + ', \'' + dtype + '\').reshape(' + shape + '))' + self.gen_return(
        )

    def gen_batch_norm(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        bias_name = op_desc.input('Bias')[0]
        self.gen_param(bias_name)
        mean_name = op_desc.input('Mean')[0]
        self.gen_param(mean_name)
        scale_name = op_desc.input('Scale')[0]
        self.gen_param(scale_name)
        variance_name = op_desc.input('Variance')[0]
        self.gen_param(variance_name)
        y_name = op_desc.output('Y')[0]
        epsilon = str(op_desc.attr('epsilon'))
        momentum = str(op_desc.attr('momentum'))
        data_layout = op_desc.attr('data_layout')
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(y_name) + ' = F.batch_norm(' + self.gen_name(
            x_name
        ) + ', ' + self.gen_name(mean_name) + ', ' + self.gen_name(
            variance_name
        ) + ', ' + self.gen_name(scale_name) + ', ' + self.gen_name(
            bias_name
        ) + ', False, ' + momentum + ', ' + epsilon + ', \'' + data_layout + '\')' + self.gen_return(
        )

    def gen_binary_ops(self, block_idx, op_desc):
        op_type = op_desc.type()
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        y_name = op_desc.input('Y')[0]
        self.gen_param(y_name)
        out_name = op_desc.output('Out')[0]
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = '
        if op_type == 'equal':
            self.generated_apis += 'paddle.equal(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'not_equal':
            self.generated_apis += 'paddle.not_equal(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'less_than':
            self.generated_apis += 'paddle.less_than(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'greater_than':
            self.generated_apis += 'paddle.greater_than(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'greater_equal':
            self.generated_apis += 'paddle.greater_equal(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'less_equal':
            self.generated_apis += 'paddle.less_equal(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'bitwise_and':
            self.generated_apis += 'paddle.bitwise_and(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'bitwise_or':
            self.generated_apis += 'paddle.bitwise_or(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'bitwise_xor':
            self.generated_apis += 'paddle.bitwise_xor(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'logical_and':
            self.generated_apis += 'paddle.logical_and(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'logical_or':
            self.generated_apis += 'paddle.logical_or(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'logical_xor':
            self.generated_apis += 'paddle.logical_xor(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        else:
            raise ValueError(
                'Unsupport to generate code for binary op \'%s\'' % op_type)
        self.generated_apis += self.gen_return()

    def gen_block(self, block_idx):
        num_blocks = self.program.num_blocks
        block = self.program.block(block_idx)
        num_ops = len(block.ops)
        indent_inc = 0 if block_idx == 0 else 1
        self.cur_indent_size += indent_inc
        for op_idx in range(num_ops):
            op_desc = block.ops[op_idx].desc
            op_type = op_desc.type()
            print('[%d/%d %d/%d] generating %s ...' %
                  (op_idx + 1, num_ops, block_idx + 1, num_blocks, op_type))
            if op_type == 'feed' or op_type == 'fetch':
                continue
            try:
                self.gen_funcs[op_type](block_idx, op_desc)
            except KeyError:
                raise ValueError('Unsupport to generate code for op \'%s\'' %
                                 op_type)
        self.cur_indent_size -= indent_inc

    def gen_cast(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        in_dtype = paddle_dtype2string(op_desc.attr('in_dtype'))
        out_dtype = paddle_dtype2string(op_desc.attr('out_dtype'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = paddle.cast(' + self.gen_name(
                x_name) + ', \'' + out_dtype + '\')' + self.gen_return()

    def gen_clip(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        if 'Min' in op_desc.input_names() and len(op_desc.input('Min')) > 0:
            min_name = op_desc.input('Min')[0]
            self.gen_param(min_name)
            min = self.gen_name(min_name)
        else:
            min = str(op_desc.attr('min'))
        if 'Max' in op_desc.input_names() and len(op_desc.input('Max')) > 0:
            max_name = op_desc.input('Max')[0]
            self.gen_param(max_name)
            max = self.gen_name(max_name)
        else:
            max = str(op_desc.attr('max'))
        out_name = op_desc.output('Out')[0]
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ' = paddle.clip(' + self.gen_name(
            x_name) + ', min=' + min + ', max=' + max + ')' + self.gen_return()

    def gen_concat(self, block_idx, op_desc):
        x_names = op_desc.input('X')
        self.gen_params(x_names)
        out_name = op_desc.output('Out')[0]
        axis = str(op_desc.attr('axis'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = paddle.concat(' + python_array2string(
                self.gen_names(x_names),
                False) + ', axis=' + axis + ')' + self.gen_return()

    def gen_conditional_block(self, block_idx, op_desc):
        cond_names = op_desc.input('Cond')
        self.gen_params(cond_names)
        is_scalar_condition = str(op_desc.attr('is_scalar_condition'))
        sub_block_idx = op_desc.attr('sub_block').id
        self.generated_apis += self.gen_indent() + 'condition_block_' + str(
            sub_block_idx
        ) + ' = ConditionalBlock(inputs=' + python_array2string(
            self.gen_names(cond_names), False
        ) + ', is_scalar_condition=' + is_scalar_condition + ')' + self.gen_return(
        )
        self.generated_apis += self.gen_indent(
        ) + 'with condition_block_' + str(
            sub_block_idx) + '.block():' + self.gen_return()
        self.gen_block(sub_block_idx)

    def gen_conv2d(self, block_idx, op_desc):
        input_name = op_desc.input('Input')[0]
        self.gen_param(input_name)
        filter_name = op_desc.input('Filter')[0]
        self.gen_param(filter_name)
        output_name = op_desc.output('Output')[0]
        strides = python_array2string(op_desc.attr('strides'))
        paddings = python_array2string(op_desc.attr('paddings'))
        if op_desc.has_attr('padding_algorithm'):
            padding_algorithm = op_desc.attr('padding_algorithm')
            if padding_algorithm != 'EXPLICIT':
                paddings = '\'' + op_desc.attr('padding_algorithm') + '\''
        dilations = python_array2string(op_desc.attr('dilations'))
        groups = str(op_desc.attr('groups'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            output_name
        ) + ' = F.conv2d(' + self.gen_name(input_name) + ', ' + self.gen_name(
            filter_name
        ) + ', bias=None, stride=' + strides + ', padding=' + paddings + ', dilation=' + dilations + ', groups=' + groups + ', data_format=\'NCHW\')' + self.gen_return(
        )

    def gen_conv2d_transpose(self, block_idx, op_desc):
        input_name = op_desc.input('Input')[0]
        self.gen_param(input_name)
        filter_name = op_desc.input('Filter')[0]
        self.gen_param(filter_name)
        output_name = op_desc.output('Output')[0]
        strides = python_array2string(op_desc.attr('strides'))
        paddings = python_array2string(op_desc.attr('paddings'))
        if op_desc.has_attr('padding_algorithm'):
            padding_algorithm = op_desc.attr('padding_algorithm')
            if padding_algorithm != 'EXPLICIT':
                paddings = '\'' + op_desc.attr('padding_algorithm') + '\''
        output_padding = op_desc.attr('output_padding')
        output_padding = python_array2string(
            output_padding) if output_padding else '0'
        dilations = python_array2string(op_desc.attr('dilations'))
        groups = str(op_desc.attr('groups'))
        output_size = op_desc.attr('output_size')
        output_size = python_array2string(
            output_size) if output_size else 'None'
        self.generated_apis += self.gen_indent() + self.gen_name(
            output_name
        ) + ' = F.conv2d_transpose(' + self.gen_name(
            input_name
        ) + ', ' + self.gen_name(
            filter_name
        ) + ', bias=None, stride=' + strides + ', padding=' + paddings + ', output_padding=' + output_padding + ', dilation=' + dilations + ', groups=' + groups + ', data_format=\'NCHW\', output_size=' + output_size + ')' + self.gen_return(
        )

    def gen_cumsum(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        axis = str(op_desc.attr('axis'))
        if op_desc.has_attr('exclusive'):
            exclusive = op_desc.attr('exclusive')
            assert exclusive == False
        if op_desc.has_attr('flatten'):
            flatten = op_desc.attr('flatten')
            assert flatten == False
        if op_desc.has_attr('reverse'):
            reverse = op_desc.attr('reverse')
            assert reverse == False
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ' = paddle.cumsum(' + self.gen_name(
            x_name) + ', axis=' + axis + ', dtype=None)' + self.gen_return()

    def gen_dropout(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        p = str(op_desc.attr('dropout_prob'))
        is_test = op_desc.attr('is_test')
        mode = op_desc.attr('dropout_implementation')
        seed = op_desc.attr('seed')
        fix_seed = op_desc.attr('fix_seed')
        assert is_test == True
        assert mode == 'downgrade_in_infer'
        assert seed == 0
        assert fix_seed == False
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ' = F.dropout(' + self.gen_name(
            x_name
        ) + ', p=' + p + ', axis=None, training=False, mode=\'downscale_in_infer\')' + self.gen_return(
        )

    def _elementwise_ops_with_axis(self, op_type, x_name, y_name, out_name,
                                   axis):
        # No paddle api found for elementwise ops with axis, use LayerHelper directly.
        self.generated_apis += self.gen_indent(
        ) + 'helper = LayerHelper(\'' + op_type + '\')' + self.gen_return()
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name
        ) + ' = helper.create_variable_for_type_inference(dtype=' + self.gen_name(
            x_name) + '.dtype)' + self.gen_return()
        self.generated_apis += self.gen_indent(
        ) + 'helper.append_op(type=\'' + op_type + '\', inputs={\'X\': ' + self.gen_name(
            x_name) + ', \'Y\': ' + self.gen_name(
                y_name) + '}, outputs={\'Out\': ' + self.gen_name(
                    out_name) + '}, attrs={\'axis\': ' + str(
                        axis) + '})' + self.gen_return()

    def gen_elementwise_ops(self, block_idx, op_desc):
        op_type = op_desc.type()
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        y_name = op_desc.input('Y')[0]
        self.gen_param(y_name)
        out_name = op_desc.output('Out')[0]
        axis = op_desc.attr('axis')
        axis = axis if axis else '-1'
        x_shape = self.program.block(block_idx).var(x_name).shape
        y_shape = self.program.block(block_idx).var(y_name).shape
        should_broadcast = not check_broadcast(x_shape, y_shape, axis)
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = '
        if op_type == 'elementwise_add':
            if should_broadcast:
                self.generated_apis += 'paddle.tensor.math._add_with_axis(' + self.gen_name(
                    x_name) + ', ' + self.gen_name(y_name) + ', axis=' + str(
                        axis) + ')'
            else:
                self.generated_apis += 'paddle.add(' + self.gen_name(
                    x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'elementwise_sub':
            if should_broadcast:
                self.generated_apis += 'paddle.tensor.math._subtract_with_axis(' + self.gen_name(
                    x_name) + ', ' + self.gen_name(y_name) + ', axis=' + str(
                        axis) + ')'
            else:
                self.generated_apis += 'paddle.subtract(' + self.gen_name(
                    x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'elementwise_mul':
            if should_broadcast:
                self.generated_apis += 'paddle.tensor.math._multiply_with_axis(' + self.gen_name(
                    x_name) + ', ' + self.gen_name(y_name) + ', axis=' + str(
                        axis) + ')'
            else:
                self.generated_apis += 'paddle.multiply(' + self.gen_name(
                    x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'elementwise_div':
            if should_broadcast:
                self.generated_apis += 'paddle.tensor.math._divide_with_axis(' + self.gen_name(
                    x_name) + ', ' + self.gen_name(y_name) + ', axis=' + str(
                        axis) + ')'
            else:
                self.generated_apis += 'paddle.divide(' + self.gen_name(
                    x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'elementwise_max':
            if should_broadcast:
                self._elementwise_ops_with_axis(op_type, x_name, y_name,
                                                out_name, axis)
            else:
                self.generated_apis += 'paddle.maximum(' + self.gen_name(
                    x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'elementwise_min':
            if should_broadcast:
                self._elementwise_ops_with_axis(op_type, x_name, y_name,
                                                out_name, axis)
            else:
                self.generated_apis += 'paddle.minimum(' + self.gen_name(
                    x_name) + ', ' + self.gen_name(y_name) + ')'
        else:
            raise ValueError(
                'Unsupport to generate code for binary op \'%s\'' % op_type)
        self.generated_apis += self.gen_return()

    def gen_elementwise_pow(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        if 'Y' in op_desc.input_names() and len(op_desc.input('Y')) > 0:
            y_name = op_desc.input('Y')[0]
            self.gen_param(y_name)
            axis = op_desc.attr('axis')
            axis = axis if axis else '-1'
            x_shape = self.program.block(block_idx).var(x_name).shape
            y_shape = self.program.block(block_idx).var(y_name).shape
            should_broadcast = not check_broadcast(x_shape, y_shape, axis)
            if should_broadcast:
                self._elementwise_ops_with_axis('elementwise_pow', x_name,
                                                y_name, out_name, axis)
            else:
                self.generated_apis += self.gen_indent() + self.gen_name(
                    out_name) + ' = paddle.pow(' + self.gen_name(
                        x_name) + ', ' + self.gen_name(
                            y_name) + ')' + self.gen_return()
        else:
            factor = str(op_desc.attr('factor'))
            self.generated_apis += self.gen_indent() + self.gen_name(
                out_name) + ' = paddle.pow(' + self.gen_name(
                    x_name) + ', ' + factor + ')' + self.gen_return()

    def gen_elu(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        alpha = str(op_desc.attr('alpha'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = F.elu(' + self.gen_name(
                x_name) + ', alpha=' + alpha + ')' + self.gen_return()

    def gen_expand(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        shape = python_array2string(op_desc.attr('shape'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = paddle.expand(' + self.gen_name(
                x_name) + ', ' + shape + ')' + self.gen_return()

    def gen_expand_as(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        y_name = op_desc.input('Y')[0]
        self.gen_param(y_name)
        out_name = op_desc.output('Out')[0]
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ' = paddle.expand_as(' + self.gen_name(
            x_name) + ', ' + self.gen_name(y_name) + ')' + self.gen_return()

    def gen_fill_any_like(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        fill_value = str(op_desc.attr('value'))
        dtype = op_desc.attr('dtype')
        dtype = '\'' + paddle_dtype2string(dtype) + '\'' if dtype else 'None'
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ' = paddle.full_like(' + self.gen_name(
            x_name
        ) + ', ' + fill_value + ', dtype=' + dtype + ')' + self.gen_return()

    def gen_fill_constant(self, block_idx, op_desc):
        if 'ShapeTensor' in op_desc.input_names() and len(
                op_desc.input('ShapeTensor')) > 0:
            shape_tensor_name = op_desc.input('ShapeTensor')[0]
            self.gen_param(shape_tensor_name)
            shape = self.gen_name(shape_tensor_name)
        else:
            shape = python_array2string(op_desc.attr('shape'))
        if 'ValueTensor' in op_desc.input_names() and len(
                op_desc.input('ValueTensor')) > 0:
            value_tensor_name = op_desc.input('ValueTensor')[0]
            self.gen_param(value_tensor_name)
            fill_value = self.gen_name(value_tensor_name)
        else:
            fill_value = str(op_desc.attr('value'))
        out_name = op_desc.output('Out')[0]
        dtype = paddle_dtype2string(op_desc.attr('dtype'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name
        ) + ' = paddle.full(' + shape + ', ' + fill_value + ', dtype=\'' + dtype + '\')' + self.gen_return(
        )

    def gen_flip(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        axis = python_array2string(op_desc.attr('axis'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = paddle.flip(' + self.gen_name(
                x_name) + ', axis=' + axis + ')' + self.gen_return()

    def gen_gather(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        index_name = op_desc.input('Index')[0]
        self.gen_param(index_name)
        out_name = op_desc.output('Out')[0]
        axis = str(op_desc.attr('axis'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = paddle.gather(' + self.gen_name(
                x_name) + ', ' + self.gen_name(
                    index_name) + ', axis=' + axis + ')' + self.gen_return()

    def gen_gather_nd(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        index_name = op_desc.input('Index')[0]
        self.gen_param(index_name)
        out_name = op_desc.output('Out')[0]
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ' = paddle.gather_nd(' + self.gen_name(
            x_name) + ', ' + self.gen_name(index_name) + ')' + self.gen_return(
            )

    def gen_hard_sigmoid(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        slope = str(op_desc.attr('slope'))
        offset = str(op_desc.attr('offset'))
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ' = F.hardsigmoid(' + self.gen_name(
            x_name
        ) + ', slope=' + slope + ', offset=' + offset + ')' + self.gen_return()

    def gen_index_select(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        index_name = op_desc.input('Index')[0]
        self.gen_param(index_name)
        out_name = op_desc.output('Out')[0]
        axis = str(op_desc.attr('dim'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = paddle.index_select(' + self.gen_name(
                x_name) + ', ' + self.gen_name(
                    index_name) + ', axis=' + axis + ')' + self.gen_return()

    def gen_instance_norm(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        y_name = op_desc.output('Y')[0]
        epsilon = str(op_desc.attr('epsilon'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            y_name) + ' = F.instance_norm(' + self.gen_name(x_name)
        if 'Scale' in op_desc.input_names() and len(op_desc.input(
                'Scale')) > 0:
            scale_name = op_desc.input('Scale')[0]
            self.gen_param(scale_name)
            self.generated_apis += ', weight=' + self.gen_name(scale_name)
        if 'Bias' in op_desc.input_names() and len(op_desc.input('Bias')) > 0:
            bias_name = op_desc.input('Bias')[0]
            self.gen_param(bias_name)
            self.generated_apis += ', bias=' + self.gen_name(bias_name)
        self.generated_apis += ', eps=' + epsilon + ')' + self.gen_return()

    def gen_interp_ops(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_params(x_name)
        out_name = op_desc.output('Out')[0]
        size = 'None'
        scale_factor = 'None'
        if 'OutSize' in op_desc.input_names() and len(
                op_desc.input('OutSize')) > 0:
            out_size_name = op_desc.input('OutSize')[0]
            self.gen_param(out_size_name)
            size = self.gen_name(out_size_name)
        elif 'SizeTensor' in op_desc.input_names() and len(
                op_desc.input('SizeTensor')) > 0:
            size_tensor_name = op_desc.input('SizeTensor')[0]
            self.gen_param(size_tensor_name)
            size = self.gen_name(size_tensor_name)
        elif 'Scale' in op_desc.input_names() and len(op_desc.input(
                'Scale')) > 0:
            scale_name = op_desc.input('Scale')[0]
            self.gen_param(scale_name)
            scale_factor = self.gen_name(scale_name)
        else:
            out_d = op_desc.attr('out_d')
            out_h = op_desc.attr('out_h')
            out_w = op_desc.attr('out_w')
            scale = op_desc.attr('scale')
            if out_d == -1 and out_h == -1 and out_w != -1:
                size = python_array2string([out_w])
            elif out_d == -1 and out_h != -1 and out_w != -1:
                size = python_array2string([out_h, out_w])
            elif out_d != -1 and out_h != -1 and out_w != -1:
                size = python_array2string([out_d, out_h, out_w])
            elif scale:
                scale_factor = python_array2string(scale)
            else:
                raise ValueError(
                    '\'OutSize\', \'SizeTensor\', \'Scale\' tensor, and \'out_d\', \'out_h\', \'out_w\', \'scale\' attribute is not found or invalid!'
                )
        mode = op_desc.attr('interp_method')
        align_corners = str(op_desc.attr('align_corners'))
        align_mode = str(op_desc.attr('align_mode'))
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ' = F.interpolate(' + self.gen_name(
            x_name
        ) + ', size=' + size + ', scale_factor=' + scale_factor + ', mode=\'' + mode + '\', align_corners=' + align_corners + ', align_mode=' + align_mode + ', data_format=\'NCHW\')' + self.gen_return(
        )

    def gen_layer_norm(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        bias_name = op_desc.input('Bias')[0]
        self.gen_param(bias_name)
        scale_name = op_desc.input('Scale')[0]
        self.gen_param(scale_name)
        y_name = op_desc.output('Y')[0]
        epsilon = str(op_desc.attr('epsilon'))
        begin_norm_axis = op_desc.attr('begin_norm_axis')
        x_shape = self.program.block(block_idx).var(x_name).shape
        normalized_shape = python_array2string(x_shape[begin_norm_axis:])
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(y_name) + ' = F.layer_norm(' + self.gen_name(
            x_name) + ', ' + normalized_shape + ', weight=' + self.gen_name(
                scale_name) + ', bias=' + self.gen_name(
                    bias_name
                ) + ', epsilon=' + epsilon + ')' + self.gen_return()

    def gen_leaky_relu(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        negative_slope = str(op_desc.attr('alpha'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = F.leaky_relu(' + self.gen_name(
                x_name
            ) + ', negative_slope=' + negative_slope + ')' + self.gen_return()

    def gen_linspace(self, block_idx, op_desc):
        num_name = op_desc.input('Num')[0]
        self.gen_param(num_name)
        start_name = op_desc.input('Start')[0]
        self.gen_param(start_name)
        stop_name = op_desc.input('Stop')[0]
        self.gen_param(stop_name)
        out_name = op_desc.output('Out')[0]
        dtype = op_desc.attr('dtype')
        dtype = '\'' + paddle_dtype2string(dtype) + '\'' if dtype else 'None'
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ' = paddle.linspace(' + self.gen_name(
            start_name) + ', ' + self.gen_name(
                stop_name) + ', ' + self.gen_name(
                    num_name) + ', dtype=' + dtype + ')' + self.gen_return()

    def gen_lookup_table(self, block_idx, op_desc):
        ids_name = op_desc.input('Ids')[0]
        self.gen_param(ids_name)
        w_name = op_desc.input('W')[0]
        self.gen_param(w_name)
        out_name = op_desc.output('Out')[0]
        padding_idx = op_desc.attr('padding_idx')
        padding_idx = str(
            padding_idx) if padding_idx and padding_idx != -1 else 'None'
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name
        ) + ' = F.embedding(' + self.gen_name(ids_name) + ', ' + self.gen_name(
            w_name
        ) + ', padding_idx=' + padding_idx + ', sparse=False)' + self.gen_return(
        )

    def gen_matmul(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        y_name = op_desc.input('Y')[0]
        self.gen_param(y_name)
        out_name = op_desc.output('Out')[0]
        transpose_x = str(op_desc.attr('trans_x'))
        transpose_y = str(op_desc.attr('trans_y'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name
        ) + ' = paddle.matmul(' + self.gen_name(x_name) + ', ' + self.gen_name(
            y_name
        ) + ', transpose_x=' + transpose_x + ', transpose_y=' + transpose_y + ')' + self.gen_return(
        )

    def gen_meshgrid(self, block_idx, op_desc):
        x_names = op_desc.input('X')
        self.gen_params(x_names)
        out_names = op_desc.output('Out')
        self.generated_apis += self.gen_indent() + python_array2string(
            self.gen_names(out_names),
            False) + ' = paddle.meshgrid(' + python_array2string(
                self.gen_names(x_names), False) + ')' + self.gen_return()

    def gen_pad(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        pad = python_array2string(op_desc.attr('paddings'))
        value = str(op_desc.attr('pad_value'))
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ' = F.pad(' + self.gen_name(
            x_name
        ) + ', pad=' + pad + ', mode=\'constant\', value=' + value + ', data_format=\'NCHW\')' + self.gen_return(
        )

    def gen_pad3d(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        pad = python_array2string(op_desc.attr('paddings'))
        mode = str(op_desc.attr('mode'))
        value = str(op_desc.attr('value'))
        data_format = str(op_desc.attr('data_format'))
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ' = F.pad(' + self.gen_name(
            x_name
        ) + ', pad=' + pad + ', mode=\'' + mode + '\', value=' + value + ', data_format=\'' + data_format + '\')' + self.gen_return(
        )

    def gen_p_norm(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        p = str(op_desc.attr('porder'))
        axis = str(op_desc.attr('axis'))
        keepdim = str(op_desc.attr('keepdim'))
        epsilon = op_desc.attr('epsilon')
        assert abs(epsilon - 9.999999960041972e-13) < 1e-6
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ' = paddle.linalg.norm(' + self.gen_name(
            x_name
        ) + ', p=' + p + ', axis=' + axis + ', keepdim=' + keepdim + ')' + self.gen_return(
        )

    def gen_pool2d(self, block_idx, op_desc):
        op_type = op_desc.type()
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        adaptive = op_desc.attr('adaptive')
        kernel_size = python_array2string(op_desc.attr('ksize'))
        pooling_type = ''
        return_mask = False
        self.generated_apis += self.gen_indent() + self.gen_name(out_name)
        if op_type == 'max_pool2d_with_index':
            # If the number of return values of adaptive_max_pool2d is fixed https://github.com/PaddlePaddle/Paddle/blob/1cea578edd29ba34b69ffe7fd18acfb2600a538b/python/paddle/nn/functional/pooling.py#L1971, delete the following code.
            if not adaptive:
                self.generated_apis += ', ' + self.gen_name(
                    op_desc.output('Mask')[0])
            return_mask = True
        else:
            pooling_type = op_desc.attr('pooling_type')
        self.generated_apis += ' = '
        if adaptive:
            if pooling_type == 'avg':
                self.generated_apis += 'F.adaptive_avg_pool2d(' + self.gen_name(
                    x_name) + ', ' + kernel_size + ')'
            elif pooling_type == 'max' or return_mask:
                self.generated_apis += 'F.adaptive_max_pool2d(' + self.gen_name(
                    x_name) + ', ' + kernel_size + ', return_mask=' + str(
                        return_mask) + ')'
            else:
                raise ValueError(
                    'Unsupport to generate code for pool2d op \'%s\'' %
                    op_type)
        else:
            ceil_mode = str(op_desc.attr('ceil_mode'))
            stride = python_array2string(op_desc.attr('strides'))
            padding = python_array2string(op_desc.attr('paddings'))
            if pooling_type == 'avg':
                exclusive = str(op_desc.attr('exclusive'))
                self.generated_apis += 'F.avg_pool2d(' + self.gen_name(
                    x_name
                ) + ', ' + kernel_size + ', stride=' + stride + ', padding=' + padding + ', ceil_mode=' + ceil_mode + ', exclusive=' + exclusive + ', divisor_override=None)'
            elif pooling_type == 'max' or return_mask:
                self.generated_apis += 'F.max_pool2d(' + self.gen_name(
                    x_name
                ) + ', ' + kernel_size + ', stride=' + stride + ', padding=' + padding + ', ceil_mode=' + ceil_mode + ', return_mask=' + str(
                    return_mask) + ')'
            else:
                raise ValueError(
                    'Unsupport to generate code for pool2d op \'%s\'' %
                    op_type)
        self.generated_apis += self.gen_return()

    def gen_range(self, block_idx, op_desc):
        start_name = op_desc.input('Start')[0]
        self.gen_param(start_name)
        if 'End' in op_desc.input_names() and len(op_desc.input('End')) > 0:
            end_name = op_desc.input('End')[0]
            self.gen_param(end_name)
        else:
            end_name = 'None'
        step_name = op_desc.input('Step')[0]
        self.gen_param(step_name)
        out_name = op_desc.output('Out')[0]
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = paddle.arange(start=' + self.gen_name(
                start_name) + ', end=' + self.gen_name(
                    end_name) + ', step=' + self.gen_name(
                        step_name) + ', dtype=None)' + self.gen_return()

    def gen_reduce_ops(self, block_idx, op_desc):
        op_type = op_desc.type()
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        keepdim = str(op_desc.attr('keep_dim'))
        reduce_all = op_desc.attr('reduce_all')
        if reduce_all:
            axis = 'None'
        else:
            axis = python_array2string(op_desc.attr('dim'))
        if op_type == 'reduce_any':
            api = 'any'
        elif op_type == 'reduce_all':
            api = 'all'
        elif op_type == 'reduce_mean':
            api = 'mean'
        elif op_type == 'reduce_sum':
            api = 'sum'
        elif op_type == 'reduce_min':
            api = 'min'
        elif op_type == 'reduce_max':
            api = 'max'
        elif op_type == 'reduce_prod':
            api = 'prod'
        else:
            raise ValueError(
                'Unsupport to generate code for reduce op \'%s\'' % op_type)
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ' = paddle.' + api + '(' + self.gen_name(
            x_name
        ) + ', axis=' + axis + ', keepdim=' + keepdim + ')' + self.gen_return()

    def gen_relu6(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        threshold = op_desc.attr('threshold')
        assert abs(threshold - 6.0) < 1e-6
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = F.relu(' + self.gen_name(
                x_name) + ')' + self.gen_return()

    def gen_reshape(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        if 'Shape' in op_desc.input_names() and len(op_desc.input(
                'Shape')) > 0:
            shape_name = op_desc.input('Shape')[0]
            self.gen_param(shape_name)
            shape = self.gen_name(shape_name)
        else:
            shape = python_array2string(op_desc.attr('shape'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = paddle.reshape(' + self.gen_name(
                x_name) + ', shape=' + shape + ')' + self.gen_return()

    def gen_rnn(self, block_idx, op_desc):
        input_name = op_desc.input('Input')[0]
        self.gen_param(input_name)
        out_name = op_desc.output('Out')[0]
        mode = op_desc.attr('mode')
        if mode == 'LSTM':
            pre_state_names = op_desc.input('PreState')
            self.gen_params(pre_state_names)
            state_names = op_desc.output('State')
            weight_list_names = op_desc.input('WeightList')
            self.gen_params(weight_list_names, 2)
            input_size = str(op_desc.attr('input_size'))
            hidden_size = str(op_desc.attr('hidden_size'))
            num_layers = str(op_desc.attr('num_layers'))
            direction = 'bidirect' if op_desc.attr(
                'is_bidirec') == True else 'forward'
            dropout = str(op_desc.attr('dropout_prob'))
            input_shape = self.program.block(block_idx).var(input_name).shape
            time_major = 'True' if input_shape[0] != -1 else 'False'
            self.generated_apis += self.gen_indent() + self.gen_name(
                out_name
            ) + ', (' + self.gen_name(state_names[0]) + ', ' + self.gen_name(
                state_names[1]
            ) + ')' + ' = paddle.nn.LSTM(' + input_size + ', ' + hidden_size + ', num_layers=' + num_layers + ', direction=\'' + direction + '\', time_major=' + time_major + ', dropout=' + dropout + ', weight_ih_attr=' + self.gen_name(
                weight_list_names[0]) + ', weight_hh_attr=' + self.gen_name(
                    weight_list_names[1]) + ', bias_ih_attr=' + self.gen_name(
                        weight_list_names[2]
                    ) + ', bias_hh_attr=' + self.gen_name(weight_list_names[
                        3]) + ')(' + self.gen_name(
                            input_name) + ', (' + self.gen_name(
                                pre_state_names[0]) + ', ' + self.gen_name(
                                    pre_state_names[
                                        1]) + '))' + self.gen_return()
        else:
            raise ValueError(
                'Unsupport to generate code for rnn op with mode \'%s\'' %
                mode)

    def gen_scale(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        scale = str(op_desc.attr('scale'))
        bias = str(op_desc.attr('bias'))
        bias_after_scale = str(op_desc.attr('bias_after_scale'))
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ' = paddle.scale(' + self.gen_name(
            x_name
        ) + ', scale=' + scale + ', bias=' + bias + ', bias_after_scale=' + bias_after_scale + ', act=None)' + self.gen_return(
        )

    def gen_scatter_nd_add(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        index_name = op_desc.input('Index')[0]
        self.gen_param(index_name)
        updates_name = op_desc.input('Updates')[0]
        self.gen_param(updates_name)
        out_name = op_desc.output('Out')[0]
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = paddle.scatter_nd_add(' + self.gen_name(
                x_name) + ', ' + self.gen_name(
                    index_name) + ', ' + self.gen_name(
                        updates_name) + ')' + self.gen_return()

    def gen_set_value(self, block_idx, op_desc):
        input_name = op_desc.input('Input')[0]
        self.gen_param(input_name)
        value_tensor_name = op_desc.input('ValueTensor')[0]
        self.gen_param(value_tensor_name)
        out_name = op_desc.output('Out')[0]
        axes = python_array2string(op_desc.attr('axes'))
        starts = python_array2string(op_desc.attr('starts'))
        ends = python_array2string(op_desc.attr('ends'))
        steps = python_array2string(op_desc.attr('steps'))
        decrease_axes = python_array2string(op_desc.attr('decrease_axes'))
        none_axes = python_array2string(op_desc.attr('none_axes'))
        dtype = str(op_desc.attr('dtype'))
        shape = python_array2string(op_desc.attr('shape'))
        # No paddle api found for set_value op, use LayerHelper directly.
        self.generated_apis += self.gen_indent(
        ) + 'helper = LayerHelper(\'set_value\')' + self.gen_return()
        if out_name != input_name:
            self.generated_apis += self.gen_indent() + self.gen_name(
                out_name
            ) + ' = helper.create_variable_for_type_inference(dtype=' + self.gen_name(
                input_name) + '.dtype)' + self.gen_return()
        self.generated_apis += self.gen_indent(
        ) + 'helper.append_op(type=\'set_value\', inputs={\'Input\': ' + self.gen_name(
            input_name
        ) + ', \'ValueTensor\': ' + self.gen_name(
            value_tensor_name
        ) + '}, outputs={\'Out\': ' + self.gen_name(
            out_name
        ) + '}, attrs={\'axes\': ' + axes + ', \'starts\': ' + starts + ', \'ends\': ' + ends + ', \'steps\': ' + steps + ', \'decrease_axes\': ' + decrease_axes + ', \'none_axes\': ' + none_axes + ', \'dtype\': ' + dtype + ', \'shape\': ' + shape + '})' + self.gen_return(
        )

    def gen_shape(self, block_idx, op_desc):
        input_name = op_desc.input('Input')[0]
        self.gen_param(input_name)
        out_name = op_desc.output('Out')[0]
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = paddle.shape(' + self.gen_name(
                input_name) + ')' + self.gen_return()

    def gen_share_data(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = ' + self.gen_name(
                x_name) + '.detach()' + self.gen_return()

    def gen_slice(self, block_idx, op_desc):
        input_name = op_desc.input('Input')[0]
        self.gen_param(input_name)
        out_name = op_desc.output('Out')[0]
        axes = python_array2string(op_desc.attr('axes'))
        starts = python_array2string(op_desc.attr('starts'))
        ends = python_array2string(op_desc.attr('ends'))
        decrease_axis = op_desc.attr('decrease_axis')
        if decrease_axis:
            # No paddle api found for slice op with attr 'decrease_axis', use LayerHelper directly.
            decrease_axis = python_array2string(op_desc.attr('decrease_axis'))
            self.generated_apis += self.gen_indent(
            ) + 'helper = LayerHelper(\'slice\')' + self.gen_return()
            self.generated_apis += self.gen_indent() + self.gen_name(
                out_name
            ) + ' = helper.create_variable_for_type_inference(dtype=' + self.gen_name(
                input_name) + '.dtype)' + self.gen_return()
            self.generated_apis += self.gen_indent(
            ) + 'helper.append_op(type=\'slice\', inputs={\'Input\': ' + self.gen_name(
                input_name
            ) + '}, outputs={\'Out\': ' + self.gen_name(
                out_name
            ) + '}, attrs={\'axes\': ' + axes + ', \'starts\': ' + starts + ', \'ends\': ' + ends + ', \'decrease_axis\': ' + decrease_axis + '})' + self.gen_return(
            )
        else:
            self.generated_apis += self.gen_indent(
            ) + self.gen_name(out_name) + ' = paddle.slice(' + self.gen_name(
                input_name
            ) + ', ' + axes + ', ' + starts + ', ' + ends + ')' + self.gen_return(
            )

    def gen_softmax(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        axis = str(op_desc.attr('axis'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = F.softmax(' + self.gen_name(
                x_name) + ', axis=' + axis + ')' + self.gen_return()

    def gen_softplus(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        beta = str(op_desc.attr('beta'))
        threshold = str(op_desc.attr('threshold'))
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ' = F.softplus(' + self.gen_name(
            x_name
        ) + ', beta=' + beta + ', threshold=' + threshold + ')' + self.gen_return(
        )

    def gen_split(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_params(x_name)
        out_names = op_desc.output('Out')
        axis = str(op_desc.attr('axis'))
        num = op_desc.attr('num')
        sections = op_desc.attr('sections')
        num_or_sections = python_array2string(sections) if num == 0 else str(
            num)
        self.generated_apis += self.gen_indent() + python_array2string(
            self.gen_names(out_names), False
        ) + ' = paddle.split(' + self.gen_name(
            x_name
        ) + ', num_or_sections=' + num_or_sections + ', axis=' + axis + ')' + self.gen_return(
        )

    def gen_squeeze(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        axis = op_desc.attr('axes')
        axis = python_array2string(axis) if axis else 'None'
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = paddle.squeeze(' + self.gen_name(
                x_name) + ', axis=' + axis + ')' + self.gen_return()

    def gen_stack(self, block_idx, op_desc):
        x_names = op_desc.input('X')
        self.gen_params(x_names)
        y_name = op_desc.output('Y')[0]
        axis = str(op_desc.attr('axis'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            y_name) + ' = paddle.stack(' + python_array2string(
                self.gen_names(x_names),
                False) + ', axis=' + axis + ')' + self.gen_return()

    def gen_strided_slice(self, block_idx, op_desc):
        input_name = op_desc.input('Input')[0]
        self.gen_param(input_name)
        out_name = op_desc.output('Out')[0]
        axes = python_array2string(op_desc.attr('axes'))
        starts = python_array2string(op_desc.attr('starts'))
        ends = python_array2string(op_desc.attr('ends'))
        strides = python_array2string(op_desc.attr('strides'))
        decrease_axis = op_desc.attr('decrease_axis')
        if decrease_axis:
            # No paddle api found for strided_slice op with attr 'decrease_axis', use LayerHelper directly.
            decrease_axis = python_array2string(op_desc.attr('decrease_axis'))
            self.generated_apis += self.gen_indent(
            ) + 'helper = LayerHelper(\'strided_slice\')' + self.gen_return()
            self.generated_apis += self.gen_indent() + self.gen_name(
                out_name
            ) + ' = helper.create_variable_for_type_inference(dtype=' + self.gen_name(
                input_name) + '.dtype)' + self.gen_return()
            self.generated_apis += self.gen_indent(
            ) + 'helper.append_op(type=\'strided_slice\', inputs={\'Input\': ' + self.gen_name(
                input_name
            ) + '}, outputs={\'Out\': ' + self.gen_name(
                out_name
            ) + '}, attrs={\'axes\': ' + axes + ', \'starts\': ' + starts + ', \'ends\': ' + ends + ', \'strides\': ' + strides + ', \'decrease_axis\': ' + decrease_axis + '})' + self.gen_return(
            )
        else:
            self.generated_apis += self.gen_indent() + self.gen_name(
                out_name
            ) + ' = paddle.strided_slice(' + self.gen_name(
                input_name
            ) + ', ' + axes + ', ' + starts + ', ' + ends + ', ' + strides + ')' + self.gen_return(
            )

    def gen_take_along_axis(self, block_idx, op_desc):
        input_name = op_desc.input('Input')[0]
        self.gen_param(input_name)
        index_name = op_desc.input('Index')[0]
        self.gen_param(index_name)
        result_name = op_desc.output('Result')[0]
        axis = str(op_desc.attr('Axis'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            result_name) + ' = paddle.take_along_axis(' + self.gen_name(
                input_name) + ', ' + self.gen_name(
                    index_name) + ', ' + axis + ')' + self.gen_return()

    def gen_tile(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        if 'RepeatTimes' in op_desc.input_names() and len(
                op_desc.input('RepeatTimes')) > 0:
            repeat_times_name = op_desc.input('RepeatTimes')[0]
            self.gen_param(repeat_times_name)
            repeat_times = self.gen_name(repeat_times_name)
        else:
            repeat_times = python_array2string(op_desc.attr('repeat_times'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = paddle.tile(' + self.gen_name(
                x_name) + ', ' + repeat_times + ')' + self.gen_return()

    def gen_transpose(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        perm = python_array2string(op_desc.attr('axis'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = paddle.transpose(' + self.gen_name(
                x_name) + ', perm=' + perm + ')' + self.gen_return()

    def gen_top_k(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        indices_name = op_desc.output('Indices')[0]
        axis = str(op_desc.attr('axis'))
        k = str(op_desc.attr('k'))
        largest = str(op_desc.attr('largest'))
        sorted = str(op_desc.attr('sorted'))
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ', ' + self.gen_name(
            indices_name
        ) + ' = paddle.topk(' + self.gen_name(
            x_name
        ) + ', ' + k + ', axis=' + axis + ', largest=' + largest + ', sorted=' + sorted + ')' + self.gen_return(
        )

    def gen_unary_ops(self, block_idx, op_desc):
        op_type = op_desc.type()
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = '
        if op_type == 'abs':
            self.generated_apis += 'paddle.abs(' + self.gen_name(x_name) + ')'
        elif op_type == 'acos':
            self.generated_apis += 'paddle.acos(' + self.gen_name(x_name) + ')'
        elif op_type == 'acosh':
            self.generated_apis += 'paddle.acosh(' + self.gen_name(
                x_name) + ')'
        elif op_type == 'asin':
            self.generated_apis += 'paddle.asin(' + self.gen_name(x_name) + ')'
        elif op_type == 'asinh':
            self.generated_apis += 'paddle.asinh(' + self.gen_name(
                x_name) + ')'
        elif op_type == 'atan':
            self.generated_apis += 'paddle.atan(' + self.gen_name(x_name) + ')'
        elif op_type == 'atanh':
            self.generated_apis += 'paddle.atanh(' + self.gen_name(
                x_name) + ')'
        elif op_type == 'ceil':
            self.generated_apis += 'paddle.ceil(' + self.gen_name(x_name) + ')'
        elif op_type == 'cos':
            self.generated_apis += 'paddle.cos(' + self.gen_name(x_name) + ')'
        elif op_type == 'cosh':
            self.generated_apis += 'paddle.cosh(' + self.gen_name(x_name) + ')'
        elif op_type == 'exp':
            self.generated_apis += 'paddle.exp(' + self.gen_name(x_name) + ')'
        elif op_type == 'expm1':
            self.generated_apis += 'paddle.expm1(' + self.gen_name(
                x_name) + ')'
        elif op_type == 'floor':
            self.generated_apis += 'paddle.floor(' + self.gen_name(
                x_name) + ')'
        elif op_type == 'reciprocal':
            self.generated_apis += 'paddle.reciprocal(' + self.gen_name(
                x_name) + ')'
        elif op_type == 'round':
            self.generated_apis += 'paddle.round(' + self.gen_name(
                x_name) + ')'
        elif op_type == 'rsqrt':
            self.generated_apis += 'paddle.rsqrt(' + self.gen_name(
                x_name) + ')'
        elif op_type == 'sigmoid':
            self.generated_apis += 'F.sigmoid(' + self.gen_name(x_name) + ')'
        elif op_type == 'sin':
            self.generated_apis += 'paddle.sin(' + self.gen_name(x_name) + ')'
        elif op_type == 'sinh':
            self.generated_apis += 'paddle.sinh(' + self.gen_name(x_name) + ')'
        elif op_type == 'sqrt':
            self.generated_apis += 'paddle.sqrt(' + self.gen_name(x_name) + ')'
        elif op_type == 'square':
            self.generated_apis += 'paddle.square(' + self.gen_name(
                x_name) + ')'
        elif op_type == 'tan':
            self.generated_apis += 'paddle.tan(' + self.gen_name(x_name) + ')'
        elif op_type == 'tanh':
            self.generated_apis += 'paddle.tanh(' + self.gen_name(x_name) + ')'
        elif op_type == 'erf':
            self.generated_apis += 'paddle.erf(' + self.gen_name(x_name) + ')'
        elif op_type == 'logical_not':
            self.generated_apis += 'paddle.logical_not(' + self.gen_name(
                x_name) + ')'
        elif op_type == 'bitwise_not':
            self.generated_apis += 'paddle.bitwise_not(' + self.gen_name(
                x_name) + ')'
        elif op_type == 'relu':
            self.generated_apis += 'F.relu(' + self.gen_name(x_name) + ')'
        else:
            raise ValueError('Unsupport to generate code for unary op \'%s\'' %
                             op_type)
        self.generated_apis += self.gen_return()

    def gen_unsqueeze(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        axis = python_array2string(op_desc.attr('axes'))
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = paddle.unsqueeze(' + self.gen_name(
                x_name) + ', axis=' + axis + ')' + self.gen_return()

    def gen_where(self, block_idx, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        y_name = op_desc.input('Y')[0]
        self.gen_param(y_name)
        condition_name = op_desc.input('Condition')[0]
        self.gen_param(condition_name)
        out_name = op_desc.output('Out')[0]
        self.generated_apis += self.gen_indent(
        ) + self.gen_name(out_name) + ' = paddle.where(' + self.gen_name(
            condition_name) + ', ' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')' + self.gen_return(
                )

    def gen_where_index(self, block_idx, op_desc):
        condition_name = op_desc.input('Condition')[0]
        self.gen_param(condition_name)
        out_name = op_desc.output('Out')[0]
        self.generated_apis += self.gen_indent() + self.gen_name(
            out_name) + ' = paddle.nonzero(' + self.gen_name(
                condition_name) + ')' + self.gen_return()

    def load_model(self, path_prefix):
        self.place = paddle.CPUPlace()
        self.exe = paddle.static.Executor(place=self.place)
        self.global_scope = paddle.static.global_scope()
        [self.program, self.feed_target_names, self.fetch_targets
         ] = paddle.static.load_inference_model(path_prefix, self.exe)
        print('--- feed_target_names ---')
        print(self.feed_target_names)
        print('--- fetch_targets ---')
        print(self.fetch_targets)

    def gen_code(self, code_dir, param_name='params', script_name='model.py'):
        self.init_indent_size = 1
        self.cur_indent_size = self.init_indent_size
        self.generated_code = "\
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.\n\
#\
# Licensed under the Apache License, Version 2.0 (the \"License\");\n\
# you may not use this file except in compliance with the License.\n\
# You may obtain a copy of the License at\n\
#\n\
#     http://www.apache.org/licenses/LICENSE-2.0\n\
#\n\
# Unless required by applicable law or agreed to in writing, software\n\
# distributed under the License is distributed on an \"AS IS\" BASIS,\n\
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n\
# See the License for the specific language governing permissions and\n\
# limitations under the License.\n\
import os\n\
import numpy as np\n\
import paddle\n\
import paddle.nn as nn\n\
import paddle.nn.functional as F\n\
from paddle.static.nn.control_flow import ConditionalBlock\n\
from paddle.framework import LayerHelper\n\
\n\
paddle.enable_static()\n\
\n\
def main(argv=None):\n\
    main_program = paddle.static.default_main_program()\n\
    global_scope = paddle.static.global_scope()\n\
    place = paddle.CPUPlace()\n\
"

        self.generated_params = []
        self.generated_vars = ''
        self.generated_apis = ''
        self.code_dir = code_dir
        try:
            os.makedirs(self.code_dir)
        except OSError as e:
            if e.errno != 17:
                raise
        self.param_name = param_name
        self.param_dir = code_dir + os.sep + param_name
        try:
            os.makedirs(self.param_dir)
        except OSError as e:
            if e.errno != 17:
                raise
        self.gen_head()
        self.gen_block(0)
        self.generated_code += self.gen_indent(
        ) + '# Initialize weights' + self.gen_return() + self.generated_vars
        self.generated_code += self.gen_indent(
        ) + '# Build network' + self.gen_return() + self.generated_apis
        self.gen_tail()
        with open(self.code_dir + os.sep + script_name, 'w') as f:
            f.write(self.generated_code)

    def __init__(self):
        self.gen_funcs = {
            'abs': self.gen_unary_ops,
            'acos': self.gen_unary_ops,
            'acosh': self.gen_unary_ops,
            'arg_max': self.gen_arg_max,
            'asin': self.gen_unary_ops,
            'asinh': self.gen_unary_ops,
            'assign_value': self.gen_assign_value,
            'atan': self.gen_unary_ops,
            'atanh': self.gen_unary_ops,
            'batch_norm': self.gen_batch_norm,
            'bilinear_interp_v2': self.gen_interp_ops,
            'bitwise_and': self.gen_binary_ops,
            'bitwise_not': self.gen_unary_ops,
            'bitwise_or': self.gen_binary_ops,
            'bitwise_xor': self.gen_binary_ops,
            'cast': self.gen_cast,
            'ceil': self.gen_unary_ops,
            'clip': self.gen_clip,
            'concat': self.gen_concat,
            'conditional_block': self.gen_conditional_block,
            'conv2d': self.gen_conv2d,
            'conv2d_transpose': self.gen_conv2d_transpose,
            'cos': self.gen_unary_ops,
            'cosh': self.gen_unary_ops,
            'cumsum': self.gen_cumsum,
            'depthwise_conv2d': self.gen_conv2d,
            'depthwise_conv2d_transpose': self.gen_conv2d_transpose,
            'dropout': self.gen_dropout,
            'elementwise_add': self.gen_elementwise_ops,
            'elementwise_div': self.gen_elementwise_ops,
            'elementwise_mul': self.gen_elementwise_ops,
            'elementwise_pow': self.gen_elementwise_pow,
            'elementwise_sub': self.gen_elementwise_ops,
            'elu': self.gen_elu,
            'equal': self.gen_binary_ops,
            'erf': self.gen_binary_ops,
            'exp': self.gen_unary_ops,
            'expand_v2': self.gen_expand,
            'expand_as_v2': self.gen_expand_as,
            'expm1': self.gen_unary_ops,
            'fill_any_like': self.gen_fill_any_like,
            'fill_constant': self.gen_fill_constant,
            'flip': self.gen_flip,
            'floor': self.gen_unary_ops,
            'gather': self.gen_gather,
            'gather_nd': self.gen_gather_nd,
            'greater_equal': self.gen_binary_ops,
            'greater_than': self.gen_binary_ops,
            'hard_sigmoid': self.gen_hard_sigmoid,
            'index_select': self.gen_index_select,
            'instance_norm': self.gen_instance_norm,
            'layer_norm': self.gen_layer_norm,
            'leaky_relu': self.gen_leaky_relu,
            'less_equal': self.gen_binary_ops,
            'less_than': self.gen_binary_ops,
            'logical_and': self.gen_binary_ops,
            'logical_not': self.gen_unary_ops,
            'logical_or': self.gen_binary_ops,
            'logical_xor': self.gen_binary_ops,
            'linspace': self.gen_linspace,
            'lookup_table_v2': self.gen_lookup_table,
            'matmul_v2': self.gen_matmul,
            'max_pool2d_with_index': self.gen_pool2d,
            'meshgrid': self.gen_meshgrid,
            'elementwise_max': self.gen_elementwise_ops,
            'elementwise_min': self.gen_elementwise_ops,
            'nearest_interp_v2': self.gen_interp_ops,
            'not_equal': self.gen_binary_ops,
            'p_norm': self.gen_p_norm,
            'pad': self.gen_pad,
            'pad3d': self.gen_pad3d,
            'pool2d': self.gen_pool2d,
            'pow': self.gen_elementwise_pow,
            'shape': self.gen_shape,
            'range': self.gen_range,
            'reciprocal': self.gen_unary_ops,
            'reduce_all': self.gen_reduce_ops,
            'reduce_any': self.gen_reduce_ops,
            'reduce_max': self.gen_reduce_ops,
            'reduce_mean': self.gen_reduce_ops,
            'reduce_min': self.gen_reduce_ops,
            'reduce_prod': self.gen_reduce_ops,
            'reduce_sum': self.gen_reduce_ops,
            'relu': self.gen_unary_ops,
            'relu6': self.gen_relu6,
            'reshape2': self.gen_reshape,
            'rnn': self.gen_rnn,
            'round': self.gen_unary_ops,
            'rsqrt': self.gen_unary_ops,
            'scale': self.gen_scale,
            'scatter_nd_add': self.gen_scatter_nd_add,
            'split': self.gen_split,
            'set_value': self.gen_set_value,
            'share_data': self.gen_share_data,
            'sigmoid': self.gen_unary_ops,
            'sin': self.gen_unary_ops,
            'sinh': self.gen_unary_ops,
            'slice': self.gen_slice,
            'softmax': self.gen_softmax,
            'softplus': self.gen_softplus,
            'sqrt': self.gen_unary_ops,
            'square': self.gen_unary_ops,
            'squeeze2': self.gen_squeeze,
            'stack': self.gen_stack,
            'strided_slice': self.gen_strided_slice,
            'tan': self.gen_unary_ops,
            'tanh': self.gen_unary_ops,
            'top_k_v2': self.gen_top_k,
            'take_along_axis': self.gen_take_along_axis,
            'tile': self.gen_tile,
            'transpose2': self.gen_transpose,
            'unsqueeze2': self.gen_unsqueeze,
            'where': self.gen_where,
            'where_index': self.gen_where_index
        }
        self.init_indent_size = 1
        self.cur_indent_size = 1
        self.generated_code = ""
        self.generated_params = []
        self.generated_vars = ''
        self.generated_apis = ''


def main(argv=None):
    code_generator = CodeGenerator()
    code_generator.load_model(args.model_path_prefix)
    code_generator.gen_code(args.code_dir)
    print("Done.")


if __name__ == '__main__':
    main()
