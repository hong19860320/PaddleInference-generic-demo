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


class CodeGenerator:
    def gen_name(self, name):
        syms = ['.']
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
        self.generated_code += '    ' * self.indent_size

    def gen_return(self):
        self.generated_code += '\n'

    def gen_head(self):
        for feed_target_name in self.feed_target_names:
            feed_target_var = self.program.global_block().var(feed_target_name)
            feed_target_shape = python_array2string(feed_target_var.shape)
            feed_target_dtype = paddle_dtype2string(feed_target_var.dtype)
            self.gen_indent()
            self.generated_code += self.gen_name(
                feed_target_name
            ) + ' = paddle.static.data(name=\'' + self.gen_name(
                feed_target_name
            ) + '\', shape=' + feed_target_shape + ', dtype=\'' + feed_target_dtype + '\')'
            self.gen_return()

    def gen_tail(self):
        #self.gen_indent()
        #self.generated_code += '# Restore fetch target names'
        #self.gen_return()
        #self.gen_indent()
        #self.generated_code += 'for block in main_program.blocks:'
        #self.gen_return()
        #self.indent_size += 1
        #self.gen_indent()
        #self.generated_code += 'for op in block.ops:'
        #self.gen_return()
        #self.indent_size += 1
        #for fetch_target_idx in range(len(self.fetch_targets)):
        #    self.gen_indent()
        #    self.generated_code += 'op._rename_output(' + self.gen_name(
        #        self.fetch_targets[fetch_target_idx].name
        #    ) + '.name, \'' + self.fetch_targets[fetch_target_idx].name + '\')'
        #    self.gen_return()
        #self.indent_size -= 2
        #for fetch_target_idx in range(len(self.fetch_targets)):
        #    self.gen_indent()
        #    self.generated_code += self.gen_name(self.fetch_targets[
        #        fetch_target_idx].name) + '.name = \'' + self.fetch_targets[
        #            fetch_target_idx].name + '\''
        #    self.gen_return()
        self.gen_indent()
        self.generated_code += '# Compile and output an inference model'
        self.gen_return()
        self.gen_indent()
        self.generated_code += 'exe = paddle.static.Executor(place)'
        self.gen_return()
        self.gen_indent()
        self.generated_code += 'startup_program = paddle.static.default_startup_program()'
        self.gen_return()
        self.gen_indent()
        self.generated_code += 'exe.run(startup_program)'
        self.gen_return()
        fetch_target_names = self.gen_name(self.fetch_targets[0].name)
        for fetch_target_idx in range(1, len(self.fetch_targets)):
            fetch_target_names += ',' + self.gen_name(self.fetch_targets[
                fetch_target_idx].name)
        self.gen_indent()
        self.generated_code += 'paddle.static.save_inference_model(\'./model\', ' + python_array2string(
            self.feed_target_names,
            False) + ', [' + fetch_target_names + '], exe)'
        self.gen_return()
        self.gen_indent()
        self.generated_code += '# Prepare the input data, reload and run the inference model'
        self.gen_return()
        self.gen_indent()
        self.generated_code += '# [inference_program, feed_target_names, fetch_targets] = paddle.static.load_inference_model(\'./model\', exe)'
        self.gen_return()
        for feed_target_name in self.feed_target_names:
            feed_target_var = self.program.global_block().var(feed_target_name)
            feed_target_shape = python_array2string(
                [1 if i == -1 else i for i in feed_target_var.shape])
            feed_target_dtype = paddle_dtype2string(feed_target_var.dtype)
            self.gen_indent()
            self.generated_code += '# ' + self.gen_name(
                feed_target_name
            ) + '_tensor = np.zeros(shape=' + feed_target_shape + ', dtype=\'' + feed_target_dtype + '\')'
            self.gen_return()
        feed_target_dict = '\'' + self.feed_target_names[
            0] + '\': ' + self.gen_name(self.feed_target_names[0]) + '_tensor'
        for feed_target_idx in range(1, len(self.feed_target_names)):
            feed_target_dict += ', \'' + self.feed_target_names[
                feed_target_idx] + '\': ' + self.gen_name(
                    self.feed_target_names[feed_target_idx]) + '_tensor'
        self.gen_indent()
        self.generated_code += '# output_tensors = exe.run(inference_program, feed={' + feed_target_dict + '}, fetch_list=fetch_targets, return_numpy=True)'
        self.gen_return()
        self.gen_indent()
        self.generated_code += '# print(output_tensors)'
        self.gen_return()
        self.gen_indent()
        self.generated_code += 'print(\'Done.\')'
        self.gen_return()
        self.gen_return()
        self.indent_size -= 1
        self.gen_indent()
        self.generated_code += 'if __name__ == \'__main__\':'
        self.gen_return()
        self.indent_size += 1
        self.gen_indent()
        self.generated_code += 'main()'
        self.gen_return()

    def gen_param(self, name):
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
        self.gen_indent()
        self.generated_code += self.gen_name(
            name
        ) + ' = main_program.global_block().create_var(name=\'' + name + '\', shape=' + shape + ', dtype=\'' + dtype + '\', persistable=True)'
        self.gen_return()
        self.gen_indent()
        self.generated_code += 'global_scope.var(\'' + name + '\').get_tensor().set(np.load(\'./' + self.param_name + os.sep + name + '.npy\'), place)'
        self.gen_return()
        return param

    def gen_params(self, names):
        params = []
        for name in names:
            params.append(self.gen_param(name))
        return params

    def gen_arg_max(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        axis = op_desc.attr('axis')
        axis = str(axis) if axis else 'None'
        keepdim = str(op_desc.attr('keepdims'))
        dtype = paddle_dtype2string(op_desc.attr('dtype'))
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name
        ) + ' = paddle.argmax(' + self.gen_name(
            x_name
        ) + ', axis=' + axis + ', keepdim=' + keepdim + ', dtype=\'' + dtype + '\')'
        self.gen_return()

    def gen_assign_value(self, op_desc):
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
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name
        ) + ' = paddle.assign(np.array(' + values + ', \'' + dtype + '\').reshape(' + shape + '))'
        self.gen_return()

    def gen_batch_norm(self, op_desc):
        op_type = op_desc.type()
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
        self.gen_indent()
        self.generated_code += self.gen_name(
            y_name
        ) + ' = F.batch_norm(' + self.gen_name(x_name) + ', ' + self.gen_name(
            mean_name
        ) + ', ' + self.gen_name(variance_name) + ', ' + self.gen_name(
            scale_name
        ) + ', ' + self.gen_name(
            bias_name
        ) + ', False, ' + momentum + ', ' + epsilon + ', \'' + data_layout + '\')'
        self.gen_return()

    def gen_binary_ops(self, op_desc):
        op_type = op_desc.type()
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        y_name = op_desc.input('Y')[0]
        self.gen_param(y_name)
        out_name = op_desc.output('Out')[0]
        self.gen_indent()
        self.generated_code += self.gen_name(out_name) + ' = '
        if op_type == 'equal':
            self.generated_code += 'paddle.equal(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'not_equal':
            self.generated_code += 'paddle.not_equal(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'less_than':
            self.generated_code += 'paddle.less_than(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'greater_than':
            self.generated_code += 'paddle.greater_than(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'greater_equal':
            self.generated_code += 'paddle.greater_equal(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'less_equal':
            self.generated_code += 'paddle.less_equal(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'bitwise_and':
            self.generated_code += 'paddle.bitwise_and(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'bitwise_or':
            self.generated_code += 'paddle.bitwise_or(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        elif op_type == 'bitwise_xor':
            self.generated_code += 'paddle.bitwise_xor(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ')'
        else:
            raise ValueError(
                'Unsupport to generate code for binary op \'%s\'' % op_type)
        self.gen_return()

    def gen_block(self, block_idx):
        num_blocks = self.program.num_blocks
        block = self.program.block(block_idx)
        num_ops = len(block.ops)
        indent_inc = 0 if block_idx == 0 else 1
        self.indent_size += indent_inc
        for op_idx in range(num_ops):
            op_desc = block.ops[op_idx].desc
            op_type = op_desc.type()
            print('[%d/%d %d/%d] generating %s ...' %
                  (op_idx + 1, num_ops, block_idx + 1, num_blocks, op_type))
            if op_type == 'feed' or op_type == 'fetch':
                continue
            try:
                self.gen_funcs[op_type](op_desc)
            except KeyError:
                raise ValueError('Unsupport to generate code for op \'%s\'' %
                                 op_type)
        self.indent_size -= indent_inc

    def gen_cast(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        in_dtype = paddle_dtype2string(op_desc.attr('in_dtype'))
        out_dtype = paddle_dtype2string(op_desc.attr('out_dtype'))
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name) + ' = paddle.cast(' + self.gen_name(
                x_name) + ', \'' + out_dtype + '\')'
        self.gen_return()

    def gen_concat(self, op_desc):
        x_names = op_desc.input('X')
        self.gen_params(x_names)
        out_name = op_desc.output('Out')[0]
        axis = str(op_desc.attr('axis'))
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name) + ' = paddle.concat(' + python_array2string(
                self.gen_names(x_names), False) + ', axis=' + axis + ')'
        self.gen_return()

    def gen_conditional_block(self, op_desc):
        cond_names = op_desc.input('Cond')
        self.gen_params(cond_names)
        is_scalar_condition = str(op_desc.attr('is_scalar_condition'))
        sub_block_idx = op_desc.attr('sub_block').id
        self.gen_indent()
        self.generated_code += 'condition_block_' + str(
            sub_block_idx
        ) + ' = ConditionalBlock(inputs=' + python_array2string(
            self.gen_names(cond_names),
            False) + ', is_scalar_condition=' + is_scalar_condition + ')'
        self.gen_return()
        self.gen_indent()
        self.generated_code += 'with condition_block_' + str(
            sub_block_idx) + '.block():'
        self.gen_return()
        self.gen_block(sub_block_idx)

    def gen_conv2d(self, op_desc):
        input_name = op_desc.input('Input')[0]
        self.gen_param(input_name)
        filter_name = op_desc.input('Filter')[0]
        self.gen_param(filter_name)
        output_name = op_desc.output('Output')[0]
        strides = python_array2string(op_desc.attr('strides'))
        paddings = python_array2string(op_desc.attr('paddings'))
        dilations = python_array2string(op_desc.attr('dilations'))
        groups = str(op_desc.attr('groups'))
        self.gen_indent()
        self.generated_code += self.gen_name(
            output_name
        ) + ' = F.conv2d(' + self.gen_name(input_name) + ', ' + self.gen_name(
            filter_name
        ) + ', bias=None, stride=' + strides + ', padding=' + paddings + ', dilation=' + dilations + ', groups=' + groups + ', data_format=\'NCHW\')'
        self.gen_return()

    def gen_conv2d_transpose(self, op_desc):
        input_name = op_desc.input('Input')[0]
        self.gen_param(input_name)
        filter_name = op_desc.input('Filter')[0]
        self.gen_param(filter_name)
        output_name = op_desc.output('Output')[0]
        strides = python_array2string(op_desc.attr('strides'))
        paddings = python_array2string(op_desc.attr('paddings'))
        output_padding = op_desc.attr('output_padding')
        output_padding = python_array2string(
            output_padding) if output_padding else '0'
        dilations = python_array2string(op_desc.attr('dilations'))
        groups = str(op_desc.attr('groups'))
        output_size = op_desc.attr('output_size')
        output_size = python_array2string(
            output_size) if output_size else 'None'
        self.gen_indent()
        self.generated_code += self.gen_name(
            output_name
        ) + ' = F.conv2d_transpose(' + self.gen_name(
            input_name
        ) + ', ' + self.gen_name(
            filter_name
        ) + ', bias=None, stride=' + strides + ', padding=' + paddings + ', output_padding=' + output_padding + ', dilation=' + dilations + ', groups=' + groups + ', data_format=\'NCHW\', output_size=' + output_size + ')'
        self.gen_return()

    def gen_elementwise_ops(self, op_desc):
        op_type = op_desc.type()
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        y_name = op_desc.input('Y')[0]
        self.gen_param(y_name)
        out_name = op_desc.output('Out')[0]
        axis = op_desc.attr('axis')
        axis = str(axis) if axis else '-1'
        self.gen_indent()
        self.generated_code += self.gen_name(out_name) + ' = '
        if op_type == 'elementwise_add':
            self.generated_code += 'paddle.tensor.math._add_with_axis(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ', axis=' + axis + ')'
        elif op_type == 'elementwise_sub':
            self.generated_code += 'paddle.tensor.math._subtract_with_axis(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ', axis=' + axis + ')'
        elif op_type == 'elementwise_mul':
            self.generated_code += 'paddle.tensor.math._multiply_with_axis(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ', axis=' + axis + ')'
        elif op_type == 'elementwise_div':
            self.generated_code += 'paddle.tensor.math._divide_with_axis(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ', axis=' + axis + ')'
        elif op_type == 'elementwise_pow':
            self.generated_code += 'paddle.tensor.math._divide_with_axis(' + self.gen_name(
                x_name) + ', ' + self.gen_name(y_name) + ', axis=' + axis + ')'
        else:
            raise ValueError(
                'Unsupport to generate code for binary op \'%s\'' % op_type)
        self.gen_return()

    def gen_expand(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        shape = python_array2string(op_desc.attr('shape'))
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name) + ' = paddle.expand(' + self.gen_name(
                x_name) + ', ' + shape + ')'
        self.gen_return()

    def gen_fill_constant(self, op_desc):
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
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name
        ) + ' = paddle.full(' + shape + ', ' + fill_value + ', dtype=\'' + dtype + '\')'
        self.gen_return()

    def gen_gather_nd(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        index_name = op_desc.input('Index')[0]
        self.gen_param(index_name)
        out_name = op_desc.output('Out')[0]
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name) + ' = paddle.gather_nd(' + self.gen_name(
                x_name) + ', index=' + self.gen_name(index_name) + ')'
        self.gen_return()

    def gen_index_select(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        index_name = op_desc.input('Index')[0]
        self.gen_param(index_name)
        out_name = op_desc.output('Out')[0]
        axis = str(op_desc.attr('dim'))
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name) + ' = paddle.index_select(' + self.gen_name(
                x_name) + ', ' + self.gen_name(
                    index_name) + ', axis=' + axis + ')'
        self.gen_return()

    def gen_interp_ops(self, op_desc):
        op_type = op_desc.type()
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
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name
        ) + ' = F.interpolate(' + self.gen_name(
            x_name
        ) + ', size=' + size + ', scale_factor=' + scale_factor + ', mode=\'' + mode + '\', align_corners=' + align_corners + ', align_mode=' + align_mode + ', data_format=\'NCHW\')'
        self.gen_return()

    def gen_matmul(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        y_name = op_desc.input('Y')[0]
        self.gen_param(y_name)
        out_name = op_desc.output('Out')[0]
        transpose_x = str(op_desc.attr('trans_x'))
        transpose_y = str(op_desc.attr('trans_y'))
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name
        ) + ' = paddle.matmul(' + self.gen_name(x_name) + ', ' + self.gen_name(
            y_name
        ) + ', transpose_x=' + transpose_x + ', transpose_y=' + transpose_y + ')'
        self.gen_return()

    def gen_pool2d(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        adaptive = op_desc.attr('adaptive')
        pooling_type = op_desc.attr('pooling_type')
        kernel_size = python_array2string(op_desc.attr('ksize'))
        self.gen_indent()
        self.generated_code += self.gen_name(out_name) + ' = '
        if adaptive:
            if pooling_type == 'avg':
                self.generated_code += 'F.adaptive_avg_pool2d(' + self.gen_name(
                    x_name) + ', ' + kernel_size + ', data_format=\'NCHW\')'
            elif pooling_type == 'max':
                self.generated_code += 'F.adaptive_max_pool2d(' + self.gen_name(
                    x_name
                ) + ', ' + kernel_size + ', return_mask=False, data_format=\'NCHW\')'
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
                self.generated_code += 'F.avg_pool2d(' + self.gen_name(
                    x_name
                ) + ', ' + kernel_size + ', stride=' + stride + ', padding=' + padding + ', ceil_mode=' + ceil_mode + ', exclusive=' + exclusive + ', divisor_override=None, data_format=\'NCHW\')'
            elif pooling_type == 'max':
                self.generated_code += 'F.max_pool2d(' + self.gen_name(
                    x_name
                ) + ', ' + kernel_size + ', stride=' + stride + ', padding=' + padding + ', ceil_mode=' + ceil_mode + ', return_mask=False, data_format=\'NCHW\')'
            else:
                raise ValueError(
                    'Unsupport to generate code for pool2d op \'%s\'' %
                    op_type)
        self.gen_return()

    def gen_shape(self, op_desc):
        input_name = op_desc.input('Input')[0]
        self.gen_param(input_name)
        out_name = op_desc.output('Out')[0]
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name) + ' = paddle.shape(' + self.gen_name(input_name) + ')'
        self.gen_return()

    def gen_range(self, op_desc):
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
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name) + ' = paddle.arange(start=' + self.gen_name(
                start_name) + ', end=' + self.gen_name(
                    end_name) + ', step=' + self.gen_name(
                        step_name) + ', dtype=None)'
        self.gen_return()

    def gen_reduce_ops(self, op_desc):
        op_type = op_desc.type()
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        axis = op_desc.attr('dim')
        axis = python_array2string(axis) if axis else 'None'
        keepdim = str(op_desc.attr('keep_dim'))
        self.gen_indent()
        self.generated_code += self.gen_name(out_name) + ' = '
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
        self.generated_code += 'paddle.' + api + '(' + self.gen_name(
            x_name) + ', axis=' + axis + ', keepdim=' + keepdim + ')'
        self.gen_return()

    def gen_relu(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name) + ' = F.relu(' + self.gen_name(x_name) + ')'
        self.gen_return()

    def gen_sigmoid(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name) + ' = F.sigmoid(' + self.gen_name(x_name) + ')'
        self.gen_return()

    def gen_reshape(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        shape = python_array2string(op_desc.attr('shape'))
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name) + ' = paddle.reshape(' + self.gen_name(
                x_name) + ', shape=' + shape + ')'
        self.gen_return()

    def gen_scale(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        scale = str(op_desc.attr('scale'))
        bias = str(op_desc.attr('bias'))
        bias_after_scale = str(op_desc.attr('bias_after_scale'))
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name
        ) + ' = paddle.scale(' + self.gen_name(
            x_name
        ) + ', scale=' + scale + ', bias=' + bias + ', bias_after_scale=' + bias_after_scale + ', act=None)'
        self.gen_return()

    def gen_scatter_nd_add(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        index_name = op_desc.input('Index')[0]
        self.gen_param(index_name)
        updates_name = op_desc.input('Updates')[0]
        self.gen_param(updates_name)
        out_name = op_desc.output('Out')[0]
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name) + ' = paddle.scatter_nd_add(' + self.gen_name(
                x_name) + ', ' + self.gen_name(
                    index_name) + ', ' + self.gen_name(updates_name) + ')'
        self.gen_return()

    def gen_split(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_params(x_name)
        out_names = op_desc.output('Out')
        axis = str(op_desc.attr('axis'))
        num = op_desc.attr('num')
        sections = op_desc.attr('sections')
        num_or_sections = python_array2string(sections) if num == 0 else str(
            num)
        self.gen_indent()
        self.generated_code += python_array2string(
            self.gen_names(out_names),
            False) + ' = paddle.split(' + self.gen_name(
                x_name
            ) + ', num_or_sections=' + num_or_sections + ', axis=' + axis + ')'
        self.gen_return()

    def gen_set_value(self, op_desc):
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
        self.gen_indent()
        self.generated_code += 'helper = LayerHelper(\'set_value\')'
        self.gen_return()
        if out_name != input_name:
            self.gen_indent()
            self.generated_code += self.gen_name(
                out_name
            ) + ' = helper.create_variable_for_type_inference(dtype=' + self.gen_name(
                input_name) + '.dtype)'
            self.gen_return()
        self.gen_indent()
        self.generated_code += 'helper.append_op(type=\'set_value\', inputs={\'Input\': ' + self.gen_name(
            input_name
        ) + ', \'ValueTensor\': ' + self.gen_name(
            value_tensor_name
        ) + '}, outputs={\'Out\': ' + self.gen_name(
            out_name
        ) + '}, attrs={\'axes\': ' + axes + ', \'starts\': ' + starts + ', \'ends\': ' + ends + ', \'steps\': ' + steps + ', \'decrease_axes\': ' + decrease_axes + ', \'none_axes\': ' + none_axes + ', \'dtype\': ' + dtype + ', \'shape\': ' + shape + '}, inplace_map={"Input": "Out"})'
        self.gen_return()

    def gen_slice(self, op_desc):
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
            self.gen_indent()
            self.generated_code += 'helper = LayerHelper(\'slice\')'
            self.gen_return()
            self.gen_indent()
            self.generated_code += self.gen_name(
                out_name
            ) + ' = helper.create_variable_for_type_inference(dtype=' + self.gen_name(
                input_name) + '.dtype)'
            self.gen_return()
            self.gen_indent()
            self.generated_code += 'helper.append_op(type=\'slice\', inputs={\'Input\': ' + self.gen_name(
                input_name
            ) + '}, outputs={\'Out\': ' + self.gen_name(
                out_name
            ) + '}, attrs={\'axes\': ' + axes + ', \'starts\': ' + starts + ', \'ends\': ' + ends + ', \'decrease_axis\': ' + decrease_axis + '})'
            self.gen_return()
        else:
            self.gen_indent()
            self.generated_code += self.gen_name(
                out_name) + ' = paddle.slice(' + self.gen_name(
                    input_name
                ) + ', ' + axes + ', ' + starts + ', ' + ends + ')'
            self.gen_return()

    def gen_softmax(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        axis = str(op_desc.attr('axis'))
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name) + ' = F.softmax(' + self.gen_name(
                x_name) + ', axis=' + axis + ')'
        self.gen_return()

    def gen_squeeze(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        axis = op_desc.attr('axes')
        axis = python_array2string(axis) if axis else 'None'
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name) + ' = paddle.squeeze(' + self.gen_name(
                x_name) + ', axis=' + axis + ')'
        self.gen_return()

    def gen_stack(self, op_desc):
        x_names = op_desc.input('X')
        self.gen_params(x_names)
        y_name = op_desc.output('Y')[0]
        axis = str(op_desc.attr('axis'))
        self.gen_indent()
        self.generated_code += self.gen_name(
            y_name) + ' = paddle.stack(' + python_array2string(
                self.gen_names(x_names), False) + ', axis=' + axis + ')'
        self.gen_return()

    def gen_transpose(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        perm = python_array2string(op_desc.attr('axis'))
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name) + ' = paddle.transpose(' + self.gen_name(
                x_name) + ', perm=' + perm + ')'
        self.gen_return()

    def gen_top_k(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        indices_name = op_desc.output('Indices')[0]
        axis = str(op_desc.attr('axis'))
        k = str(op_desc.attr('k'))
        largest = str(op_desc.attr('largest'))
        sorted = str(op_desc.attr('sorted'))
        self.gen_indent()
        self.generated_code += self.gen_name(out_name) + ', ' + self.gen_name(
            indices_name
        ) + ' = paddle.topk(' + self.gen_name(
            x_name
        ) + ', ' + k + ', axis=' + axis + ', largest=' + largest + ', sorted=' + sorted + ')'
        self.gen_return()

    def gen_unary_ops(self, op_desc):
        op_type = op_desc.type()
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        self.gen_indent()
        self.generated_code += self.gen_name(out_name) + ' = '
        if op_type == 'logical_not':
            self.generated_code += 'paddle.logical_not(' + self.gen_name(
                x_name) + ')'
        elif op_type == 'bitwise_not':
            self.generated_code += 'paddle.bitwise_not(' + self.gen_name(
                x_name) + ')'
        else:
            raise ValueError('Unsupport to generate code for unary op \'%s\'' %
                             op_type)
        self.gen_return()

    def gen_unsqueeze(self, op_desc):
        x_name = op_desc.input('X')[0]
        self.gen_param(x_name)
        out_name = op_desc.output('Out')[0]
        axis = python_array2string(op_desc.attr('axes'))
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name) + ' = paddle.unsqueeze(' + self.gen_name(
                x_name) + ', axis=' + axis + ')'
        self.gen_return()

    def gen_where_index(self, op_desc):
        condition_name = op_desc.input('Condition')[0]
        self.gen_param(condition_name)
        out_name = op_desc.output('Out')[0]
        self.gen_indent()
        self.generated_code += self.gen_name(
            out_name) + ' = paddle.nonzero(' + self.gen_name(
                condition_name) + ')'
        self.gen_return()

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
        self.indent_size = 1
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
    # Build network\n\
"

        self.generated_params = []
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
        self.gen_tail()
        with open(self.code_dir + os.sep + script_name, 'w') as f:
            f.write(self.generated_code)

    def __init__(self):
        self.gen_funcs = {
            'arg_max': self.gen_arg_max,
            'assign_value': self.gen_assign_value,
            'batch_norm': self.gen_batch_norm,
            'bilinear_interp_v2': self.gen_interp_ops,
            'bitwise_and': self.gen_binary_ops,
            'bitwise_not': self.gen_unary_ops,
            'bitwise_or': self.gen_binary_ops,
            'bitwise_xor': self.gen_binary_ops,
            'cast': self.gen_cast,
            'concat': self.gen_concat,
            'conditional_block': self.gen_conditional_block,
            'conv2d': self.gen_conv2d,
            'conv2d_transpose': self.gen_conv2d_transpose,
            'elementwise_add': self.gen_elementwise_ops,
            'elementwise_div': self.gen_elementwise_ops,
            'elementwise_mul': self.gen_elementwise_ops,
            'elementwise_sub': self.gen_elementwise_ops,
            'elementwise_pow': self.gen_elementwise_ops,
            'equal': self.gen_binary_ops,
            'expand_v2': self.gen_expand,
            'fill_constant': self.gen_fill_constant,
            'gather_nd': self.gen_gather_nd,
            'greater_equal': self.gen_binary_ops,
            'greater_than': self.gen_binary_ops,
            'index_select': self.gen_index_select,
            'less_equal': self.gen_binary_ops,
            'less_than': self.gen_binary_ops,
            'logical_not': self.gen_unary_ops,
            'matmul_v2': self.gen_matmul,
            'nearest_interp_v2': self.gen_interp_ops,
            'not_equal': self.gen_binary_ops,
            'pool2d': self.gen_pool2d,
            'shape': self.gen_shape,
            'range': self.gen_range,
            'reduce_all': self.gen_reduce_ops,
            'reduce_any': self.gen_reduce_ops,
            'reduce_max': self.gen_reduce_ops,
            'reduce_mean': self.gen_reduce_ops,
            'reduce_min': self.gen_reduce_ops,
            'reduce_prod': self.gen_reduce_ops,
            'reduce_sum': self.gen_reduce_ops,
            'relu': self.gen_relu,
            'sigmoid': self.gen_sigmoid,
            'reshape2': self.gen_reshape,
            'scale': self.gen_scale,
            'scatter_nd_add': self.gen_scatter_nd_add,
            'split': self.gen_split,
            'set_value': self.gen_set_value,
            'slice': self.gen_slice,
            'softmax': self.gen_softmax,
            'squeeze2': self.gen_squeeze,
            'stack': self.gen_stack,
            'top_k_v2': self.gen_top_k,
            'transpose2': self.gen_transpose,
            'unsqueeze2': self.gen_unsqueeze,
            'where_index': self.gen_where_index
        }
        self.indent_size = 1
        self.generated_code = ""
        self.generated_params = []


def main(argv=None):
    code_generator = CodeGenerator()
    code_generator.load_model('./simple_model/paddle/deploy3d')
    code_generator.gen_code('./output_code/')
    print("Done.")


if __name__ == '__main__':
    main()
