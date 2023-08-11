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


def python_array2string(python_array, separator=',', prefix='', suffix=''):
    if type(python_array) is tuple:
        python_array = np.array(python_array)
    if type(python_array) is list:
        python_array = np.array(python_array)
    if type(python_array) is np.ndarray:
        return np.array2string(
            python_array, separator=separator, prefix='', suffix='')
    else:
        raise ValueError("Unsupport to convert python array to string")


class CodeGenerator:
    def __init__(self):
        self.indent_size = 1
        self.generated_code = "\
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.\n\
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
import paddle.fluid as fluid\n\
from paddle.fluid import core\n\
\n\
paddle.enable_static()\n\
\n\
def main(argv=None):\n\
    # Build network\n\
"

    def gen_name(self, name):
        illegal_symbols = ['.']
        for symbol in illegal_symbols:
            if symbol in name:
                name = name.replace(symbol, "_")
        return name

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
        self.gen_indent()
        self.generated_code += '# Restore fetch target names'
        self.gen_return()
        self.gen_indent()
        self.generated_code += 'main_program = paddle.static.default_main_program()'
        self.gen_return()
        self.gen_indent()
        self.generated_code += 'for block in main_program.blocks:'
        self.gen_return()
        self.indent_size += 1
        self.gen_indent()
        self.generated_code += 'for op in block.ops:'
        self.gen_return()
        self.indent_size += 1
        for fetch_target_idx in range(len(self.fetch_targets)):
            self.gen_indent()
            self.generated_code += 'op._rename_output(' + self.gen_name(
                self.fetch_targets[fetch_target_idx].name
            ) + '.name, \'' + self.fetch_targets[fetch_target_idx].name + '\')'
            self.gen_return()
        self.indent_size -= 2
        for fetch_target_idx in range(len(self.fetch_targets)):
            self.gen_indent()
            self.generated_code += self.gen_name(self.fetch_targets[
                fetch_target_idx].name) + '.name = \'' + self.fetch_targets[
                    fetch_target_idx].name + '\''
            self.gen_return()
        self.gen_indent()
        self.generated_code += '# Compile the network and output an inference model'
        self.gen_return()
        self.gen_indent()
        self.generated_code += 'place = paddle.CPUPlace()'
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
        self.generated_code += 'fluid.io.save_inference_model(\'./\', ' + python_array2string(
            self.feed_target_names
        ) + ', [' + fetch_target_names + '], exe, model_filename=\'model.pdmodel\', params_filename=\'model.pdiparams\')'
        self.gen_return()
        self.gen_indent()
        self.generated_code += '# Prepare the input data, reload and run the inference model'
        self.gen_return()
        self.gen_indent()
        self.generated_code += '[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(\'./\', exe, model_filename=\'model.pdmodel\', params_filename=\'model.pdiparams\')'
        self.gen_return()
        for feed_target_name in self.feed_target_names:
            feed_target_var = self.program.global_block().var(feed_target_name)
            feed_target_shape = python_array2string(
                [1 if i == -1 else i for i in feed_target_var.shape])
            feed_target_dtype = paddle_dtype2string(feed_target_var.dtype)
            self.gen_indent()
            self.generated_code += '# ' + self.gen_name(
                feed_target_name
            ) + '_data = np.zeros(shape=' + feed_target_shape + ', dtype=\'' + feed_target_dtype + '\')'
            self.gen_return()
        feed_target_dict = '\'' + self.feed_target_names[
            0] + '\': ' + self.gen_name(self.feed_target_names[0]) + '_data'
        for feed_target_idx in range(1, len(self.feed_target_names)):
            feed_target_dict += ', \'' + self.feed_target_names[
                feed_target_idx] + '\': ' + self.gen_name(
                    self.feed_target_names[feed_target_idx]) + '_data'
        self.gen_indent()
        self.generated_code += '# outputs = exe.run(inference_program, feed={' + feed_target_dict + '}, fetch_list=fetch_targets, return_numpy=True)'
        self.gen_return()
        self.gen_indent()
        self.generated_code += '# print(outputs)'
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

    def gen_data(self, name):
        data = np.array(self.scope.var(name).get_tensor())
        path = self.data_dir + os.sep + self.gen_name(name) + ".npy"
        shape = python_array2string(data.shape)
        dtype = numpy_dtype2string(data.dtype)
        np.save(path, data)
        self.gen_indent()
        self.generated_code += self.gen_name(
            name
        ) + ' = paddle.static.create_parameter(name=\'' + name + '\', shape=' + shape + ', dtype=\'' + dtype + '\', attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(np.load(\'./' + self.data_name + os.sep + self.gen_name(
            name) + '.npy\')), trainable=False))'
        self.gen_return()
        return data

    def gen_batch_norm(self, op_desc):
        op_type = op_desc.type()
        x_name = op_desc.input('X')[0]
        bias_name = op_desc.input('Bias')[0]
        self.gen_data(bias_name)
        mean_name = op_desc.input('Mean')[0]
        self.gen_data(mean_name)
        scale_name = op_desc.input('Scale')[0]
        self.gen_data(scale_name)
        variance_name = op_desc.input('Variance')[0]
        self.gen_data(variance_name)
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

    def gen_conv2d(self, op_desc):
        op_type = op_desc.type()
        input_name = op_desc.input('Input')[0]
        filter_name = op_desc.input('Filter')[0]
        filter_data = self.gen_data(filter_name)
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
        ) + ', None, ' + strides + ', ' + paddings + ', ' + dilations + ', ' + groups + ', \'NCHW\')'
        self.gen_return()

    def gen_elementwise_ops(self, op_desc):
        op_type = op_desc.type()
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        out_name = op_desc.output('Out')[0]
        self.gen_indent()
        self.generated_code += self.gen_name(out_name) + ' = '
        if op_type == 'elementwise_add':
            self.generated_code += self.gen_name(
                x_name) + ' + ' + self.gen_name(y_name)
        elif op_type == 'elementwise_sub':
            self.generated_code += self.gen_name(
                x_name) + ' - ' + self.gen_name(y_name)
        elif op_type == 'elementwise_mul':
            self.generated_code += self.gen_name(
                x_name) + ' * ' + self.gen_name(y_name)
        elif op_type == 'elementwise_div':
            self.generated_code += self.gen_name(
                x_name) + ' / ' + self.gen_name(y_name)
        else:
            raise ValueError('Not supported to generate code for %s' % op_type)
        self.gen_return()

    def gen_relu(self, op_desc):
        self.gen_indent()
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        self.generated_code += self.gen_name(
            out_name) + ' = F.relu(' + self.gen_name(x_name) + ')'
        self.gen_return()

    def load_model(self, model_dir, model_filename, params_filename):
        self.place = paddle.CPUPlace()
        self.exe = paddle.static.Executor(place=self.place)
        self.scope = paddle.static.global_scope()
        if len(model_filename) == 0 and len(params_filename) == 0:
            [self.program, self.feed_target_names, self.fetch_targets
             ] = paddle.static.load_inference_model(model_dir, self.exe)
        else:
            [self.program, self.feed_target_names,
             self.fetch_targets] = paddle.static.load_inference_model(
                 model_dir,
                 self.exe,
                 model_filename=model_filename,
                 params_filename=params_filename)
        print('--- feed_target_names ---')
        print(self.feed_target_names)
        print('--- fetch_targets ---')
        print(self.fetch_targets)

    def gen_code(self, code_dir, data_name='data', script_name='model.py'):
        self.code_dir = code_dir
        try:
            os.makedirs(self.code_dir)
        except OSError as e:
            if e.errno != 17:
                raise
        self.data_name = data_name
        self.data_dir = code_dir + os.sep + data_name
        try:
            os.makedirs(self.data_dir)
        except OSError as e:
            if e.errno != 17:
                raise
        self.gen_funcs = {
            'batch_norm': self.gen_batch_norm,
            'conv2d': self.gen_conv2d,
            'elementwise_add': self.gen_elementwise_ops,
            'elementwise_div': self.gen_elementwise_ops,
            'elementwise_mul': self.gen_elementwise_ops,
            'elementwise_sub': self.gen_elementwise_ops,
            'relu': self.gen_relu
        }
        self.gen_head()
        for block_id in range(self.program.num_blocks):
            block = self.program.block(block_id)
            for op_idx in range(len(block.ops)):
                op_desc = block.ops[op_idx].desc
                op_type = op_desc.type()
                if op_type == 'feed' or op_type == 'fetch':
                    continue
                print('Generating ' + op_type + ' ...')
                try:
                    self.gen_funcs[op_type](op_desc)
                except KeyError:
                    raise ValueError("Not supported to generate code for %s " %
                                     op_type)
        self.gen_tail()
        with open(self.code_dir + os.sep + script_name, 'w') as f:
            f.write(self.generated_code)


def main(argv=None):
    code_generator = CodeGenerator()
    code_generator.load_model('./simple_model', 'model.pdmodel',
                              'model.pdiparams')
    code_generator.gen_code('./output_code')
    print("Done.")


if __name__ == '__main__':
    main()
