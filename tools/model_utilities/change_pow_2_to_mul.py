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

SRC_MODEL_DIR = "./input_model"
DST_MODEL_DIR = "./output_model"
MODEL_FILE = "model.pdmodel"
PARAMS_FILE = "model.pdiparams"

#MODEL_FILE = ""
#PARAMS_FILE = ""


def main(argv=None):
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place=place)
    if len(MODEL_FILE) == 0 and len(PARAMS_FILE) == 0:
        [program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(SRC_MODEL_DIR, exe)
    else:
        [program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
             SRC_MODEL_DIR,
             exe,
             model_filename=MODEL_FILE,
             params_filename=PARAMS_FILE)
    print('--- origin feed_target_names ---')
    print(feed_target_names)
    print('--- origin fetch_targets ---')
    print(fetch_targets)
    try:
        os.makedirs(DST_MODEL_DIR)
    except OSError as e:
        if e.errno != 17:
            raise
    # Convert elementwise_pow op to elementwise_mul
    scope = fluid.global_scope()
    main_block = program.block(0)
    for i in range(len(main_block.ops)):
        op_desc = main_block.ops[i].desc
        op_type = op_desc.type()
        if op_type == 'elementwise_pow':
            y_name = op_desc.input("Y")[0]
            y_var = main_block.var(y_name)
            y_shape = y_var.shape
            if y_var.desc.persistable() == True:
                if len(y_shape) == 1 and y_shape[0] == 1:
                    y_tensor = np.array(scope.var(y_name).get_tensor())
                    if y_tensor[0] == 2:
                        print("Found Y=%s!" % y_name)
                        x_name = op_desc.input("X")[0]
                        print("Found X=%s!" % x_name)
                        op_desc.set_input("Y", [x_name])
                        op_desc.set_type("elementwise_mul")

    print('--- new feed_target_names ---')
    print(feed_target_names)
    print('--- new fetch_targets ---')
    print(fetch_targets)
    if len(MODEL_FILE) == 0 and len(PARAMS_FILE) == 0:
        fluid.io.save_inference_model(DST_MODEL_DIR, feed_target_names,
                                      fetch_targets, exe, program)
    else:
        fluid.io.save_inference_model(
            DST_MODEL_DIR,
            feed_target_names,
            fetch_targets,
            exe,
            program,
            model_filename=MODEL_FILE,
            params_filename=PARAMS_FILE)
    print("Done.")


if __name__ == '__main__':
    main()
