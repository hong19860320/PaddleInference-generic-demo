#!/bin/bash
set -e

source settings.sh

# For publish
cd $ROOT_DIR/.. && tar -czvf PaddleInference-generic-demo.tar.gz --exclude=".git" --exclude="log.txt" --exclude="Makefile" --exclude="CMakeFiles" --exclude="build.*.*.*" --exclude="shell/list.txt" --exclude="assets/models/*" --exclude="projects/*"  --exclude="tools/compile_scripts/sdk/*" --exclude="tools/compile_scripts/workspace/*" --exclude="tools/compile_scripts/settings.sh" PaddleInference-generic-demo

# For test
# cd $ROOT_DIR/.. && tar -czvf PaddleInference-generic-demo.tar.gz --exclude=".git" --exclude="log.txt" --exclude="Makefile" --exclude="CMakeFiles" --exclude="build.*.*.*" --exclude="shell/list.txt" --exclude="libs/PaddleInference/*/*/cpu" --exclude="libs/PaddleInference/*/*/xpu" --exclude="assets/models/*" --exclude="projects/*"  --exclude="tools/compile_scripts/sdk/*" --exclude="tools/compile_scripts/workspace/*" --exclude="tools/compile_scripts/settings.sh" PaddleInference-generic-demo

echo "Done."
