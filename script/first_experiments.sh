#!/usr/bin/env sh
CUR="${PWD}"
ROOT="${CUR%/*}"
cd $ROOT/src
python train.py AlexNet --batch-size 64 --epoch 200
python train.py DefaultNet --batch-size 64 --epoch 200
python test_for_scene.py --job-name AlexNet
python test_for_scene.py --job-name DefaultNet