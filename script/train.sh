#!/usr/bin/env sh
CUR="${PWD}"
ROOT="${CUR%/*}"
cd $ROOT/src
python train.py AlexNet