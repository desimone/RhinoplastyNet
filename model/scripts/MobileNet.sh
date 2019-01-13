#!/bin/bash
python train.py \
	--epochs 30 \
	--b 64 \
	--lr 1e-3

python train.py \
	--b 64 \
	--resume ./models/MobileNetV2.hdf5 \
	--lr 1e-4

python train.py \
	--b 64 \
	--resume ./models/MobileNetV2.hdf5 \
	--lr 1e-5

python train.py \
	--b 64 \
	--resume ./models/MobileNetV2.hdf5 \
	--lr 1e-6
