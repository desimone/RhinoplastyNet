#!/bin/bash

python train.py \
	--epochs 30 \
	--b 128

python train.py \
	--b 128 \
	--resume ./models/NASNetMobile.hdf5 \
	--lr 1e-4

python train.py \
	--b 128 \
	--resume ./models/NASNetMobile.hdf5 \
	--lr 1e-5

python train.py \
	--b 128 \
	--resume ./models/NASNetMobile.hdf5 \
	--lr 1e-6
