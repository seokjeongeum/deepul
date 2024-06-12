#!/bin/sh

python train_mnist.py --model rnn
python train_mnist.py --model rnn_loc
python train_mnist.py --model made --order_id 0
python train_mnist.py --model pixelcnn
python train_mnist.py --model wavenet
python train_mnist.py --model wavenet_loc
python train_mnist.py --model transformer
