# selfattention_old

# Continuously Variable Networks



Best result:

python main.py --arch cvnet --append False --scaling_rate 0.75 --growth_rate 64  --n_channel 128 --n_epoch 350 --batch_size 512 --optimizer Adam --lr 0.001

python main.py --arch cvnet --append False --scaling_rate 0.75 --growth_rate 64  --n_channel 128 --n_epoch 350 --batch_size 512 --optimizer SGD --lr 0.1


Other ways to run:

Use all available GPUs.
python main.py --multi_gpu True

Use a specific gpu:
python main.py --gpu_id 3

Visualize:
python main.py --visualize True --dataset stl10
