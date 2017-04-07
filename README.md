# BEGAN in PyTorch

**This project is still in progress. If you are looking for the working code, use** [BEGAN-tensorflow](https://github.com/carpedm20/BEGAN-tensorflow).


## Requirements

- Python 2.7
- [Pillow](https://pillow.readthedocs.io/en/4.0.x/)
- [tqdm](https://github.com/tqdm/tqdm)
- [PyTorch](https://github.com/pytorch/pytorch)
- [torch-vision](https://github.com/pytorch/vision)
- [requests](https://github.com/kennethreitz/requests) (Only used for downloading CelebA dataset)
- [TensorFlow](https://github.com/tensorflow/tensorflow) (Only used TensorBoard for logging)


## Usage

First download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets with:

    $ apt-get install p7zip-full # ubuntu
    $ brew install p7zip # Mac
    $ python download.py

or you can use your own dataset by placing images like:

    data
    └── YOUR_DATASET_NAME
        ├── xxx.jpg (name doesn't matter)
        ├── yyy.jpg
        └── ...

To train a model:

    $ python main.py --dataset=CelebA --num_gpu=1
    $ python main.py --dataset=YOUR_DATASET_NAME --num_gpu=4 --use_tensorboard=True

To test a model (use your `load_path`):

    $ python main.py --dataset=CelebA --load_path=./logs/CelebA_0405_124806 --num_gpu=0 --is_train=False --split valid


## Results

- [BEGAN-tensorflow](https://github.com/carpedm20/began-tensorflow) at least can generate human faces but [BEGAN-pytorch](https://github.com/carpedm20/BEGAN-pytorch) can't.
- Both [BEGAN-tensorflow](https://github.com/carpedm20/began-tensorflow) and [BEGAN-pytorch](https://github.com/carpedm20/BEGAN-pytorch) shows **modal collapses** and I guess this is due to a wrong scheuduling of lr (Paper mentioned that *simply reducing the lr was sufficient to avoid them*).

![alt tag](./assets/board-2017-04-04.png)

(in progress)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io)
