import os
import argparse
import tensorflow_datasets as tfds
parser = argparse.ArgumentParser(description='Download dataset')
parser.add_argument("--dataset", type=str, choices=['celeba'])

def download_celeba():
    tfds.load('celeb_a',data_dir='./data/celeba')

def load_imagenet64():
    tfds.load('imagenet_resized/64x64',data_dir='./data/imagenet_64')

def make_folder():
    path='./data'
    if not os.path.exists(path):
        os.mkdir(path)
        
if __name__ == '__main__':
    args = parser.parse_args()
    
    make_folder()
    if args.dataset == 'celeba':
        print("Downloading CelebA dataset...")
        download_celeba()
        print("Downloading Complete")