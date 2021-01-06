import os
import argparse
import tensorflow_datasets as tfds
import tensorflow as tf
import progressbar
import matplotlib.pyplot as plt
import numpy as np
import scipy
import model

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#Define arguments 
parser = argparse.ArgumentParser(description='Download dataset')
parser.add_argument("--samples", type=int,default=1000)
parser.add_argument("--generate_image", type=str2bool,default=True)
parser.add_argument("--metric", type=str,default='fid',choices=['fid','is'])
parser.add_argument("--dataset", type=str, choices=['celeba'])

def load_model():
    dir='./logs'
    try:
        g=tf.keras.models.load_model(os.path.join(dir,'generator.h5'))
        return g
    except:
        print('No appropriate weight file...')
        g=model.build_generator()
        return g
      
def load_celeba():
    return tfds.load('celeb_a',data_dir='./data')['train']


def generate_and_save_images(model,images):
  noise=tf.random.normal([16, 100])
  predictions = model(noise, training=False)
  fig = plt.figure(figsize=(8,4))
  fig.suptitle('Gen images   True images')
  for i in range(predictions.shape[0]):
      plt.subplot(4, 8, i+1+4*(i//4))
      plt.imshow((predictions[i, :, :, :].numpy() * 127.5 + 127.5).astype(int))
      plt.axis('off')

      plt.subplot(4, 8,i+5+4*(i//4))
      plt.imshow((images.numpy()[i] * 127.5 + 127.5).astype(int))
      plt.axis('off')
  
  plt.savefig(f'./final_image.png')

def calculate_fid(act1,act2):
  mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
  mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
  # calculate sum squared difference between means
  ssdiff = np.sum((mu1 - mu2)**2.0)
  # calculate sqrt of product between cov
  covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
  # check and correct imaginary numbers from sqrt
  if np.iscomplexobj(covmean):
      covmean = covmean.real
  # calculate score
  fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

  return fid
def calculate_fid_score(gen_image,true_images):
  input_pipeline=tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(299, 299)
  ])
  input_pipeline_real=tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(299, 299),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5,offset=-1)
  ])
  inception_model=tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
  #Split the process to n_split mini batches
  print('Preprocessing images')
  preprocessed_gen=input_pipeline(gen_image)
  preprocessed_real=input_pipeline_real(true_images)
  print('Calculating FID score')
  act1=inception_model.predict(preprocessed_gen)
  act2=inception_model.predict(preprocessed_real)
  
  return calculate_fid(act1,act2)

def calculate_inception_score(images,eps=1E-16):
    input_pipeline=tf.keras.models.Sequential([tf.keras.layers.experimental.preprocessing.Resizing(299, 299)])
    inception_model=tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

    images = images.astype('float32')
    images = input_pipeline.predict(images)
    p_yx = inception_model.predict(images)

    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    sum_kl_d = kl_d.sum(axis=1)
    avg_kl_d = np.mean(sum_kl_d)
    is_score = np.exp(avg_kl_d)
    return is_score

if __name__ == '__main__':
    args = parser.parse_args()
    tf.random.set_seed(42)
    #Load data
    if args.dataset == 'celeba':
        print("Downloading CelebA dataset...")
        dataset=load_celeba()
        print("Downloading Complete")
    #Build model
    
    input_pipeline=model.build_input()
    print('Loading model...')
    generator=load_model()
    
    if args.generate_image:
        for batch in dataset.batch(16):
            truth_image=input_pipeline(batch['image'])
            break
        generate_and_save_images(generator,truth_image)

    #Evaluate score
    num_examples_to_generate=args.samples
    noise_dim=100
        
    for batch in dataset.batch(num_examples_to_generate):
        truth_image=batch['image']
        break
        
    noise=tf.random.normal([num_examples_to_generate, noise_dim])
    gen_image=generator(noise,training=False)
    
    if args.metric=='fid':
        fid=calculate_fid_score(gen_image,truth_image)
        print('FID Score:',fid)
    elif args.metric=='is':
        inception_s=calculate_inception_score(gen_image)
        print('Inception Score:',inception_s)