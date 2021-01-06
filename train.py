import os
import argparse
import tensorflow_datasets as tfds
import tensorflow as tf
import progressbar
import matplotlib.pyplot as plt
import time

import model
import evaluate

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
parser.add_argument("--samples_for_eval", type=int,default=1000)
parser.add_argument("--initial_epoch", type=int,default=0)
parser.add_argument("--epoch", type=int,default=100)
parser.add_argument("--evaluate_FID", type=str2bool,default=True)
parser.add_argument("--load_model", type=str2bool,default=True)
parser.add_argument("--dataset", type=str, choices=['celeba','cifar10'])
parser.add_argument("--generate_image", type=str2bool,default=True)
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--learning_rate_dis",type=float,default=0.0001)
parser.add_argument("--learning_rate_gen",type=float,default=0.0001)

parser.add_argument("--feature_matching", type=str2bool,default=True)

def save_model(g,d):
    dir='./logs'
    g.save(os.path.join(dir,'generator.h5'))
    d.save(os.path.join(dir,'discriminator.h5'))

def create_model(image_size):
    i=model.build_input(image_size)
    g=model.build_generator(image_size)
    d=model.build_discriminator(image_size)
    return i,g,d
    
def load_model(image_size):
    
    dir='./logs'
    try:
        i=model.build_input(image_size)
        g=tf.keras.models.load_model(os.path.join(dir,'generator.h5'))
        d=tf.keras.models.load_model(os.path.join(dir,'discriminator.h5'))
        
        if tuple(d.input.shape)[1:-1]==image_size and tuple(g.output.shape)[1:-1]==image_size:
            print('Loading weights')
            return i,g,d
        else:
            print('Wrong weight file dimensions.')
            return create_model(image_size)
    except:
        #If file doesn't exist
        print('No existing weight file...')
        return create_model(image_size)
      
def load_celeba():
    return tfds.load('celeb_a',data_dir='./data')['train']

def load_cifar10():
    (train_images, _), (_, _)=tf.keras.datasets.cifar10.load_data()
    return tf.data.Dataset.from_tensor_slices(train_images)

def make_folder():
    paths=['./logs','./logs/images']
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
    
def generate_and_save_images(model, epoch, test_input):
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow((predictions[i, :, :, :].numpy() * 127.5 + 127.5).astype(int))
      plt.axis('off')

  plt.savefig(f'./logs/images/epoch_{epoch}.png')
  plt.close()
  
@tf.function
def train_step(images):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    
    logs={}
    noise = tf.random.normal([args.batch_size, noise_dim])
    #Define feature matching model
    if args.feature_matching:
        feature_discriminator=tf.keras.models.Sequential(discriminator.layers[:-2])
        final_model=tf.keras.models.Sequential(discriminator.layers[-2:])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        #Forward propagate through GAN
        if args.feature_matching:
            feature_real=feature_discriminator(input_pipeline(images), training=True)
            feature_fake=feature_discriminator(generated_images, training=True)

            real_output = final_model(feature_real, training=True)
            fake_output = final_model(feature_fake, training=True)
        else: 
            real_output = discriminator(input_pipeline(images), training=True)
            fake_output = discriminator(generated_images, training=True)

        #Calculate loss
        if args.feature_matching:
            gen_loss=tf.keras.losses.MSE(tf.reduce_mean(feature_fake,axis=0),tf.reduce_mean(feature_real,axis=0))
        else:
            gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        logs['g_loss']=gen_loss
        logs['d_loss']=disc_loss

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return logs
def get_FID(gen,images):
    num_samples=images.shape[0]
    seed = tf.random.normal([num_samples, noise_dim])
    return evaluate.calculate_fid_score(gen.predict(seed),images)

def plot_losses(args,lists):
    plt.plot(lists['G_loss'],label='G_loss')
    plt.plot(lists['D_loss'],label='D_loss')
    plt.legend()
    plt.savefig(f'./logs/loss_graph.png')
    plt.close()

    if args.evaluate_FID:
        plt.plot(lists['FID'],label='FID')
        plt.legend()
        plt.savefig(f'./logs/FID_graph.png')
        plt.close()

if __name__ == '__main__':
    args = parser.parse_args()
    tf.random.set_seed(42)
    #Load data/model
    make_folder()
    
    if args.dataset == 'celeba':
        print("Downloading CelebA dataset...")
        dataset=load_celeba()
        image_size=(64,64)
        print("Downloading Complete")
    elif args.dataset=='cifar10':
        print("Downloading CIFAR 10 dataset...")
        dataset=load_cifar10()
        image_size=(32,32)
        print("Downloading Complete")
    
    #Load model
    if args.load_model:
        input_pipeline,generator,discriminator=load_model(image_size)
    else:
        input_pipeline,generator,discriminator=create_model(image_size)

    
    #Train loop
    tf.random.set_seed(42)
    noise_dim = 100
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    losses={'G_loss':[],'D_loss':[],'FID':[]}

    generator_optimizer = tf.keras.optimizers.Adam(args.learning_rate_gen)
    discriminator_optimizer = tf.keras.optimizers.Adam(args.learning_rate_dis)

    for epoch in range(args.initial_epoch,args.epoch):
        start = time.time()
    
        for image_batch in progressbar.progressbar(dataset.batch(args.batch_size)):
          if args.dataset=='celeba':
              image_batch=image_batch['image']
          logs=train_step(image_batch)
          losses['G_loss'].append(logs['g_loss'])
          losses['D_loss'].append(logs['d_loss'])

          
        # Produce images for the GIF as we go
        if args.generate_image:
            generate_and_save_images(generator,
                                     epoch + 1,
                                     seed)
        save_model(generator,discriminator)
        plot_losses(args,losses)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        for image_batch in dataset.batch(args.samples_for_eval):
            if args.dataset=='celeba':
              image_batch=image_batch['image']
            
            if args.evaluate_FID:
                FID=get_FID(generator,image_batch)
                losses['FID'].append(FID)
                print('FID Score:',FID)
            break
    generate_and_save_images(generator,
                                 'final',
                                 seed)

    save_model(generator,discriminator)
    print("Training completed...")
    
