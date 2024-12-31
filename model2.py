import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as K
import os
from pathlib import Path
import cv2

class VDSR:
    """
    Implementation of VDSR following the original paper:
    'Accurate Image Super-Resolution Using Very Deep Convolutional Networks'
    """
    def __init__(self):
        self.model = None
        self.build_model()
        
    def build_model(self):
        inputs = Input(shape=(None, None, 1))  # Flexible input size
        
        # Extract features
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        
        # 18 layers with residual learning (20 layers total)
        for _ in range(18):
            x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
            
        # Final reconstruction
        x = layers.Conv2D(1, (3, 3), padding='same')(x)
        
        # Global residual learning
        outputs = layers.Add()([inputs, x])
        
        self.model = Model(inputs, outputs, name='vdsr')
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='mse',
            metrics=['psnr']
        )

class ESRGAN:
    """
    Implementation of ESRGAN following the original paper:
    'ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks'
    """
    def __init__(self):
        self.generator = None
        self.discriminator = None
        self.vgg = None
        self.build_models()
        
    def _residual_dense_block(self, x, growth_channels=32):
        """
        Residual Dense Block as described in ESRGAN paper
        """
        original = x
        channel_num = growth_channels
        
        # Dense connections
        conv_layers = []
        for i in range(5):
            if i > 0:
                x = layers.Concatenate()(conv_layers + [x])
            x = layers.Conv2D(channel_num, (3, 3), padding='same')(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            conv_layers.append(x)
            
        # Local feature fusion
        x = layers.Concatenate()(conv_layers)
        x = layers.Conv2D(64, (1, 1), padding='same')(x)
        
        # Local residual learning
        return layers.Add()([original, x * 0.2])
    
    def _rrdb_block(self, x):
        """
        Residual in Residual Dense Block
        """
        original = x
        
        # 3 Residual Dense Blocks
        for _ in range(3):
            x = self._residual_dense_block(x)
            
        # Residual scaling
        return layers.Add()([original, x * 0.2])
    
    def build_models(self):
        # Build Generator
        inputs = Input(shape=(None, None, 1))
        
        # First conv
        x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
        conv1 = x
        
        # RRDB blocks (23 as in paper)
        for _ in range(23):
            x = self._rrdb_block(x)
            
        # Global feature fusion
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.Add()([conv1, x])
        
        # Upsampling (4x using pixel shuffle)
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = tf.nn.depth_to_space(x, 2)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = tf.nn.depth_to_space(x, 2)
        x = layers.LeakyReLU(0.2)(x)
        
        outputs = layers.Conv2D(1, (3, 3), padding='same')(x)
        
        self.generator = Model(inputs, outputs, name='generator')
        
        # Build Discriminator (VGG-style)
        def discriminator_block(x, filters, strides=1, bn=True):
            x = layers.Conv2D(filters, (3, 3), strides=strides, padding='same')(x)
            if bn:
                x = layers.BatchNormalization()(x)
            return layers.LeakyReLU(alpha=0.2)(x)
        
        d_input = Input(shape=(None, None, 1))
        x = discriminator_block(d_input, 64, bn=False)
        x = discriminator_block(x, 64, strides=2)
        x = discriminator_block(x, 128)
        x = discriminator_block(x, 128, strides=2)
        x = discriminator_block(x, 256)
        x = discriminator_block(x, 256, strides=2)
        x = discriminator_block(x, 512)
        x = discriminator_block(x, 512, strides=2)
        
        x = layers.Dense(64)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dense(1)(x)
        
        self.discriminator = Model(d_input, x, name='discriminator')
        
        # Build VGG for perceptual loss
        vgg = VGG19(include_top=False, weights='imagenet')
        self.vgg = Model(inputs=vgg.input, 
                        outputs=vgg.get_layer('block5_conv4').output,
                        name='vgg')
        self.vgg.trainable = False

class HybridSuperResolution:
    def __init__(self):
        self.vdsr = VDSR()
        self.esrgan = ESRGAN()
        
        # Optimizers
        self.vdsr_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.gen_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.disc_optimizer = tf.keras.optimizers.Adam(1e-4)
        
        # Loss weights
        self.content_weight = 1.0
        self.adversarial_weight = 0.005
        self.perceptual_weight = 1.0
    
    @tf.function
    def train_step(self, lr_images, hr_images):
        # Train VDSR
        with tf.GradientTape() as vdsr_tape:
            vdsr_output = self.vdsr.model(lr_images, training=True)
            vdsr_loss = tf.reduce_mean(tf.square(hr_images - vdsr_output))
        
        # Update VDSR
        vdsr_grads = vdsr_tape.gradient(vdsr_loss, self.vdsr.model.trainable_variables)
        self.vdsr_optimizer.apply_gradients(zip(vdsr_grads, self.vdsr.model.trainable_variables))
        
        # Get VDSR enhanced images for ESRGAN
        vdsr_enhanced = self.vdsr.model(lr_images, training=False)
        
        # Train ESRGAN
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generator forward pass
            sr_images = self.esrgan.generator(vdsr_enhanced, training=True)
            
            # Discriminator forward pass
            hr_output = self.esrgan.discriminator(hr_images, training=True)
            sr_output = self.esrgan.discriminator(sr_images, training=True)
            
            # Calculate losses
            # Content loss
            content_loss = tf.reduce_mean(tf.square(hr_images - sr_images))
            
            # Perceptual loss
            hr_features = self.esrgan.vgg(tf.tile(hr_images, [1, 1, 1, 3]))
            sr_features = self.esrgan.vgg(tf.tile(sr_images, [1, 1, 1, 3]))
            perceptual_loss = tf.reduce_mean(tf.square(hr_features - sr_features))
            
            # Adversarial loss
            gen_loss = tf.reduce_mean(tf.square(sr_output - 1))
            disc_loss = tf.reduce_mean(tf.square(hr_output - 1) + tf.square(sr_output))
            
            # Total generator loss
            total_gen_loss = (self.content_weight * content_loss + 
                            self.perceptual_weight * perceptual_loss +
                            self.adversarial_weight * gen_loss)
        
        # Update discriminator
        disc_grads = disc_tape.gradient(
            disc_loss, self.esrgan.discriminator.trainable_variables
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.esrgan.discriminator.trainable_variables)
        )
        
        # Update generator
        gen_grads = gen_tape.gradient(
            total_gen_loss, self.esrgan.generator.trainable_variables
        )
        self.gen_optimizer.apply_gradients(
            zip(gen_grads, self.esrgan.generator.trainable_variables)
        )
        
        return {
            'vdsr_loss': vdsr_loss,
            'content_loss': content_loss,
            'perceptual_loss': perceptual_loss,
            'gen_loss': gen_loss,
            'disc_loss': disc_loss
        }

class DataLoader:
    def __init__(self, image_dir, batch_size=16, low_res=128, high_res=512):
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.low_res = low_res
        self.high_res = high_res
        
        # Get image files
        self.image_files = list(self.image_dir.glob('*.png')) + \
                          list(self.image_dir.glob('*.jpg'))
        
        self.dataset = self._create_dataset()
    
    def _load_image(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.cast(img, tf.float32) / 255.0
        
        # Create high-res ground truth
        hr_img = tf.image.resize(img, [self.high_res, self.high_res],
                               method='bicubic')
        
        # Create low-res input
        lr_img = tf.image.resize(img, [self.low_res, self.low_res],
                               method='bicubic')
        
        return lr_img, hr_img
    
    def _create_dataset(self):
        paths = tf.data.Dataset.from_tensor_slices([str(f) for f in self.image_files])
        dataset = paths.map(self._load_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

def train(train_dir, val_dir, output_dir, epochs=100, batch_size=16):
    # Create model and data loaders
    model = HybridSuperResolution()
    train_loader = DataLoader(train_dir, batch_size=batch_size)
    val_loader = DataLoader(val_dir, batch_size=batch_size)
    
    # Create checkpoint manager
    ckpt = tf.train.Checkpoint(
        vdsr=model.vdsr.model,
        esrgan_gen=model.esrgan.generator,
        esrgan_disc=model.esrgan.discriminator,
        vdsr_opt=model.vdsr_optimizer,
        gen_opt=model.gen_optimizer,
        disc_opt=model.disc_optimizer
    )
    
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, output_dir, max_to_keep=3
    )
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training
        for batch_idx, (lr_images, hr_images) in enumerate(train_loader.dataset):
            losses = model.train_step(lr_images, hr_images)
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: VDSR Loss = {losses['vdsr_loss']:.4f}, "
                      f"Content Loss = {losses['content_loss']:.4f}, "
                      f"Gen Loss = {losses['gen_loss']:.4f}")
        
        # Validation and checkpoint saving
        if (epoch + 1) % 5 == 0:
            # Calculate validation metrics
            psnr_values = []
            ssim_values = []
            
            for lr_images, hr_images in val_loader.dataset:
                sr_images = model.esrgan.generator(
                    model.vdsr.model(lr_images, training=False),
                    training=False
                )
                
                psnr = tf.reduce_mean(tf.image.psnr(hr_images, sr_images, max_val=1.0))
                ssim = tf.reduce_mean(tf.image.ssim(hr_images, sr_images, max_val=1.0))
                
                psnr_values.append(psnr)
                ssim_values.append(ssim)
            
            print(f"Validation PSNR: {np.mean(psnr_values):.2f}")
            print(f"Validation SSIM: {np.mean(ssim_values):.2f}")
            
            # Save checkpoint
            ckpt_manager.save()

if __name__ == "__main__":
    # Example usage
    train(
        train_dir="path/to/train/images",
        val_dir="path/to/val/images",
        output_dir="checkpoints",
        epochs=100,
        batch_size=16
    )