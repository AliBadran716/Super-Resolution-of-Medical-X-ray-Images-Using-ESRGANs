import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications.vgg19 import VGG19
import numpy as np

class ESRGAN:
    """
    Implementation of ESRGAN following the paper:
    'ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks'
    For grayscale medical images.
    """
    def __init__(self, scale_factor=4):
        self.scale_factor = scale_factor
        self.generator = None
        self.discriminator = None
        self.vgg = None
        self.build_models()
        
    def _residual_dense_block(self, x, features=64):
        """Residual Dense Block"""
        concat_features = []
        input_features = x
        
        for i in range(5):
            if concat_features:
                x = layers.Concatenate()(concat_features + [x])
            x = layers.Conv2D(features, (3, 3), padding='same')(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            concat_features.append(x)
            
        x = layers.Concatenate()(concat_features)
        x = layers.Conv2D(features, (1, 1), padding='same')(x)
        
        # Local residual learning
        return layers.Add()([input_features, x * 0.2])
    
    def _rrdb_block(self, x, features=64):
        """Residual in Residual Dense Block"""
        input_features = x
        
        for _ in range(3):
            x = self._residual_dense_block(x, features)
            
        # Residual scaling
        return layers.Add()([input_features, x * 0.2])
    
    def build_models(self):
        # Generator (Modified for Grayscale)
        lr_input = Input(shape=(None, None, 1))  # Grayscale input (single channel)
        
        # First conv
        x = layers.Conv2D(64, (3, 3), padding='same')(lr_input)
        initial_feature = x
        
        # RRDB blocks (23 blocks as in paper)
        for _ in range(23):
            x = self._rrdb_block(x)
            
        # Global feature fusion
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        trunk = layers.Add()([initial_feature, x])
        
        # Upsampling blocks (4x)
        for _ in range(2):  # Two blocks for 4x upsampling
            x = layers.Conv2D(256, (3, 3), padding='same')(trunk)
            x = tf.nn.depth_to_space(x, 2)  # Pixel shuffle
            x = layers.LeakyReLU(0.2)(x)
            trunk = x
        
        # Final conv
        sr_output = layers.Conv2D(1, (3, 3), padding='same', activation='tanh')(trunk)  # Single channel output
        
        self.generator = Model(lr_input, sr_output, name='generator')
        
        # Discriminator (Modified for Grayscale)
        def d_block(x, filters, strides=1, bn=True):
            x = layers.Conv2D(filters, (3, 3), strides=strides, padding='same')(x)
            if bn:
                x = layers.BatchNormalization()(x)
            return layers.LeakyReLU(alpha=0.2)(x)
        
        d_input = Input(shape=(None, None, 1))  # Grayscale input
        
        # Series of Conv + LeakyReLU + BN
        features = [64, 64, 128, 128, 256, 256, 512, 512]
        x = d_input
        
        for idx, f in enumerate(features):
            x = d_block(x, f, strides=2 if idx % 2 == 1 else 1)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dense(1, activation='sigmoid')(x)
        
        self.discriminator = Model(d_input, x, name='discriminator')
        
        # VGG feature extractor for perceptual loss
        vgg = VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))  # VGG expects 3 channels, but we'll use grayscale
        self.vgg = Model(inputs=vgg.input,
                        outputs=vgg.get_layer('block5_conv4').output,
                        name='vgg')
        self.vgg.trainable = False
        
    def compile(self, 
                gen_lr=1e-4, 
                disc_lr=1e-4,
                content_weight=1.0,
                perceptual_weight=1.0,
                adversarial_weight=0.1):
        
        self.gen_optimizer = tf.keras.optimizers.Adam(gen_lr, beta_1=0.9, beta_2=0.99)
        self.disc_optimizer = tf.keras.optimizers.Adam(disc_lr, beta_1=0.9, beta_2=0.99)
        
        self.content_weight = content_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        
    @tf.function
    def train_step(self, lr_images, hr_images):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake images
            sr_images = self.generator(lr_images, training=True)
            
            # Discriminator outputs
            real_output = self.discriminator(hr_images, training=True)
            fake_output = self.discriminator(sr_images, training=True)
            
            # Content loss (L1 loss as per paper)
            content_loss = tf.reduce_mean(tf.abs(hr_images - sr_images))
            
            # Perceptual loss
            hr_features = self.vgg(hr_images)
            sr_features = self.vgg(sr_images)
            perceptual_loss = tf.reduce_mean(tf.abs(hr_features - sr_features))
            
            # Relativistic average GAN loss
            real_logits = real_output - tf.reduce_mean(fake_output)
            fake_logits = fake_output - tf.reduce_mean(real_output)
            
            disc_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(real_logits), logits=real_logits
                ) +
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(fake_logits), logits=fake_logits
                )
            )
            
            gen_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(fake_logits), logits=fake_logits
                ) +
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(real_logits), logits=real_logits
                )
            )
            
            # Total generator loss
            total_gen_loss = (
                self.content_weight * content_loss +
                self.perceptual_weight * perceptual_loss +
                self.adversarial_weight * gen_loss
            )
            
        # Compute gradients
        gen_gradients = gen_tape.gradient(
            total_gen_loss, self.generator.trainable_variables
        )
        disc_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )
        
        # Apply gradients
        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )
        
        return {
            'content_loss': content_loss,
            'perceptual_loss': perceptual_loss,
            'gen_loss': gen_loss,
            'disc_loss': disc_loss
        }

class DataLoader:
    def __init__(self, image_dir, batch_size=16, hr_size=128, scale_factor=4):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.hr_size = hr_size
        self.lr_size = hr_size // scale_factor
        self.scale_factor = scale_factor
        
        self.dataset = self._create_dataset()
    
    def _load_and_process(self, path):
        # Load image
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=1)  # Read as grayscale
        img = tf.cast(img, tf.float32) / 127.5 - 1  # Normalize to [-1, 1]
        
        # Random crop
        img = tf.image.random_crop(img, [self.hr_size, self.hr_size, 1])
        
        # Create low-res version
        lr_img = tf.image.resize(img, [self.lr_size, self.lr_size],
                               method='bicubic')
        
        return lr_img, img
    
    def _create_dataset(self):
        # Get image paths
        image_paths = tf.data.Dataset.list_files(str(self.image_dir + '/*'))
        
        # Create dataset
        dataset = (image_paths
                  .map(self._load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(self.batch_size)
                  .prefetch(tf.data.AUTOTUNE))
        
        return dataset
    
# Initialize model
model = ESRGAN(scale_factor=4)

# Compile with custom loss weights if needed
model.compile(
    gen_lr=1e-4, 
    disc_lr=1e-4,
    content_weight=1.0, 
    perceptual_weight=1.0,
    adversarial_weight=0.1
)
