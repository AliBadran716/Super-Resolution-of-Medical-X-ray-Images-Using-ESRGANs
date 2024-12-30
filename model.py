import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as K

# Constants
LOW_RES = 128
HIGH_RES = 512
CHANNELS = 1  # Grayscale images

class VDSRModel:
    def __init__(self):
        self.model = None
        self.build_model()
    
    def build_model(self):
        inputs = Input(shape=(LOW_RES, LOW_RES, CHANNELS))
        
        # Initial feature extraction
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        
        # 20 residual blocks
        for _ in range(20):
            x = self._residual_block(x)
        
        # Final reconstruction
        x = layers.Conv2D(CHANNELS, (3, 3), padding='same')(x)
        
        # Upsampling to target resolution
        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
        
        self.model = Model(inputs=inputs, outputs=x, name='vdsr')
    
    def _residual_block(self, x):
        skip = x
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        return layers.Add()([x, skip])

class ESRGANModel:
    def __init__(self):
        self.generator = None
        self.discriminator = None
        self.vgg = None
        self.build_models()
    
    def build_models(self):
        self._build_generator()
        self._build_discriminator()
        self._build_vgg()
    
    def _build_generator(self):
        inputs = Input(shape=(LOW_RES, LOW_RES, CHANNELS))
        
        # Initial convolution
        x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
        x = layers.LeakyReLU(0.2)(x)
        
        # RRDB blocks
        for _ in range(23):
            x = self._rrdb_block(x)
        
        # Upsampling
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
        x = layers.Conv2D(CHANNELS, (3, 3), padding='same', activation='tanh')(x)
        
        self.generator = Model(inputs=inputs, outputs=x, name='generator')
    
    def _build_discriminator(self):
        inputs = Input(shape=(HIGH_RES, HIGH_RES, CHANNELS))
        
        x = layers.Conv2D(64, (3, 3), strides=1, padding='same')(inputs)
        x = layers.LeakyReLU(0.2)(x)
        
        features = 64
        for i in range(6):
            x = layers.Conv2D(features * 2, (3, 3), strides=2, padding='same')(x)
            x = layers.LeakyReLU(0.2)(x)
            features *= 2
        
        x = layers.Flatten()(x)
        x = layers.Dense(1024)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dense(1, activation='sigmoid')(x)
        
        self.discriminator = Model(inputs=inputs, outputs=x, name='discriminator')
    
    def _build_vgg(self):
        vgg = VGG19(include_top=False, weights='imagenet')
        self.vgg = Model(inputs=vgg.input, 
                        outputs=vgg.get_layer('block5_conv4').output,
                        name='vgg')
        self.vgg.trainable = False
    
    def _rrdb_block(self, x):
        skip = x
        
        # Dense blocks
        for _ in range(3):
            x_temp = x
            for _ in range(5):
                y = layers.Conv2D(64, (3, 3), padding='same')(x_temp)
                y = layers.LeakyReLU(0.2)(y)
                x_temp = layers.Concatenate()([x_temp, y])
            
            x = layers.Conv2D(64, (3, 3), padding='same')(x_temp)
            x = layers.LeakyReLU(0.2)(x)
            x = layers.Add()([x, skip])
            
        return x * 0.2 + skip

class HybridSuperResolution:
    def __init__(self):
        self.vdsr = VDSRModel()
        self.esrgan = ESRGANModel()
        self.combined_model = None
        self.build_combined_model()
    
    def build_combined_model(self):
        inputs = Input(shape=(LOW_RES, LOW_RES, CHANNELS))
        
        # VDSR intermediate output
        vdsr_output = self.vdsr.model(inputs)
        
        # ESRGAN final output
        final_output = self.esrgan.generator(vdsr_output)
        
        self.combined_model = Model(inputs=inputs, outputs=final_output)
    
    def compile_models(self):
        # Compile VDSR
        self.vdsr.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='mse',
            metrics=[self.psnr, self.ssim]
        )
        
        # Compile ESRGAN
        self.esrgan.discriminator.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy'
        )
        
        # Compile combined model
        self.combined_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=[self.content_loss, self.adversarial_loss],
            loss_weights=[1.0, 0.1],
            metrics=[self.psnr, self.ssim]
        )
    
    def psnr(self, y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1.0)
    
    def ssim(self, y_true, y_pred):
        return tf.image.ssim(y_true, y_pred, max_val=1.0)
    
    def content_loss(self, y_true, y_pred):
        # VGG feature loss
        vgg_true = self.esrgan.vgg(y_true)
        vgg_pred = self.esrgan.vgg(y_pred)
        return tf.keras.losses.MeanSquaredError()(vgg_true, vgg_pred)
    
    def adversarial_loss(self, y_true, y_pred):
        return tf.keras.losses.BinaryCrossentropy()(
            tf.ones_like(y_pred),
            self.esrgan.discriminator(y_pred)
        )

# Data loading and preprocessing
def load_and_preprocess_data(data_path):
    def preprocess_image(image):
        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        # Convert to grayscale if needed
        if image.shape[-1] == 3:
            image = tf.image.rgb_to_grayscale(image)
        return image
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(data_path)
    dataset = dataset.map(lambda x: (
        preprocess_image(tf.image.resize(x, [LOW_RES, LOW_RES])),
        preprocess_image(tf.image.resize(x, [HIGH_RES, HIGH_RES]))
    ))
    
    return dataset

# Training function
def train_model(model, train_dataset, val_dataset, epochs=100, batch_size=32):
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Train VDSR first
        model.vdsr.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=1,
            batch_size=batch_size
        )
        
        # Train ESRGAN
        for batch in train_dataset:
            # Generate intermediate results using VDSR
            vdsr_output = model.vdsr.model.predict(batch[0])
            
            # Train discriminator
            real_loss = model.esrgan.discriminator.train_on_batch(
                batch[1], tf.ones((batch_size, 1))
            )
            fake_output = model.esrgan.generator.predict(vdsr_output)
            fake_loss = model.esrgan.discriminator.train_on_batch(
                fake_output, tf.zeros((batch_size, 1))
            )
            
            # Train generator (combined model)
            g_loss = model.combined_model.train_on_batch(
                batch[0], batch[1]
            )
        
        # Validate and print metrics
        val_metrics = model.combined_model.evaluate(val_dataset)
        print(f"Validation metrics: {val_metrics}")

# Usage example
if __name__ == "__main__":
    # Initialize model
    model = HybridSuperResolution()
    model.compile_models()
    
    # Load and preprocess data
    train_dataset = load_and_preprocess_data("path_to_train_data")
    val_dataset = load_and_preprocess_data("path_to_val_data")
    
    # Train model
    train_model(model, train_dataset, val_dataset)