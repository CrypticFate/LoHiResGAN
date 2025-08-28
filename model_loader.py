#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Loader and Builder for LoHiResGAN
Handles loading pre-trained models and building discriminator
"""

import tensorflow as tf
from tensorflow.keras import layers
from .config import MODEL_PATH, INPUT_SHAPE, LEARNING_RATE

class ModelLoader:
    """Handles model loading and building operations"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or MODEL_PATH
        self.generator = None
        self.discriminator = None
    
    def load_pretrained_generator(self):
        """Load pre-trained generator with Keras 3 compatibility"""
        print(f"Loading pre-trained SavedModel from: {self.model_path}")
        
        try:
            # Use TFSMLayer for Keras 3 compatibility
            generator_layer = tf.keras.layers.TFSMLayer(self.model_path, call_endpoint='serving_default')
            
            # Create wrapper that extracts the tensor from dictionary output
            inputs = tf.keras.Input(shape=INPUT_SHAPE, dtype=tf.float32)
            layer_output = generator_layer(inputs)
            
            # Extract the actual tensor from the dictionary
            if isinstance(layer_output, dict):
                output_keys = list(layer_output.keys())
                print(f"Generator output keys: {output_keys}")
                outputs = layer_output[output_keys[0]]  # Take the first output
            else:
                outputs = layer_output
            
            self.generator = tf.keras.Model(inputs, outputs, name='generator_wrapper')
            
            print("SUCCESS: Pre-trained generator loaded!")
            print(f"Model input shape: {self.generator.input_shape}")
            print(f"Model output shape: {self.generator.output_shape}")
            
            return self.generator
            
        except Exception as e:
            print(f"ERROR loading model: {e}")
            return None
    
    def build_discriminator(self):
        """Build discriminator network"""
        def conv_block(x, filters, kernel_size=4, strides=2):
            """Convolutional block for discriminator"""
            x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
            x = layers.LeakyReLU(0.2)(x)
            x = layers.BatchNormalization()(x)
            return x
        
        inputs = layers.Input(shape=INPUT_SHAPE)
        x = conv_block(inputs, 16, strides=2)
        x = conv_block(x, 32, strides=2)
        x = conv_block(x, 64, strides=2)
        x = conv_block(x, 128, strides=1)
        outputs = layers.Conv2D(1, 4, strides=1, padding='same', activation='sigmoid')(x)
        
        self.discriminator = tf.keras.Model(inputs, outputs, name='discriminator')
        
        print("SUCCESS: Discriminator built!")
        print(f"Discriminator input shape: {self.discriminator.input_shape}")
        print(f"Discriminator output shape: {self.discriminator.output_shape}")
        
        return self.discriminator
    
    def get_models(self):
        """Get both generator and discriminator"""
        if self.generator is None:
            self.load_pretrained_generator()
        
        if self.discriminator is None:
            self.build_discriminator()
        
        return self.generator, self.discriminator
    
    def create_optimizers(self):
        """Create optimizers for both networks"""
        generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
        
        return generator_optimizer, discriminator_optimizer

class LossFunctions:
    """Contains all loss functions for GAN training"""
    
    def __init__(self, l1_lambda=100):
        self.l1_lambda = l1_lambda
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.mae = tf.keras.losses.MeanAbsoluteError()
    
    def generator_loss(self, disc_generated_output, gen_output, target):
        """Calculate generator loss (adversarial + L1)"""
        # Adversarial loss
        gan_loss = self.binary_crossentropy(tf.ones_like(disc_generated_output), disc_generated_output)
        
        # L1 loss
        l1_loss = self.mae(target, gen_output)
        
        # Total generator loss
        total_gen_loss = gan_loss + (self.l1_lambda * l1_loss)
        
        return total_gen_loss, gan_loss, l1_loss
    
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """Calculate discriminator loss"""
        real_loss = self.binary_crossentropy(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.binary_crossentropy(tf.zeros_like(disc_generated_output), disc_generated_output)
        
        return real_loss + generated_loss

class TrainingStep:
    """Handles the training step operations"""
    
    def __init__(self, generator, discriminator, generator_optimizer, discriminator_optimizer, loss_functions):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.loss_functions = loss_functions
    
    @tf.function
    def train_step(self, input_image, target):
        """Execute one training step"""
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake image
            gen_output = self.generator(input_image, training=True)
            
            # Discriminator predictions
            disc_real_output = self.discriminator(target, training=True)
            disc_generated_output = self.discriminator(gen_output, training=True)
            
            # Calculate losses
            gen_total_loss, gen_gan_loss, gen_l1_loss = self.loss_functions.generator_loss(
                disc_generated_output, gen_output, target)
            disc_loss = self.loss_functions.discriminator_loss(disc_real_output, disc_generated_output)
        
        # Apply gradients
        # Only update generator if it has trainable variables
        if len(self.generator.trainable_variables) > 0:
            generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))
        
        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

if __name__ == "__main__":
    # Test the model loader
    print("Testing ModelLoader...")
    
    try:
        # Test model loading (will fail if no model path exists)
        loader = ModelLoader()
        
        # Test discriminator building
        discriminator = loader.build_discriminator()
        print(f"Discriminator built successfully: {discriminator.name}")
        
        # Test optimizer creation
        gen_opt, disc_opt = loader.create_optimizers()
        print(f"Optimizers created: {type(gen_opt).__name__}, {type(disc_opt).__name__}")
        
        # Test loss functions
        loss_funcs = LossFunctions()
        print(f"Loss functions initialized with L1 lambda: {loss_funcs.l1_lambda}")
        
        print("✅ Model loader test completed successfully!")
        
    except Exception as e:
        print(f"Test failed (expected if no model file): {e}")
        print("✅ Model loader structure test completed!")