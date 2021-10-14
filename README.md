
# Implementation of Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks

This repository contains the implementation of a famous image-to-image translation paper, "Image-to-Image Translation with Conditional Adversarial Networks".

Image-to-image translation translates the one possible representation of a scene into another for example conversion of sketch image to real look colorful image. 


In this paper, authors explored conditional GAN. Conditional GAN learns a conditional generative model. This Makes them suitable for image-to-image translation where we condition on an input image and generate a corresponding output image. In GAN's generator author used UNET that is encoder-decoder with a skips connection. In the discriminator, they used convolutional 'PatchGAN', which only penalizes structure at the scale of image patches.

Some of the Pix2Pix translation examples:


![pix2pix](https://user-images.githubusercontent.com/16369846/137012186-95512879-c6e7-4336-90ab-9ce71c895968.jpg)

## Repository:

The repositories contain the following files.
* UNET (Directory)
	* This directory contains the code of unet model. The UNET model is used as a generator in this paper.
* pretrained(Directory)
	* We are utilizing a pretrained pix2pix in order to save time and visualize the results faster
* helper_functions. py
	* This file contains small functions that were use in other files.
*  disciminator. py
	* This file contains of discrimnator of GAN.
*  train. py
	* This file combines all component and perform the training.


In order to run the training and visualize the results, you need to run this command. 
```
python train.py
```
