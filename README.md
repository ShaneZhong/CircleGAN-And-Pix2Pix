# Using CircleGAN and Pix2Pix in Tensorflow 2.0
This repo contains Colab notebooks that allow you to implement Pix2Pix and CircleGAN using TensorFlow 2.0. Below is the high-level overview of the two models.

## Pix2Pix:
paper: https://phillipi.github.io/pix2pix/

pix2pix uses a conditional generative adversarial network (cGAN) to learn a mapping from an input image to an output image.

An example of a dataset would be that the input image is a black and white picture and the target image is the color version of the picture. The generator in this case is trying to learn how to colorize a black and white image. The discriminator is looking at the generator’s colorization attempts and trying to learn to tell the difference between the colorizations the generator provides and the true colorized target image provided in the dataset.

The structure of the generator is called an “encoder-decoder” and in pix2pix the encoder-decoder looks more or less like this:
![Generator_structure](https://cdn-images-1.medium.com/max/1600/1*grPpbT-8fwA4twYZkAHwmw.png)

The volumes are there to give you a sense of the shape of the tensor dimensions next to them. The input in this example is a 256x256 image with 3 color channels (red, green, and blue, all equal for a black and white image), and the output is the same.

The generator takes some input and tries to reduce it with a series of encoders (convolution + activation function) into a much smaller representation. The idea is that by compressing it this way we hopefully have a higher level representation of the data after the final encode layer. The decode layers do the opposite (deconvolution + activation function) and reverse the action of the encoder layers.

In order to improve the performance of the image-to-image transform in the paper, the authors used a “U-Net” instead of an encoder-decoder. This is the same thing, but with “skip connections” directly connecting encoder layers to decoder layers:

![U-Net](https://cdn-images-1.medium.com/max/1600/1*kpWvVdQOmbMuX2ls-d78TA.png)

The skip connections give the network the option of bypassing the encoding/decoding part if it doesn’t have a use for it.

These diagrams are a slight simplification. For instance, the first and last layers of the network have no batch norm layer and a few layers in the middle have dropout units.

The Discriminator

The Discriminator has the job of taking two images, an input image and an unknown image (which will be either a target or output image from the generator), and deciding if the second image was produced by the generator or not.

![Discriminator](https://cdn-images-1.medium.com/max/1600/1*-iPXj4C0sCK0UzW1aPzJZg.png)

The structure looks a lot like the encoder section of the generator, but works a little differently. The output is a 30x30 image where each pixel value (0 to 1) represents how believable the corresponding section of the unknown image is. In the pix2pix implementation, each pixel from this 30x30 image corresponds to the believability of a 70x70 patch of the input image (the patches overlap a lot since the input images are 256x256). The architecture is called a “PatchGAN”.

### Training

To train this network, there are two steps: training the discriminator and training the generator.

To train the discriminator, first the generator generates an output image. The discriminator looks at the input/target pair and the input/output pair and produces its guess about how realistic they look. The weights of the discriminator are then adjusted based on the classification error of the input/output pair and the input/target pair.

The generator’s weights are then adjusted based on the output of the discriminator as well as the difference between the output and target image.

![Discriminator_training](https://cdn-images-1.medium.com/max/1200/1*EUUrcoQ9nBGyNzIeqzCDJQ.png)
![Generator_training](https://cdn-images-1.medium.com/max/1200/1*vwCOla_RiuTD6pEEMitTgQ.png)
_Discriminator and Generator training_

## CycleGANs
Original [CycleGAN paper](https://arxiv.org/pdf/1703.10593.pdf)

While PIX2PIX can produce truly magical results, the challenge is in training data. The two image spaces that you wanted to learn to translate between needed to be pre-formatted into a single X/Y image that held both tightly-correlated images. This could be time-consuming, infeasible, or even impossible based on what two image types you were trying to translate between (for instance, if you didn’t have one-to-one matches between the two image profiles). This is where the CycleGAN comes in.

The key idea behind CycleGANs is that they can build upon the power of the PIX2PIX architecture, but allow you to point the model at two discrete, unpaired collections of images. For example, one collection of images, Group X, would be full of sunny beach photos while Group Y would be a collection of overcast beach photos. The CycleGAN model can learn to translate the images between these two aesthetics without the need to merge tightly correlated matches together into a single X/Y training image.

The way CycleGANs are able to learn such great translations without having explicit X/Y training images involves introducing the idea of a full translation cycle to determine how good the entire translation system is, thus improving both generators at the same time.

![cycle-consistency loss](https://cdn-images-1.medium.com/max/1600/0*D5yQU7v0NHXsb1Ep.jpg)

This approach is the clever power that CycleGANs brings to image-to-image translations and how it enables better translations among non-paired image styles.

The original CycleGANs paper, [“Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks”](https://arxiv.org/pdf/1703.10593.pdf), was published by Jun-Yan Zhu, et al.

### Loss functions
The power of CycleGANs is in how they set up the loss function, and use the full cycle loss as an additional optimization target.

As a refresher: we’re dealing with 2 generators and 2 discriminators.

#### Generator Loss
Let’s start with the generator’s loss functions, which consist of 2 parts.

*Part 1*: The generator is successful if fake (generated) images are so good that discriminator can not distinguish those from real images. In other words, the discriminator’s output for fake images should be as close to 1 as possible. In TensorFlow terms, the generator would like to minimize:
```
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# for generator, we mark it as sucessful if the generated image is discrimated as 1
# i.e. whether we successfuly fool the disc
def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)
```

*Part 2*: We need to capture cyclic loss: as we go from one generator back to the original space of images using another generator, the difference between the original image (where we started the cycle) and the cyclic image should be minimized.

In other words, cycle consistency means the result should be close to the original input. For example, if one translates a sentence from English to French, and then translates it back from French to English, then the resulting sentence should be the same as the  original sentence.

In cycle consistency loss,

* Image  𝑋  is passed via generator  𝐺  that yields generated image  𝑌̂  .
* Generated image  𝑌̂   is passed via generator  𝐹  that yields cycled image  𝑋̂  .
* Mean absolute error is calculated between  𝑋  and  𝑋̂  .
  * 𝑓𝑜𝑟𝑤𝑎𝑟𝑑 𝑐𝑦𝑐𝑙𝑒 𝑐𝑜𝑛𝑠𝑖𝑠𝑡𝑒𝑛𝑐𝑦 𝑙𝑜𝑠𝑠:𝑋−>𝐺(𝑋)−>𝐹(𝐺(𝑋))∼𝑋̂  
  * 𝑏𝑎𝑐𝑘𝑤𝑎𝑟𝑑 𝑐𝑦𝑐𝑙𝑒 𝑐𝑜𝑛𝑠𝑖𝑠𝑡𝑒𝑛𝑐𝑦 𝑙𝑜𝑠𝑠:𝑌−>𝐹(𝑌)−>𝐺(𝐹(𝑌))∼𝑌̂  
We multipy the cycle-consistency loss by LAMBDA afterwards.

![Cycle loss](https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/generative/images/cycle_loss.png?raw=1)
```
# It is important that the generator re-crates something similar to the orignial
# therefore a multiplication (e.g 10 times) is applied.
def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  
  return LAMBDA * loss1
```

As shown above, generator  𝐺  is responsible for translating image  𝑋  to image  𝑌 . Identity loss says that, if you fed image  𝑌  to generator  𝐺 , it should yield the real image  𝑌  or something close to image  𝑌 .

𝐼𝑑𝑒𝑛𝑡𝑖𝑡𝑦 𝑙𝑜𝑠𝑠=|𝐺(𝑌)−𝑌|+|𝐹(𝑋)−𝑋|
```
# to make sure the generator_g using the real photo
# output something similar to the real photo
def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss
```

The final generator loss:
```
# Total generator loss = adversarial loss + cycle loss + identity loss
total_gen_g_loss = gen_g_loss + calc_cycle_loss(real_x, cycled_x) + identity_loss(real_x, same_x)
total_gen_f_loss = gen_f_loss + calc_cycle_loss(real_y, cycled_y) + identity_loss(real_y, same_y)
```

#### Discriminator loss
The Discriminator has 2 decisions to make:
1. Real images should be marked as real (recommendation should be as close to 1 as possible)
1. The discriminator should be able to recognize generated images and thus predict 0 for fake images.

```
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
  # if the image is real, disc should identify it as 1
  real_loss = loss_obj(tf.ones_like(real), real)
  
  # if the image is generated, disc sould identify it as 0
  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  # sum up
  total_disc_loss = real_loss + generated_loss

  # the total loss divid by two
  return total_disc_loss * 0.5
  
# Total discrimator loss = 1 for real and 0 for fake
disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
```
