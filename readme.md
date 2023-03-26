The official JAX codebase for the paper "High Fidelity Image Synthesis with Deep VAEs in Latent Space". 

This paper shows how VAEs can be successfully applied to image generation of complex datasets like ImageNet. By training them in the latent space of a dimensionality-reducing autoencoder, the VAE learns to ignore perceptually irrelevant information, letting them focus more on image structure. 

This repository is designed to be fairly minimal and hackable, and reflects the codebase used during our actual experiments. 
For convenience, we have also provided a PyTorch implementation at https://github.com/ericl122333/latent-vae. 

Limitations of this repository:

Our Jax repository does NOT provide an interface with the first-stage autoencoders. This is because we use the pretrained ones from github.com/CompVis/latent-diffusion, which was implemented in PyTorch. For this reason, if you are interested only in generating samples, we recommend using the PyTorch repository.

Multi-node training is currently not supported. 


The VAE weights are stored on google drive.  
ImageNet: https://drive.google.com/file/d/1ziFTKcCGGBSKQ1klkksWoFuJUocQp84e/view?usp=share_link
LSUN Bedroom: https://drive.google.com/file/d/1oGic2H2IWs53L8DYdwaS_ChQbywKR2NT/view?usp=share_link
LSUN Church: https://drive.google.com/file/d/1vQSswx94_xr9fjPEnenQnirjiyebV5iM/view?usp=share_link

