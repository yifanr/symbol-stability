from network import CVQVAE, VQVAE
import equinox as eqx
import jax
import jax.numpy as jnp
import distrax

@eqx.filter_value_and_grad
def update_VQ(model: VQVAE, input):
    # input: array of shape (n, d), where d is the size of expected inputs.

    encodings = eqx.filter_vmap(model.encode)(input)
    quantized, indices = eqx.filter_vmap(model.quantize)(encodings)
    codebook_vectors = model.embedding[indices]
    reconstructions = eqx.filter_vmap(model.decode)(quantized)

    e_latent_loss = jnp.mean((jax.lax.stop_gradient(codebook_vectors) - encodings) ** 2)
    q_latent_loss = jnp.mean((jax.lax.stop_gradient(encodings) - codebook_vectors) ** 2)
    reconstruction_loss = jnp.mean((reconstructions - input) ** 2)
    
    return reconstruction_loss + e_latent_loss + q_latent_loss * 0.9

@eqx.filter_value_and_grad
def update_CVQ(model: CVQVAE, input):
    # input: array of shape (n, d), where d is the size of expected inputs.

    encodings = eqx.filter_vmap(model.encode)(input)
    quantized, indices = eqx.filter_vmap(model.quantize)(encodings)
    codebook_vectors = jnp.moveaxis(model.embedding[indices], -1, -3)
    reconstructions = eqx.filter_vmap(model.decode)(quantized)

    e_latent_loss = jnp.mean((jax.lax.stop_gradient(codebook_vectors) - encodings) ** 2)
    q_latent_loss = jnp.mean((jax.lax.stop_gradient(encodings) - codebook_vectors) ** 2)
    reconstruction_loss = jnp.mean((reconstructions - input) ** 2)
    
    return reconstruction_loss + e_latent_loss + q_latent_loss * 0.9
