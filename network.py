import torch
import jax
import jax.numpy as jnp
import equinox as eqx

class CGSAE(torch.nn.Module):
    def __init__(self, in_channels, embed_size, ratio=0.5):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, 5, 1),
            torch.nn.SELU(),
            torch.nn.Conv2d(32, 32, 5, 1),
            torch.nn.SELU(),
            torch.nn.Conv2d(32, embed_size, 5, 1)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(embed_size, 32, 5, 1),
            torch.nn.SELU(),
            torch.nn.ConvTranspose2d(32, 32, 5, 1),
            torch.nn.SELU(),
            torch.nn.ConvTranspose2d(32, in_channels, 5, 1),
        )

    def sample_logistic_noise(self, x):
        U = torch.distributions.Uniform(torch.zeros_like(x), torch.ones_like(x))
        sample = U.sample()
        return torch.log(sample) - torch.log(1 - sample)
    
    def encode(self, input, temperature=0.5, testing=False):
        x = input
        for layer in self.encoder:
            x = layer(x)
        if testing:
            return torch.where(x > 0, 1, 0)
            # return torch.sigmoid(x / temperature)
        else:
            y = x + self.sample_logistic_noise(x)

            return torch.sigmoid(y / temperature)
        
    def decode(self, input):
        x = input
        for layer in self.decoder:
            x = layer(x)

        return x
    
    def forward(self, input, temperature=0.5, testing=False):
        encoded = self.encode(input, temperature, testing)
        decoded = self.decode(encoded)
        return decoded
    
class GSAE(torch.nn.Module):
    def __init__(self, in_size, embed_size, ratio=0.5):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_size, 256),
            torch.nn.SELU(),
            torch.nn.Linear(256, 128),
            torch.nn.SELU(),
            torch.nn.Linear(128, embed_size)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embed_size, 128),
            torch.nn.SELU(),
            torch.nn.Linear(128, 256),
            torch.nn.SELU(),
            torch.nn.Linear(256, in_size),
        )

    def sample_logistic_noise(self, x):
        U = torch.distributions.Uniform(torch.zeros_like(x), torch.ones_like(x))
        sample = U.sample()
        return torch.log(sample) - torch.log(1 - sample)
    
    def encode(self, input, temperature=0.5, testing=False):
        x = input
        for layer in self.encoder:
            x = layer(x)
        if testing:
            return torch.where(x > 0, 1, 0)
            # return torch.sigmoid(x / temperature)
        else:
            y = x + self.sample_logistic_noise(x)

            return torch.sigmoid(y / temperature)
        
    def decode(self, input):
        x = input
        for layer in self.decoder:
            x = layer(x)

        return x
    
    def forward(self, input, temperature=0.5, testing=False):
        encoded = self.encode(input, temperature, testing)
        decoded = self.decode(encoded)
        return decoded
    
class CVQVAE(eqx.Module):
    in_channels: int
    num_embeddings: int
    embedding_size: int
    embedding: eqx.nn.Embedding
    encoder: list
    decoder: list

    def __init__(self, in_channels, embedding_size, num_embeddings, key):
        self.in_channels = in_channels
        self.embedding_size = embedding_size
        self.num_embeddings = num_embeddings
        keys = jax.random.split(key, 7)
        self.encoder = [
            eqx.nn.Conv2d(in_channels, 32, 5, stride=1, key=keys[0]),
            jax.nn.selu,
            eqx.nn.Conv2d(32, 32, 5, stride=1, key=keys[1]),
            jax.nn.selu,
            eqx.nn.Conv2d(32, embedding_size, 5, stride=1, key=keys[2])
        ]
        self.decoder = [
            eqx.nn.ConvTranspose2d(embedding_size, 32, 5, stride=1, key=keys[3]),
            jax.nn.selu,
            eqx.nn.ConvTranspose2d(32, 32, 5, stride=1, key=keys[4]),
            jax.nn.selu,
            eqx.nn.ConvTranspose2d(32, in_channels, 5, stride=1, key=keys[5])
        ]
        self.embedding = jax.random.normal(keys[6], (num_embeddings, embedding_size))
        self.embedding /= 1000

    def encode(self, input):
        x = input
        for layer in self.encoder:
            x = layer(x)
        return x
    
    def decode(self, encoding):
        x = encoding
        for layer in self.decoder:
            x = layer(x)
        return x
        
    def quantize(self, input):
        # Distances for every pair of codebook vector and encoding
        input = jnp.moveaxis(input, 0, 2)
        # input: h x w x c
        # embeddings: n x c
        distances = jnp.sum((self.embedding[:,None,None,:] - input) ** 2, axis=-1)
        # similarities = calculate_similarity(self.embedding.weight, input)
        encoding_indices = jnp.argmin(distances, axis=0)
        # encoding_index = jnp.argmax(similarities)
        encoding = jax.nn.one_hot(encoding_indices, self.num_embeddings)

        quantized = input + jax.lax.stop_gradient(self.embedding[encoding_indices] - input)

        return jnp.moveaxis(quantized, 2, 0), encoding_indices
    
    
class VQVAE(eqx.Module):
    in_dim: int
    num_embeddings: int
    embedding_size: int
    embedding: eqx.nn.Embedding
    encoder: list
    decoder: list

    def __init__(self, in_dim, embedding_size, num_embeddings, key):
        self.in_dim = in_dim
        self.embedding_size = embedding_size
        self.num_embeddings = num_embeddings
        keys = jax.random.split(key, 7)
        dense_size = 2 * (in_dim + embedding_size)
        self.encoder = [
            eqx.nn.Linear(self.in_dim, dense_size, key=keys[0]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, dense_size, key=keys[1]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, self.embedding_size, key=keys[2])
        ]
        self.decoder = [
            eqx.nn.Linear(self.embedding_size, dense_size, key=keys[3]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, dense_size, key=keys[4]),
            jax.nn.selu,
            eqx.nn.Linear(dense_size, self.in_dim, key=keys[5])
        ]
        # self.embedding = eqx.nn.Embedding(num_embeddings, embedding_size, key=keys[6])
        self.embedding = jax.random.normal(keys[6], (num_embeddings, embedding_size))
        self.embedding /= 1000

    def encode(self, input):
        x = input
        for layer in self.encoder:
            x = layer(x)
        return x
    
    def decode(self, encoding):
        x = encoding
        for layer in self.decoder:
            x = layer(x)
        return x
        
    def quantize(self, input):
        distances = jnp.sum((self.embedding - input) ** 2, axis=1)
        # similarities = calculate_similarity(self.embedding.weight, input)
        encoding_index = jnp.argmin(distances)
        # encoding_index = jnp.argmax(similarities)
        encoding = jax.nn.one_hot(encoding_index, self.num_embeddings)

        return input + jax.lax.stop_gradient(self.embedding[encoding_index] - input), encoding_index
    