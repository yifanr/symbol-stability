import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
from network import GSAE
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
TOTAL_STEPS = 64000
STEPS = int(TOTAL_STEPS/BATCH_SIZE)
VISUALIZE = True
EMBEDDING_DIM = 50
MAX_TEMP = 5
MIN_TEMP = 0.2
RATIO = 0.5

normalise_data = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_dataset = torchvision.datasets.MNIST(
    "MNIST",
    train=True,
    download=True,
    transform=normalise_data,
)
test_dataset = torchvision.datasets.MNIST(
    "MNIST",
    train=False,
    download=True,
    transform=normalise_data,
)
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)
model = GSAE(28*28, EMBEDDING_DIM)
# model = AE(28*28, 8, jax.random.PRNGKey(42))
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    

# Loop over our training dataset as many times as we need.
def infinite_trainloader():
    while True:
        yield from trainloader

def loss_fn(x, encodings, reconstructions):
    eps = 0.000001
    kld = (encodings * (torch.log(eps + encodings / RATIO))) + ((1 - encodings) * (torch.log(eps + (1 - encodings) / (1 - RATIO))))
    elbo = torch.mean((reconstructions - x) ** 2) - torch.mean(kld)
    return elbo

for step, (x, y) in zip(range(STEPS), infinite_trainloader()):
    x = torch.reshape(x, (x.shape[0], 28*28))
    temperature = MAX_TEMP * (MIN_TEMP/MAX_TEMP)**(step/STEPS)
    encodings = model.encode(x, temperature)
    reconstructions = model.decode(encodings)
    loss = loss_fn(x, encodings, reconstructions)
    
    optim.zero_grad()
    loss.backward()
    optim.step()

    if (step % 100) == 0 or (step == STEPS - 1):
        print(
            f"{step=}, train_loss={loss.item()}, ratio={torch.mean(encodings).item()}"
        )

    if VISUALIZE and ((step % 200) == 0 or (step == STEPS - 1)):
        #TODO
        pass

if VISUALIZE:
    for i in range(20):
        dummy_x, dummy_y = next(iter(testloader))
        dummy_x = dummy_x[0]
        pixels = dummy_x.reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.show()
        encoded = model.encode(dummy_x.reshape(1,28*28))
        reconstruction = model.decode(encoded)
        pixels = reconstruction.detach().numpy().reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.show()

total_num_changed = 0
total = 0
n = 100
k = 10
for i in range(n):
    x, y = next(iter(testloader))
    x = torch.reshape(x, (x.shape[0], 28*28))
    original_encodings = model.encode(x, MIN_TEMP, testing=True)
    for j in range(k):
        noise = torch.normal(0, 0.15, size=x.shape)
        noisy = x + noise
        encodings = model.encode(noisy, MIN_TEMP, testing=True)
        # if (i*j == 0):
        #     pixels = x[0].reshape((28, 28))
        #     plt.imshow(pixels, cmap='gray')
        #     plt.show()
        #     pixels = noisy[0].reshape((28, 28))
        #     plt.imshow(pixels, cmap='gray')
        #     plt.show()
        total += torch.numel(encodings)
        total_num_changed += torch.sum(torch.abs(encodings - original_encodings))

print("Variables per image:")
print(total_num_changed/n/k/BATCH_SIZE)
print(total/n/k/BATCH_SIZE)
print("Ratio changed")
print(total_num_changed/total)