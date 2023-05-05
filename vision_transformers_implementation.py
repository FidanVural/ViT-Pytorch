import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

np.random.seed(0) # always displays the same random numbers, the number inside the function can be different.
torch.manual_seed(0) # generate random numbers according to the seed value --> if the seed value doesn't change, generating numbers also don't change

batch_size = 128
image_size = (1,28,28)
n_epochs = 10
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(torch.cuda.get_device_name(device)))

transform = ToTensor()

# DATASET

# Download the MNIST dataset
train_set = MNIST(root="datasets/", train=True, download=True, transform=transform)
test_set = MNIST(root="datasets/", train=False, download=True, transform=transform)

# DataLoader --> helps us to load and iterate over elements in a dataset
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Train the network
def training_loop(model):

    loss_func = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        train_loss = []
        for batch in tqdm(train_loader, desc="Training"):
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Predict
            pred = model(x)

            # Calculate the loss
            loss = loss_func(pred, y)

            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = sum(train_loss) / len(train_loss)
        print(f"Epoch {epoch+1} Loss {mean_loss}")

# Check accuracy on test data
def evaluation(model):
    loss_func = CrossEntropyLoss()

    num_correct = 0
    num_samples = 0
    test_loss = []

    model.eval()

    with torch.no_grad():
        correct, total = 0, 0
        test_loss = []
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)

            pred = model(x)

            loss = loss_func(pred, y)
            test_loss.append(loss)

            num_correct += torch.sum(torch.argmax(pred, dim=1) == y).item() # if prediction class equals to the actual class, correct increases.
            num_samples += len(x) # batch size

        mean_loss = sum(test_loss) / len(test_loss)
        print(f"Test loss: {mean_loss:.2f}")
        print(f"Test accuracy: {(num_correct/num_samples):.2f}")

    model.train()


# VISION TRANSFORMERS

# PATCHIFY THE EVERY IMAGE IN A DATASET
def patchify(images, n_patches):
    n, c, h, w = images.shape # (N, 1, 28, 28)

    assert h == w # Patchify method is implemented for square images only

    patches = torch.zeros(n, n_patches**2, h*w*c // n_patches**2) # new shape --> (n, 49, (28*28*1)/49) --> (n, 49, 16) --> her bir resim için 49 satırdan ve her bir satırın 16 uzunluğunda olduğu bir tensor.
    patch_size = h // n_patches # 28 / 7 = 4 --> every patch size 4x4

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] # 4x4'lük patchleri alır.
                #print("\n", patch)
                patches[idx, i*n_patches + j] = patch.flatten() # Linear projection/ Linear embedding
                # Fonksiyona girdi olarak torch.randn ile (N, 1, 28, 28) boyutu oluşturarak ve aşağıdaki yorum satırlarını da kaldırarak sadece bu fonksiyonu çalıştırıp patch'lere ayırma işleminin nasıl olduğu daha iyi incelenebilir.
                # print("Patch: ", patch)
                # print("Patch size: ", patch.shape)
                # print("Patch_flatten: ", patch.flatten())
                # print("\n\n")

    return patches


def positional_encoding(sequence_length, emb_dim): # embedding_dimension --> d --> bu kod için bu değer 8.
    result = torch.zeros(sequence_length, emb_dim)
    for i in range(sequence_length):
        for j in range(emb_dim):
            result[i][j] = np.sin(i / (10000**(j/emb_dim))) if j%2==0 else np.cos(i / (10000**((j-1)/emb_dim))) # j çift ise sin, tek ise cos fonksiyonu kullanılır.

    return result


# MULTI-HEAD SELF ATTENTION
class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()

        self.d = d  # 8
        self.n_heads = n_heads

        assert d % n_heads == 0, f"{d}, {n_heads}'e bölünemez"

        d_head = int(self.d / self.n_heads)

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)]) # n_heads kadar query matrisi oluşturulur.
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)]) # n_heads kadar key matrisi oluşturulur.
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)]) # n_heads kadar value matrisi oluşturulur.

        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):  # sequences => (N, sequence_length, token_dim) => bizim için bu değerler (N, 50, 8)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_map = self.q_mappings[head]
                k_map = self.k_mappings[head]
                v_map = self.v_mappings[head]

                seq = sequence[:, head * self.d_head:(head + 1) * self.d_head]  # head=0 --> [:, 0:4], head=1 --> [:, 4:8]
                q, k, v = q_map(seq), k_map(seq), v_map(seq)  # her biri için shape => torch.Size([50, 4])

                attention = self.softmax(q @ k.T / (
                            self.d_head ** 0.5))  # query ve key değerlerini çarpıp bunu d_head'in kareköküne böleriz.
                seq_result.append(attention @ v)

            result.append(torch.hstack(seq_result))

        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])  # After concatanation => [N,50,8]

# Adding MLP(Multi-layer perceptron) and Residual Connections
class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)

        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


# Classify the images using ViT with shape (N x 1 x 28 x 28)
class MyViT(nn.Module):
    def __init__(self, chw=(1,28,28), n_patches=7, hidden_d=8, n_heads=2, n_blocks=2, out_d=10):
        super(MyViT, self).__init__()
        # Attributes
        self.chw = chw
        self.n_patches = n_patches
        self.hidden_d = hidden_d

        assert chw[1] % n_patches == 0
        assert chw[2] % n_patches == 0

        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches) # (4,4)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1]) # 1x4x4 = 16
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d) # 16 --> 8

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional encoding/embedding
        self.pos_embedding = nn.Parameter((positional_encoding(n_patches**2 + 1, self.hidden_d)).clone().detach())
        self.pos_embedding.requires_grad = False

        # 4) Transformer encoder block
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # 5) Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # 1) Resimleri patch'lere ayırma
        patches = patchify(images, self.n_patches).to(self.pos_embedding.device)

        # 2) Linear mapping uygulama
        tokens = self.linear_mapper(patches) # patches are flattened and map to D dimensions. D = hidden_d --> 16'dan 8'e

        # 3) Classification token ekleme
        # add classification token to every sequence --> (N, 49, 8) yani N tane resmim her biri 49 uzunluklu diziye sahip. Her bir dizinin BAŞINA bu classification token eklenir ve dizilerim artık 50 uzunkuklu olur.
        # new shape --> (N, 50, 8)
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        # 4) Positional encoding/embedding
        pos_emb = self.pos_embedding.repeat(images.shape[0], 1, 1) # We have to repeat the (50, 8) positional encoding matrix N times. N tane resim için tekrar et diyoruz.
        # pos_emb.shape --> (N, 50, 8)
        out = tokens + pos_emb

        # 5) Multi-head self attention
        for block in self.blocks:
            out = block(out)

        # Classification token'ı alma
        cls_token = out[:, 0]

        return self.mlp(cls_token)

# DEFINING MODEL
model = MyViT(image_size, n_patches=7, hidden_d=8, n_heads=2, n_blocks=2, out_d=10).to(device)
# n_patches --> Each image is split into patches
# n_blocks --> Number of blocks
# hidden_d --> Embedding dimension
# n_heads --> Number of heads
# out_d --> Number of classes in the dataset. MNIST has 10 hand-written images [0-9]

training_loop(model)
evaluation(model)

