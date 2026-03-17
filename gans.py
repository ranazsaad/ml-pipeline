# -*- coding: utf-8 -*-
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# MLflow Experiment
# ----------------------
mlflow.set_experiment("Assignment3_RanaSaad")

# ----------------------
# Device
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------
# Load Dataset
# ----------------------
data = pd.read_csv("dataset.csv")
print("Shape of dataset:", data.shape)

images = data.iloc[:, 1:].values.astype(np.float32)
images = images.reshape(-1, 1, 28, 28)
images = torch.tensor(images, dtype=torch.float32) / 255.0

print("Tensor shape:", images.shape)
print("Min value:", images.min().item(), "Max value:", images.max().item())

dataset = torch.utils.data.TensorDataset(images)
loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

# ----------------------
# GAN Model
# ----------------------
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Sigmoid()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(-1, 1, 28, 28)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

# ----------------------
# Hyperparameters
# ----------------------
z_dim = 50
epochs = 40
lr = 0.001
batch_size = 128

generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# ----------------------
# MLflow Run
# ----------------------
with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.set_tag("student_id", "RanaSaad")

    g_losses = []
    d_losses = []

    # ----------------------
    # Training Loop
    # ----------------------
    for epoch in range(epochs):
        for batch_idx, (real_imgs,) in enumerate(loader):
            real_imgs = real_imgs.to(device)
            current_batch_size = real_imgs.size(0)

            real_labels = torch.ones(current_batch_size, 1).to(device) * 0.9
            fake_labels = torch.zeros(current_batch_size, 1).to(device) + 0.1

            # Train Discriminator
            optimizer_D.zero_grad()
            real_output = discriminator(real_imgs)
            d_real_loss = criterion(real_output, real_labels)

            z = torch.randn(current_batch_size, z_dim).to(device)
            fake_imgs = generator(z)
            fake_output = discriminator(fake_imgs.detach())
            d_fake_loss = criterion(fake_output, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(current_batch_size, z_dim).to(device)
            fake_imgs = generator(z)
            fake_output = discriminator(fake_imgs)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizer_G.step()

            # Track losses for plotting
            if batch_idx % 10 == 0:
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

        # Print epoch summary
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("generator_loss", g_loss.item(), step=epoch)
        mlflow.log_metric("discriminator_loss", d_loss.item(), step=epoch)

    # ----------------------
    # Log models
    # ----------------------
    mlflow.pytorch.log_model(generator, "generator_model")
    mlflow.pytorch.log_model(discriminator, "discriminator_model")

# ----------------------
# Visualization
# ----------------------
generator.eval()
with torch.no_grad():
    z = torch.randn(16, z_dim).to(device)
    generated_imgs = generator(z).cpu()

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated_imgs[i][0], cmap='gray')
    ax.axis("off")
plt.suptitle("Generated Images After Training")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Iterations (x10)')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Losses')
plt.show()