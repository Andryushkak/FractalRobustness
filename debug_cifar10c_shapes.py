import numpy as np

c_path = "data/CIFAR-10-C"  # якщо в тебе інший шлях – підстав свій

corr = "gaussian_noise"
imgs = np.load(f"{c_path}/{corr}.npy")
labels = np.load(f"{c_path}/labels.npy")

print("imgs shape :", imgs.shape)
print("labels shape:", labels.shape)
