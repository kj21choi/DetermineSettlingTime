import copy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import torch.nn.functional as F

import DataLoader
import torch
from torch import nn
import seaborn as sns
import datetime
import matplotlib.pyplot as plt

from VAEmodel import VAE, VAE2
from pyts.image import RecurrencePlot

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Hyper parameter
WINDOW_SIZE = 12  # 1 = 5 min
EPOCH = 150
PARAM_INDEX = 8726725
print('PARAM_INDEX:', PARAM_INDEX)

# train & test data
normal_data = DataLoader.NormalDataLoader(PARAM_INDEX)
normal_data.plotFigure()
plt.close()
train_data = normal_data.data

minMaxScaler = MinMaxScaler()
minMaxScaler.fit(train_data.x_data)
X_train, X_valid = normal_data.divide_data(train_data, True)  # train
# X_train, X_valid = normal_data.divide_data(train_data, False)  # validate


# remove some data to reshape
X_train_remove = X_train.shape[0] % WINDOW_SIZE
X_valid_remove = X_valid.shape[0] % WINDOW_SIZE
X_train = X_train[: -X_train_remove]
X_valid = X_valid[: -X_valid_remove]
print("data shape:", X_train.shape, X_valid.shape)
plt.plot(X_train)
plt.plot(X_valid)
plt.show()

# reshape inputs for LSTM [timesteps, samples]
X_train = X_train.reshape(-1, WINDOW_SIZE)  # 12(window)
X_valid = X_valid.reshape(-1, WINDOW_SIZE)
print("data shape:", X_train.shape, X_valid.shape)


# collect mean, std
def collect_mean_std(dataset):
    mean, std = [], []
    for seq in dataset:
        mean.append(seq.mean())
        std.append(seq.std())
    return mean, std


train_mean, train_std = collect_mean_std(X_train)
valid_mean, valid_std = collect_mean_std(X_valid)
train_mean = train_mean + valid_mean
train_std = train_std + valid_std

# 0: left tail, 1: right tail, 2: right tail(std)
mean_std_threshold = [np.percentile(train_mean, 5), np.percentile(train_mean, 95), np.percentile(train_std, 95)]
np.save('./model/'+ str(PARAM_INDEX)+'_mean_std', mean_std_threshold)


# normalize the data
X_train = minMaxScaler.transform(X_train.reshape(-1, 1))
X_valid = minMaxScaler.transform(X_valid.reshape(-1, 1))
print("data shape:", X_train.shape, X_valid.shape)
plt.plot(X_train)
plt.plot(X_valid)
plt.show()

# reshape inputs for LSTM [timesteps, samples]
X_train = X_train.reshape(-1, WINDOW_SIZE)  # 12(window)
X_valid = X_valid.reshape(-1, WINDOW_SIZE)
print("data shape:", X_train.shape, X_valid.shape)


# create dataset
def create_dataset(df):
    dataset = [torch.tensor(s).unsqueeze(0).float() for s in df]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features


def create_recurrent_plots(df):
    rp = RecurrencePlot(dimension=1, percentage=10)
    rp_array = rp.fit_transform(df)
    rp_tensor = [torch.tensor(s).float() for s in rp_array]
    return rp_tensor


train_dataset = create_recurrent_plots(X_train)
val_dataset = create_recurrent_plots(X_valid)


def loss_func(x_hat, x, mu, logvar):
    BCE = F.binary_cross_entropy(x_hat, x)
    # KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1 - logvar)

    return BCE + KLD, BCE, KLD


# criterion = nn.L1Loss(reduction='sum').to(device)
# model
model = VAE2(image_channels=1)
model = model.to(device)

# training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss(reduction='sum').to(device)
# criterion = nn.CrossEntropyLoss()
history = dict(train=[], val=[])

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 10000.0

# early stop
n_epochs_stop = 12
epochs_no_improve = 0

now = datetime.datetime.now()
print(now)
for epoch in range(1, EPOCH + 1):
    model = model.train()

    train_losses = []
    for seq_true in train_dataset:
        optimizer.zero_grad()

        # seq_pred = model(seq_true.unsqueeze(0).unsqueeze(0).to(device))
        recon, mu, logvar = model(seq_true.unsqueeze(0).unsqueeze(0).to(device))
        seq_true = seq_true.to(device)
        # loss, bce, kld = loss_func(recon.squeeze(0).squeeze(0), seq_true, mu, logvar)
        loss = criterion(recon.squeeze(0).squeeze(0), seq_true)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
        for seq_true in val_dataset:
            recon, mu, logvar = model(seq_true.unsqueeze(0).unsqueeze(0).to(device))
            seq_true = seq_true.to(device)
            # loss, bce, kld = loss_func(recon.squeeze(0).squeeze(0), seq_true, mu, logvar)
            loss = criterion(recon.squeeze(0).squeeze(0), seq_true)
            val_losses.append(loss.item())

    train_loss = np.mean(train_losses)  # mean is too small
    val_loss = np.mean(val_losses)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_loss:
        epochs_no_improve = 0
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
    else:
        epochs_no_improve += 1

    if epoch > 5 and epochs_no_improve == n_epochs_stop:
        print('Early stopping!')
        break

    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

now = datetime.datetime.now()
print(now)

model.load_state_dict(best_model_wts)
model.load_state_dict(best_model_wts)
model.eval()

# save model
MODEL_PATH = './model/pH_'+str(PARAM_INDEX)+'_vae_model_12.pth'
torch.save(model, MODEL_PATH)


# load model
model = torch.load('./model/pH_'+str(PARAM_INDEX)+'_vae_model_12.pth')
model = model.to(device)

train_mean_std_load = np.load('./model/'+str(PARAM_INDEX)+'_mean_std.npy')


# choosing a threshold
def validate(model, dataset):
    losses = []
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            recon, mu, logvar = model(seq_true.unsqueeze(0).unsqueeze(0).to(device))

            # plt.figure(figsize=(8, 8))
            # plt.imshow(seq_true.cpu().numpy())
            # plt.figure(figsize=(8, 8))
            # plt.imshow(recon.squeeze(0).squeeze(0).cpu().numpy())

            seq_true = seq_true.to(device)
            # loss, _, _ = loss_func(recon.squeeze(0).squeeze(0), seq_true, mu, logvar)
            loss = criterion(recon.squeeze(0).squeeze(0), seq_true)
            losses.append(loss.item())
    return losses

plt.close()
correct_pred = validate(model, train_dataset)
sns.distplot(correct_pred, bins=50, kde=True)

correct_pred = validate(model, val_dataset)
sns.distplot(correct_pred, bins=50, kde=True)
plt.show()
THRESHOLD = np.percentile(correct_pred, 95)
# THRESHOLD = np.median(correct_pred)

correct = sum(l <= THRESHOLD for l in correct_pred)
print(f'Correct normal predictions: {correct}/{len(val_dataset)}')
# Correct normal predictions: 697/734
print(THRESHOLD)


def predict(model, data):
    with torch.no_grad():
        model = model.eval()
        for seq_true in data:
            recon, mu, logvar = model(seq_true.unsqueeze(0).unsqueeze(0).to(device))

            # plt.figure(figsize=(8,8))
            # plt.imshow(seq_true.cpu().numpy())
            # plt.figure(figsize=(8, 8))
            # plt.imshow(recon.squeeze(0).squeeze(0).cpu().numpy())

            seq_true = seq_true.to(device)
            # loss, _, _ = loss_func(recon.squeeze(0).squeeze(0), seq_true, mu, logvar)
            loss = criterion(recon.squeeze(0).squeeze(0), seq_true)
    return loss


# inference data
unstable_data = DataLoader.UnstableDataLoader(PARAM_INDEX)
eval_data = unstable_data.data.x_data
stableStarted = 4988

for i in range(len(eval_data) - WINDOW_SIZE):
    mean = eval_data[i: i + WINDOW_SIZE].mean()
    std = eval_data[i: i + WINDOW_SIZE].std()
    # normalize
    normalized_eval_data = minMaxScaler.transform(eval_data[i: i + WINDOW_SIZE])
    # normalized_eval_data = normalized_eval_data.reshape(-1, WINDOW_SIZE)
    normalized_eval_data = normalized_eval_data.reshape(-1, WINDOW_SIZE)
    data = create_recurrent_plots(normalized_eval_data)
    loss = predict(model, data)

    if loss < THRESHOLD:
        print(f'LML/UML/USL: {np.around(train_mean_std_load, 3)} mean:{np.around(mean, 3)} std: {np.around(std, 4)}')
        print(f'threshold({np.around(THRESHOLD, 2)}) vs. loss({np.around(loss.cpu().numpy(), 2)})')
        if train_mean_std_load[0] < mean.item() < train_mean_std_load[1] and std.item() < train_mean_std_load[2]:
            stableStarted = i
            break
        else:
            continue

print("unstable time:", unstable_data.data.time_axis['act_time'].get(0))
print("stable time:", unstable_data.data.time_axis['act_time'].get(stableStarted))
print("decision time:", unstable_data.data.time_axis['act_time'].get(stableStarted + WINDOW_SIZE - 1))
