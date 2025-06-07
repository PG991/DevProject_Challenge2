import torch
esc50_path = 'data/esc50'

runs_path = 'results'
# sub-epoch (batch-level) progress bar display
disable_bat_pbar = False
#disable_bat_pbar = True

# do not change this block
n_classes = 50
folds = 5
test_folds = [1, 2, 3, 4, 5]
# use only first fold for internal testing
#test_folds = [1]

# sampling rate for waves
sr = 44100
n_mels = 128
hop_length = 512
model_constructor = "AudioResNet18(n_classes=config.n_classes)"

# ###TRAINING
# ratio to split off from training data
val_size = .2 # 20% of training data for validation 
device_id = 0
batch_size = 64
num_workers = 4

# for local Windows or Linux machine
persistent_workers = True
epochs = 200
# early stopping after epochs with no improvement
patience = 25

lr = 1e-3 # initial learning rate
weight_decay = 5e-5
warm_epochs = 8
gamma = 0.8
step_size = 2
mixup_alpha = 0.15

# ### TESTING
# model checkpoints loaded for testing
test_checkpoints = ['terminal.pt']  # ['terminal.pt', 'best_val_loss.pt']
# experiment folder used for testing (result from cross validation training)
test_experiment = 'results/sample-run'