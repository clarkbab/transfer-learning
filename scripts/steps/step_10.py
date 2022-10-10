from mymi.loaders import get_loader_n_train
from mymi.training.segmenter import train_segmenter
from mymi.regions import RegionNames

n_trains = [5, 10, 20, 50, 100, 200, 'all']
test_folds = [0, 1, 2, 3, 4]

n_train_epochs = {
    5: 900,             # BP_L/R @ n=5 took this long to plateau.
    10: 450,            # BP_L/R, L_L/R @ n=10.
    20: 300,            # BP_L/R, ON_L/R @ n=20.
    'default': 150      # All other models.
}

# Train localiser/segmenter network per region - in reality this would be performed across multiple machines.
for region in RegionNames:
    for test_fold in test_folds:
        for n_train in n_trains:
            # Skip if we don't have that many 'n_train' samples.
            n_train_max = get_loader_n_train('INST', region, test_fold=test_fold)
            if type(n_train) == int and n_train >= n_train_max:
                continue

            # Get number of epochs.
            n_epochs = n_train_epochs[n_train] if n_train in n_train_epochs else n_train_epochs['default']

            # Train institutional network.
            train_segmenter('INST', region, f'segmenter-{region}', 'inst-fold-{test_fold}-samples-{n_train}', n_epochs=n_epochs)
