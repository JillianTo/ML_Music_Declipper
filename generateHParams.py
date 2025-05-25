import pickle

# Path to store hyperparameters in
hparams_path = "hparams.txt"

# Hyperparameters
hparams = {
            "learning_rate": 0.00001,
            "batch_size": 1,
            "val_batch_size": 2,
            "test_batch_size": 2,
            "num_epochs": 100, 
            "expected_sample_rate": 44100,
            "stats_path": "db_stats.txt",
            "train_filelist_path": "filelist_train.txt",
            "val_filelist_path": "filelist_val.txt",
            "test_filelist_path": "filelist_test.txt",
            "train_label_data_path": "/mnt/XS70/declip/uncomp/train/",
            "val_label_data_path": "/mnt/XS70/declip/uncomp/val/",
            "test_label_data_path": "/mnt/XS70/declip/uncomp/test/",
            "input_data_path": "/mnt/XS70/declip/comp/",
            "checkpoint_path": "/mnt/PC801/declip/results/",
            "augmentation_labels": ["--4-21--","--7-21--","--7-3--","--9-3--"],
            "max_time": 1543500, # If using RoFormer, must be multiple of hop length
            "stats_time": 102400000,
            "short_threshold": 0.5,
            "overlap_factor": 0.1,
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": None,
            "first_out_channels": 64,
            "n_layers": 3, # Recommended: 6 for transformer, 3 for LSTM
            "n_heads": None, # Recommended: 8 for transformer, set to None to use LSTM
            "activation": 'identity',
            "preload_weights_path": None,
            "preload_weights_path": "/mnt/PC801/declip/results/model01.pth",
            #"preload_weights_path": "/mnt/PC801/declip/results/09-20/model01.pth",
            "preload_optimizer_path": None,
            "preload_optimizer_path": "/mnt/PC801/declip/results/optimizer01.pth",
            #"preload_optimizer_path": "/mnt/PC801/declip/results/09-20/optimizer01.pth",
            "preload_scheduler_path": None,
            "preload_scheduler_path": "/mnt/PC801/declip/results/scheduler01.pth",
            #"preload_scheduler_path": "/mnt/PC801/declip/results/09-20/scheduler01.pth",
            "preload_scaler_path": None,
            "preload_scaler_path": "/mnt/PC801/declip/results/scaler01.pth",
            #"preload_scaler_path": "/mnt/PC801/declip/results/09-20/scaler01.pth",
            "preload_train_info_path": "/mnt/PC801/declip/results/train_info.txt",
            "n_fft": 4096,
            "hop_length": 512,
            "stats_hop_length": 1024,
            "loss_n_ffts": (8192, 4096, 2048), 
            "loss_n_mels": (1024, 512, 256),
            "top_db": None,
            "mean_normalization": True,
            "eps": 0.00000001,
            "scheduler_state": 0,
            "scheduler_factors": [0.1, 0.1, 0.1],
            "scheduler_patiences": [0, 2, 4],
            "save_points": [0.005, 0.5, 1],
            "overwrite_preloaded_scheduler_values": False,
            "val_first": False,
            "autoclip": True,
            "multigpu": True, 
            "cuda_device": 1, # Choose which single GPU to use when not using multi-GPU 
            "use_amp": True,
            "use_tf32": True, # GPU must be NVIDIA and Ampere or newer
            "grad_accum": 1, # Higher than 1 not tested with multi-GPU
            "ext_model": None, # None for provided model, 'roformer' for BS-RoFormer, 'mel_roformer' for Mel-Band RoFormer, 'scnet' for SCNet
            "ext_dim": 256, # Dimension value when using external models
            "ext_n_mels": 64, # Number of mel bands to use for Mel-Band RoFormer
            "overwrite_loss": True # False to use external model's loss function
    }

with open(hparams_path, 'wb') as f:
    pickle.dump(hparams, f)

