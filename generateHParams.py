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
            "max_time": 1024000,
            "stats_time": 102400000,
            "short_threshold": 0.5,
            "overlap_factor": 0.1,
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": None,
            "first_out_channels": 64,
            "n_layers": 6, # Recommended: 6 for transformer, 3 for LSTM
            "n_heads": 8, # Set to None to use LSTM
            "preload_weights_path": None,
            #"preload_weights_path": "/mnt/PC801/declip/results/model01.pth",
            #"preload_weights_path": "/mnt/PC801/declip/results/09-20/model01.pth",
            "preload_optimizer_path": None,
            #"preload_optimizer_path": "/mnt/PC801/declip/results/optimizer01.pth",
            #"preload_optimizer_path": "/mnt/PC801/declip/results/09-20/optimizer01.pth",
            "preload_scheduler_path": None,
            #"preload_scheduler_path": "/mnt/PC801/declip/results/scheduler01.pth",
            #"preload_scheduler_path": "/mnt/PC801/declip/results/09-20/scheduler01.pth",
            "preload_scaler_path": None,
            #"preload_scaler_path": "/mnt/PC801/declip/results/scaler01.pth",
            #"preload_scaler_path": "/mnt/PC801/declip/results/09-20/scaler01.pth",
            "n_fft": 4096,
            "hop_length": 512,
            "stats_hop_length": 1024,
            "loss_n_ffts": [1024, 2048, 4096, 8192], 
            "n_mels": [64, 128, 256, 512],
            #"top_db": 106,
            "top_db": 160,
            "mean_normalization": False,
            "eps": 0.00000001,
            "scheduler_state": 0,
            #"scheduler_factors": [(1/3), 0.1, 0.1, 0.1],
            "scheduler_factors": [0.1, 0.1, 0.1],
            "scheduler_patiences": [0, 2, 4],
            #"save_points": [0.025, 0.05, 0.5, 1],
            "save_points": [0.025, 0.5, 1],
            "overwrite_preloaded_scheduler_values": False,
            "val_first": False,
            "autoclip": True,
            "multigpu": True, 
            "cuda_device": 0, # Choose which single GPU to use when not using multi-GPU 
            "use_amp": True,
            "use_tf32": True, # GPU must be NVIDIA and Ampere or newer
            "grad_accum": 1, # Higher than 1 not tested with multi-GPU
    }

with open(hparams_path, 'wb') as f:
    pickle.dump(hparams, f)

