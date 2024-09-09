import pickle

# Path to store hyperparameters in
hparams_path = "hparams.txt"

# Hyperparameters
hparams = {
            "learning_rate": 0.00001,
            "batch_size": 1,
            "test_batch_size": 1,
            "num_epochs": 100, 
            "expected_sample_rate": 44100,
            "stats_path": "db_stats.txt",
            "train_filelist_path": "filelist_train.txt",
            "test_filelist_path": "filelist_test.txt",
            "label_data_path": "/mnt/MP600/data/uncomp/",
            #"label_data_path": "/mnt/MP600/data/small/uncomp/",
            "input_data_path": "/mnt/MP600/data/comp/train/",
            #"input_data_path": "/mnt/MP600/data/small/comp/train/",
            "test_input_data_path": "/mnt/MP600/data/comp/test/",
            #"test_input_data_path": "/mnt/MP600/data/small/comp/test/",
            "checkpoint_path": "/mnt/PC801/declip/results/",
            "augmentation_labels": ["--01--","--10--","--11--","--20--"],
            #"max_time": 655360,
            "max_time": 540672,
            "test_max_time": 1081344,
            "stats_time": 61440000,
            "short_threshold": 0.5,
            "overlap_factor": 0.1,
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": None,
            "transformer": True,
            "preload_weights_path": None,
            #"preload_weights_path": "/mnt/PC801/declip/results/model01.pth",
            #"preload_weights_path": "/mnt/PC801/declip/results/07-28/model01.pth",
            "preload_optimizer_path": None,
            #"preload_optimizer_path": "/mnt/PC801/declip/results/optimizer01.pth",
            #"preload_optimizer_path": "/mnt/PC801/declip/results/07-28/optimizer01.pth",
            "preload_scheduler_path": None,
            #"preload_scheduler_path": "/mnt/PC801/declip/results/scheduler01.pth",
            #"preload_scheduler_path": "/mnt/PC801/declip/results/07-28/scheduler01.pth",
            "preload_scaler_path": None,
            #"preload_scaler_path": "/mnt/PC801/declip/results/scaler01.pth",
            #"preload_scaler_path": "/mnt/PC801/declip/results/07-28/scaler01.pth",
            "n_fft": 4096,
            "hop_length": 256,
            "stats_hop_length": 512,
            "top_db": 106,
            "eps": 0.00000001,
            "scheduler_state": 0,
            "scheduler_factors": [0.1, 0.1, 0.1],
            "scheduler_patiences": [0, 2, 4],
            "save_points": [0.05, 0.25,  0.5],
            "overwrite_preloaded_scheduler_values": False,
            "test_first": False,
            "autoclip": True,
            "multigpu": True, 
            "cuda_device": 0, # Choose which single GPU to use when not using multi-GPU 
            "use_amp": True,
            "use_tf32": True, # GPU must be NVIDIA and Ampere or newer
            "grad_accum": 1, # Higher than 1 doesn't work correctly with multi-GPU
    }

with open(hparams_path, 'wb') as f:
    pickle.dump(hparams, f)

