#!/bin/bash
source ../venv/bin/activate
NCCL_P2P_DISABLE=1 python3.11 train.py
