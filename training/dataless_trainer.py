#!/usr/bin/env python3
import sys
from dataset.qaoa_dataset import create_dataloader
from models.qaoa_predictor import *
from dataless_train_funcs import *
import argparse

from enum import Enum

class ModelType(Enum):
    FC = "fc"
    GNN = "gnn"
    LSTM = "lstm"
    CNN = "cnn"

def get_model(model_str: str, n_layers: int, n_nodes: int) -> QAOAPredictor:
    model_type = ModelType(model_str)
    if model_type == ModelType.FC:
        return QAOAPredictorFC(num_nodes=n_nodes, p=n_layers)
    elif model_type == ModelType.GNN:
        return QAOAPredictorGNN(p=n_layers)
    elif model_type == ModelType.LSTM:
        return QAOAPredictorLSTM(p=n_layers)
    elif model_type == ModelType.CNN:
        return QAOAPredictorCNN(p=n_layers)
    else:
        raise Exception(f"Unknown model type {model_type}")

def main(argv, stdout=sys.stdout, stderr=sys.stderr) -> int:
    arg_parser = argparse.ArgumentParser(description="Train QAOA predictor model without data")
    arg_parser.add_argument("-m", "--model", type=str, choices=[enum.value for enum in ModelType], required=True, help="Model type")
    arg_parser.add_argument("-n", "--n_nodes", type=int, default=-1, help="Number of nodes (this will affect the dataset used)")
    arg_parser.add_argument("-p", "--n_layers", type=int, required=True, help="Number of layers (this will affect the dataset used)")
    arg_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    arg_parser.add_argument("--batch_size", type=int, default=1 , help="Batch size")
    arg_parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    arg_parser.add_argument("--split_ratio", type=float, default=0.8, help="Train/test split ratio")
    args = arg_parser.parse_args(argv)
    model = get_model(args.model, args.n_layers, args.n_nodes)
    dataless_train(model=model, epochs=args.epochs, lr=args.lr, split_ratio=args.split_ratio)
    return 0
    
    
if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) == 1:
        for model_type in ModelType:
            print(f"Training {model_type.value}")
            exit(main([
                "--model", model_type.value,
                "--dataset_name", "ATLAS",
                "-p", "3",
                "--batch_size", "1",
                "--epochs", "1",
            ]))
    exit(main(sys.argv[1:]))