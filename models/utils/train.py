import os
import json
import torch
from torch.utils.data import DataLoader
from models.chatbot import Chatbot
from models.dataset import ChatbotDataset, pad_collate_fn
from models.utils.tokenizer import Tokenizer
from config import Config


def train(model, data_loader, optimizer, criterion, device):
    """
    Trains the input model on the input data.

    Args:
        model (nn.Module): The model to train.
        data_loader (DataLoader): The DataLoader object containing the training data.
        optimizer (Optimizer): The optimizer to use for training.
        criterion (nn.Module): The loss function to use for training.
        device (torch.device): The device to use for training (cpu or gpu).

    Returns:
        float: The average training loss.
    """
    model.train()
    train_loss = 0
    for batch in data_loader:
        src, tgt, src_len, tgt_len = batch
        src = src.to(device)
        tgt = tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt, src_len, tgt_len)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        tgt = tgt[1:].view(-1)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(data_loader)


def evaluate(model, data_loader, criterion, device):
    """
    Evaluates the input model on the input data.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): The DataLoader object containing the evaluation data.
        criterion (nn.Module): The loss function to use for evaluation.
        device (torch.device): The device to use for evaluation (cpu or gpu).

    Returns:
        float: The average evaluation loss.
    """
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            src, tgt, src_len, tgt_len = batch
            src = src.to(device)
            tgt = tgt.to(device)
            output = model(src, tgt, src_len, tgt_len, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            tgt = tgt[1:].view(-1)
            loss = criterion(output, tgt)
            eval_loss += loss.item()
    return eval_loss / len(data_loader)


def train_loop(model, train_data_loader, dev_data_loader, optimizer, criterion, device):
    """
    Trains the model on the training set and evaluates it on the development set.

    Args:
        model (Chatbot): The model to train and evaluate.
        train_data_loader (DataLoader): The DataLoader for the training set.
        dev_data_loader (DataLoader): The DataLoader for the development set.
        optimizer (Optimizer): The optimizer to use for training.
        criterion (Criterion): The criterion to use for computing the loss.
        device (torch.device): The device to use for training.

    Returns:
        tuple: A tuple of the training loss, development loss, and development accuracy.
    """
    model.train()

    train_loss = 0
    for batch_idx, batch in enumerate(train_data_loader):
        input_seq, input_lengths, target_seq, target_lengths = batch
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)

        optimizer.zero_grad()

        output, _, _ = model(input_seq, input_lengths, target_seq)

        # Remove the <sos> token from the target sequence
        target_seq = target_seq[:, 1:]

        loss = criterion(output.view(-1, model.output_size), target_seq.reshape(-1))
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss /= len(train_data_loader)

    dev_loss, dev_acc = evaluate(model, dev_data_loader, criterion, device)

    return train_loss, dev_loss, dev_acc
