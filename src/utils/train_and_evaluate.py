import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def create_padding_mask(x: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    return (x != pad_idx).unsqueeze(1).unsqueeze(2)


def _normalize_logits(outputs: torch.Tensor) -> torch.Tensor:
    # Cas attendus: (batch, 1) -> (batch,), (batch,) deja, ou (batch, seq, 1)
    if outputs.dim() == 3:
        logits = outputs[:, 0, :]
        if logits.size(1) == 1:
            logits = logits.squeeze(1)
    else:
        logits = outputs.squeeze(-1)
    return logits


def train(model, loader, optimizer, criterion, device, use_mask: bool = True):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        mask = create_padding_mask(batch_X).to(device) if use_mask else None

        optimizer.zero_grad()
        outputs = model(batch_X, mask)
        logits = _normalize_logits(outputs)

        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        predictions = (torch.sigmoid(logits) > 0.5).float()
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device, use_mask: bool = True):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            mask = create_padding_mask(batch_X).to(device) if use_mask else None

            outputs = model(batch_X, mask)
            logits = _normalize_logits(outputs)

            loss = criterion(logits, batch_y)
            total_loss += loss.item()

            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

    return total_loss / len(loader), correct / total


def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, scheduler=None, use_mask: bool = True):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, use_mask=use_mask)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_mask=use_mask)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f"\nEpoch =========== {epoch + 1}/{epochs}==============\n"
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n"
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    history = {
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs,
    }
    return history


def test_metrics_with_time(model, loader, device, use_mask: bool = True, threshold: float = 0.5):
    """Retourne: test_acc, test_f1, test_auc, eval_time_sec"""
    model.eval()

    all_logits = []
    all_y = []

    start = time.time()

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            batch_mask = create_padding_mask(batch_X).to(device) if use_mask else None
            logits = _normalize_logits(model(batch_X, batch_mask))

            all_logits.append(logits.detach().cpu())
            all_y.append(batch_y.detach().cpu())

    eval_time = time.time() - start

    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_y).numpy()

    y_prob = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (y_prob >= threshold).astype(np.float32)

    test_acc = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, zero_division=0)
    test_auc = roc_auc_score(y_true, y_prob)

    return test_acc, test_f1, test_auc, eval_time


def collect_predictions(model, loader, device, use_mask: bool = True, threshold: float = 0.5):
    """Retourne y_true, y_prob, y_pred (numpy)."""
    model.eval()
    all_logits = []
    all_y = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            batch_mask = create_padding_mask(batch_X).to(device) if use_mask else None
            logits = _normalize_logits(model(batch_X, batch_mask))

            all_logits.append(logits.detach().cpu())
            all_y.append(batch_y.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_y).numpy()
    y_prob = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (y_prob >= threshold).astype(np.float32)

    return y_true, y_prob, y_pred


def run_model_and_collect(
    model,
    name,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    criterion,
    device,
    epochs: int = 10,
    use_mask: bool = True,
):
    """Entraine 'epochs' epochs, evalue sur test_loader et renvoie resume + historique."""
    train_start = time.time()

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train(model, train_loader, optimizer, criterion, device, use_mask=use_mask)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_mask=use_mask)

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"[{name}] ====== Epoch {epoch}/{epochs} ====== | "
            f"train_acc={tr_acc:.4f} | val_acc={val_acc:.4f}\n"
        )

    train_time = time.time() - train_start

    test_acc, test_f1, test_auc, test_eval_time = test_metrics_with_time(
        model, test_loader, device, use_mask=use_mask
    )

    summary = {
        "Model": name,
        "Test Acc": np.round(test_acc, 3),
        "Test F1": np.round(test_f1, 3),
        "Test ROC-AUC": np.round(test_auc, 3),
        "Train Time (s)": np.round(train_time, 3),
        "Test Eval Time (s)": np.round(test_eval_time, 3),
    }

    history = {
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs,
    }

    return summary, history
