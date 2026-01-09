import argparse
import collections
import io
import os
import sklearn
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import olive

# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train note classifier with partial supervision.")

    parser.add_argument("run", help="Name of run")

    parser.add_argument("--db", type=str, default="db", help="Database directory")
    parser.add_argument("--log", type=str, default="log", help="Log directory")

    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--num-hops", type=int, default=16)

    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[10,20],
        help="Epochs per training phase, e.g. --epochs 10 5 20",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--learn-rate", type=float, default=1e-3)
    parser.add_argument("--partials-weight", type=float, default=1.0)
    parser.add_argument("--p_unknown", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    parser.add_argument("--num-octaves", type=int, default=8)
    parser.add_argument("--num-partials", type=int, default=8)

    return parser.parse_args()

# ------------------------------------------------------------

def main():
    args = parse_args()

    summary_dir = os.path.join(args.log, args.run)
    os.makedirs(summary_dir, exist_ok=True)

    feature_data = olive.load_features(args)

    stream_ids = sorted(set(s.stream_id for s in feature_data))
    num_instr = max([s.instr_id for s in feature_data]) + 1

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    kf = sklearn.model_selection.KFold(n_splits=args.folds, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(stream_ids)):
        print(f"\n[Fold {fold + 1}/{args.folds}]")

        writer = SummaryWriter(log_dir=os.path.join(summary_dir, f"fold_{fold + 1:02d}"))

        train_streams = {stream_ids[i] for i in train_idx}
        test_streams  = {stream_ids[i] for i in test_idx}

        train_dataset = [s for s in feature_data if s.stream_id in train_streams]
        test_dataset  = [s for s in feature_data if s.stream_id in test_streams]

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

        model = olive.OLIVE(
            num_instr,
            freq_bins=args.num_octaves * 12,
            num_octaves=args.num_octaves,
            num_partials=args.num_partials
        ).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

        ce_loss = torch.nn.CrossEntropyLoss()
        huber_loss = torch.nn.SmoothL1Loss(reduction="none")

        # ----------------------------------------------------
        # Training loop
        # ----------------------------------------------------
        n_phases = len(args.epochs)
        n_epochs = sum(args.epochs)
        for phase, epochs in enumerate(args.epochs):
            epochs_before = sum(args.epochs[:phase])
            if n_phases == 1:
                w_part = args.partials_weight
            else:
                w_part = (phase / (n_phases - 1)) * args.partials_weight

            for i in range(epochs):
                epoch = epochs_before + i
                model.train()
                total_loss = 0.0
                total_samples = 0.0

                instr_state_table = {}

                for X_batch, octave_target, pitch_target, partials_target, partials_mask, instr_id, _ in train_loader:
                    X_batch = X_batch.to(args.device)
                    octave_target = octave_target.to(args.device)
                    pitch_target = pitch_target.to(args.device)
                    partials_target = partials_target.to(args.device)
                    partials_mask = partials_mask.to(args.device)
                    instr_id = instr_id.to(args.device)

                    states = []
                    unknown = []
                    for iid in instr_id.tolist():
                        if iid not in instr_state_table:
                            instr_state_table[iid] = model.init_instr_state(1, args.device, instr_id=iid)
                        drop = (torch.rand(1).item() < args.p_unknown)
                        if drop:
                            states.append(model.init_instr_state(1, args.device, instr_id=None))
                        else:
                            states.append(instr_state_table[iid])
                        unknown.append(drop)
                        
                    instr_state = torch.cat(states, dim=0)  # (B, voicing_dim)

                    optimizer.zero_grad()

                    octave_logits, pitch_logits, partials_pred, instr_state_next = model(X_batch, instr_state)

                    # 3. Aggregate updates per instrument
                    accum = collections.defaultdict(list)
                    for i, iid in enumerate(instr_id.tolist()):
                        if not unknown[i]:
                            accum[iid].append(instr_state_next[i:i+1])

                    # 4. Reduce
                    for iid, updates in accum.items():
                        instr_state_table[iid] = torch.mean(
                            torch.cat(updates, dim=0),
                            dim=0,
                            keepdim=True,
                        ).detach()

                    loss = (ce_loss(octave_logits, octave_target) + ce_loss(pitch_logits, pitch_target))

                    # ----- Masked partial regression loss -----
                    partial_loss = huber_loss(partials_pred, partials_target)
                    partial_loss = partial_loss * partials_mask
                    denom = partials_mask.sum().clamp(min=1)
                    partial_loss = partial_loss.sum() / denom

                    loss = loss + w_part * partial_loss

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * X_batch.size(0)
                    total_samples += X_batch.size(0)

                # ------------------------------------------------
                # Evaluation
                # ------------------------------------------------

                model.eval()
                correct_oct = correct_pc = correct_both = total = 0
                all_true = []
                all_pred = []

                total_abs_error = 0.0
                total_count = 0

                abs_error_per_partial = torch.zeros(model.num_partials, device=args.device)
                count_per_partial = torch.zeros(model.num_partials, device=args.device)

                with torch.no_grad():
                    for X_batch, octave_target, pitch_target, partials_target, partials_mask, _, _ in test_loader:
                        X_batch = X_batch.to(args.device)
                        octave_target = octave_target.to(args.device)
                        pitch_target = pitch_target.to(args.device)
                        partials_target = partials_target.to(args.device)
                        partials_mask = partials_mask.to(args.device)

                        B = X_batch.size(0)
                        instr_state = model.init_instr_state(B, args.device, instr_id=None)  # UNKNOWN

                        octave_logits, pitch_logits, partials_pred, _ = model(X_batch, instr_state)

                        pred_oct = octave_logits.argmax(dim=1)
                        pred_pc  = pitch_logits.argmax(dim=1)

                        correct_oct += (pred_oct == octave_target).sum().item()
                        correct_pc  += (pred_pc == pitch_target).sum().item()
                        correct_both += (
                            (pred_oct == octave_target)
                            & (pred_pc == pitch_target)
                        ).sum().item()

                        total += octave_target.size(0)

                        true_full = octave_target * 12 + pitch_target
                        pred_full = pred_oct * 12 + pred_pc

                        all_true.extend(true_full.cpu().numpy())
                        all_pred.extend(pred_full.cpu().numpy())

                        # ----- partial offsets (accumulate) -----

                        err = torch.abs(partials_pred - partials_target)
                        masked_err = err * partials_mask

                        total_abs_error += masked_err.sum()
                        total_count += partials_mask.sum()

                        abs_error_per_partial += masked_err.sum(dim=0)
                        count_per_partial += partials_mask.sum(dim=0)

                # ---- pitch accuracy ----

                train_loss_epoch = total_loss / total_samples
                octave_acc = correct_oct / total
                pc_acc = correct_pc / total
                full_acc = correct_both / total

                writer.add_scalar("loss/train", train_loss_epoch, epoch)
                writer.add_scalar("accuracy/octave", octave_acc, epoch)
                writer.add_scalar("accuracy/pitch", pc_acc, epoch)
                writer.add_scalar("accuracy/combined", full_acc, epoch)

                cm_img = plot_confusion_matrix(all_true, all_pred)

                writer.add_image("confusion_matrix", cm_img, global_step=epoch)

                # ---- partials error ----

                total_mae = (total_abs_error / total_count.clamp(min=1)).item()

                mae_per_partial = (abs_error_per_partial / count_per_partial.clamp(min=1))

                writer.add_scalar("mae/partials_total", total_mae, epoch)
                for k in range(mae_per_partial.shape[0]):
                    if not torch.isnan(mae_per_partial[k]):
                        writer.add_scalar(
                            f"mae/partial_{k+1}",
                            mae_per_partial[k].item(),
                            epoch,
                        )

                print(
                    f"Phase {phase+1}/{n_phases} | "
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"Loss {train_loss_epoch:.4f} | "
                    f"Octave {100*octave_acc:.2f}% | "
                    f"Pitch {100*pc_acc:.2f}% | "
                    f"Note {100*full_acc:.2f}% | "
                    f"Partials {total_mae:.3f}"
                )

        writer.close()

# ------------------------------------------------------------

def masked_mae(pred, target, mask):
    err = torch.abs(pred - target)
    err = err * mask
    denom = mask.sum().clamp(min=1)
    return err.sum() / denom

def plot_confusion_matrix(y_true, y_pred):
    cm = sklearn.metrics.confusion_matrix(np.asarray(y_true), np.asarray(y_pred))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm = cm.astype(float) / row_sums
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # Hide tick marks & labels
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    # Convert matplotlib figure to (H, W, 3) numpy array for TensorBoard
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    import PIL.Image
    image = np.array(PIL.Image.open(buf).convert('RGB'))
    image = image.transpose(2, 0, 1)
    return image
