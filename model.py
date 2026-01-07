import torch
import torch.nn as nn

class OLIVE(nn.Module):
    """
    Input:  x of shape (B, T, Freq)
    Output:
        octave_logits (B, num_octaves)
        pitch_logits  (B, num_pitch_classes)
        partials      (B,)
        voicing       (B, voicing_dim)
        h             (num_layers * num_dirs, B, rnn_hidden)
    """

    def __init__(
        self,
        freq_bins=108,
        cnn_out=64,
        rnn_hidden=128,
        num_octaves=8,
        num_pitch_classes=12,
        bidirectional=False,
        rnn_layers=1,
        dropout=0.0,
        voicing_dim=32,
        num_partials=8,
    ):
        super().__init__()

        self.freq_bins = freq_bins
        self.num_octaves = num_octaves
        self.num_pitch_classes = num_pitch_classes
        self.bidirectional = bidirectional
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.voicing_dim = voicing_dim
        self.num_partials = num_partials

        # ---------- Per-frame CNN ----------
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(cnn_out),
        )

        # ---------- Temporal encoder ----------
        self.rnn = nn.GRU(
            input_size=cnn_out,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=bidirectional,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=True,
        )

        rnn_feat_dim = rnn_hidden * (2 if bidirectional else 1)

        # ---------- Hierarchical note heads ----------
        self.octave_head = nn.Sequential(
            nn.Linear(rnn_feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_octaves),
        )

        self.pitch_head = nn.Sequential(
            nn.Linear(rnn_feat_dim + num_octaves, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_pitch_classes),
        )

        # ---------- Latent tuning head ----------
        self.voicing_head = nn.Sequential(
            nn.Linear(rnn_feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, voicing_dim),
        )

        # ---------- Partial decoder (training supervision only) ----------
        self.partial_head = nn.Linear(voicing_dim, num_partials)

    # ------------------------------------------------------------------

    def _encode_frames(self, x_bt_freq):
        """
        x_bt_freq: (B*T, Freq)
        returns:   (B*T, cnn_out)
        """
        x = x_bt_freq.unsqueeze(1)   # (B*T, 1, Freq)
        x = self.cnn(x)              # (B*T, 32, cnn_out)
        x = x.mean(dim=1)            # (B*T, cnn_out)
        return x

    # ------------------------------------------------------------------

    def forward(self, x, h0=None, use_last=True):
        """
        x: (B, T, Freq)
        """
        B, T, Freq = x.shape
        assert Freq == self.freq_bins, f"Expected {self.freq_bins}, got {Freq}"

        # Per-frame CNN
        x_bt = x.reshape(B * T, Freq)
        emb_bt = self._encode_frames(x_bt)
        emb = emb_bt.view(B, T, -1)

        # GRU
        rnn_out, h = self.rnn(emb, h0)

        if use_last:
            ctx = rnn_out[:, -1, :]
        else:
            ctx = rnn_out.mean(dim=1)

        # ----- Note classification -----
        octave_logits = self.octave_head(ctx)
        octave_probs = nn.functional.softmax(octave_logits, dim=-1)

        pitch_in = torch.cat([ctx, octave_probs], dim=-1)
        pitch_logits = self.pitch_head(pitch_in)

        # ----- Latent tuning -----
        voicing = self.voicing_head(ctx)

        # ----- All partials -----
        partials = self.partial_head(voicing)
        # shape: (B, MAX_PARTIALS)

        return octave_logits, pitch_logits, partials, voicing, h

    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_hidden(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        num_dirs = 2 if self.bidirectional else 1
        return torch.zeros(
            self.rnn_layers * num_dirs,
            batch_size,
            self.rnn_hidden,
            device=device,
        )

    # ------------------------------------------------------------------

    def forward_step(self, frame, h_prev=None):
        """
        Streaming single-frame step.
        frame: (B, Freq)
        """
        B, Freq = frame.shape
        assert Freq == self.freq_bins, f"Expected {self.freq_bins}, got {Freq}"

        emb = self._encode_frames(frame)
        emb = emb.unsqueeze(1)

        rnn_out, h_next = self.rnn(emb, h_prev)
        ctx = rnn_out[:, -1, :]

        octave_logits = self.octave_head(ctx)
        octave_probs = nn.functional.softmax(octave_logits, dim=-1)

        pitch_in = torch.cat([ctx, octave_probs], dim=-1)
        pitch_logits = self.pitch_head(pitch_in)

        voicing = self.tuning_head(ctx)
        partials = self.partial_head(voicing)

        return octave_logits, pitch_logits, partials, voicing, h_next
