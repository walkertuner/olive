import torch
import torch.nn as nn

class OLIVE(nn.Module):
    """
    Input:  x of shape (B, T, Freq)
    Output:
        octave_logits (B, num_octaves)
        pitch_logits  (B, num_pitch_classes)
        partials      (B, num_partials)
        h             (num_layers * num_dirs, B, rnn_hidden)
    """

    def __init__(
        self,
        num_instruments,
        freq_bins=108,
        cnn_out=64,
        rnn_layers=1,
        rnn_hidden=128,
        mlp_layers=1,
        mlp_hidden=128,
        voicing_dim=32,
        num_octaves=8,
        num_pitches=12,
        num_partials=8,
    ):
        super().__init__()

        self.num_instruments = num_instruments
        self.freq_bins = freq_bins
        self.rnn_layers = rnn_layers
        self.rnn_hidden = rnn_hidden
        self.mlp_layers=mlp_layers
        self.mlp_hidden=mlp_hidden
        self.voicing_dim = voicing_dim
        self.num_octaves = num_octaves
        self.num_pitches = num_pitches
        self.num_partials = num_partials

        # ---------- Per-frame CNN ----------
        self.cnn = self._make_cnn_1d(
            in_channels=1,
            channels=[16, 32],
            kernel_sizes=[5, 3],
            out_dim=cnn_out,
        )

        # ---------- Temporal encoder ----------
        self.rnn = nn.GRU(
            input_size=cnn_out,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=False,
            dropout=0.0,
            batch_first=True
        )

        # ---------- Hierarchical note heads ----------
        self.octave_head = self._make_mlp(
            in_dim=rnn_hidden,
            out_dim=num_octaves,
        )

        self.pitch_head = self._make_mlp(
            in_dim=rnn_hidden + num_octaves,
            out_dim=num_pitches,
        )

        # ---------- Latent voicing heads ----------
        self.instrument_emb = nn.Embedding(num_instruments, voicing_dim)

        self.voicing_instr_head = self._make_mlp(
            in_dim=rnn_hidden + voicing_dim,
            out_dim=voicing_dim,
        )

        self.voicing_note_head = self._make_mlp(
            in_dim=rnn_hidden + voicing_dim + num_octaves + num_pitches,
            out_dim=voicing_dim,
        )

        # ---------- Partial decoder ----------
        self.partial_head = nn.Linear(voicing_dim, num_partials)

    # ------------------------------------------------------------------

    def _make_cnn_1d(self, in_channels: int, channels: list[int], kernel_sizes: list[int], out_dim: int):
        assert len(channels) == len(kernel_sizes)

        layers = []
        c_in = in_channels

        for c_out, k in zip(channels, kernel_sizes):
            layers.append(
                nn.Conv1d(
                    c_in,
                    c_out,
                    kernel_size=k,
                    padding=k // 2,   # preserve length
                )
            )
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out

        layers.append(nn.AdaptiveAvgPool1d(out_dim))
        return nn.Sequential(*layers)

    def _make_mlp(self, in_dim: int, out_dim: int):
        layers = []
        dims = [in_dim] + [self.mlp_hidden] * self.mlp_layers

        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(dims[-1], out_dim))
        return nn.Sequential(*layers)

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

    def forward(self, x, instrument_id):
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
        rnn_out, h = self.rnn(emb)

        ctx = rnn_out[:, -1, :]

        # ----- Note classification -----
        octave_logits = self.octave_head(ctx)
        octave_probs = nn.functional.softmax(octave_logits, dim=-1)

        pitch_in = torch.cat([ctx, octave_probs], dim=-1)
        pitch_logits = self.pitch_head(pitch_in)
        pitch_probs = nn.functional.softmax(pitch_logits, dim=-1)

        # ----- Latent voicing -----
        instr_emb = self.instrument_emb(instrument_id)
        instr_in = torch.cat([instr_emb, ctx.detach()], dim=-1)
        voicing_instr = self.voicing_instr_head(instr_in)

        note_in = torch.cat([ctx, voicing_instr, octave_probs, pitch_probs], dim=-1)
        voicing_note = self.voicing_note_head(note_in)

        voicing = voicing_instr + voicing_note

        # ----- All partials -----
        partials = self.partial_head(voicing)

        return octave_logits, pitch_logits, partials

    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_hidden(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return torch.zeros(
            self.rnn_layers,
            batch_size,
            self.rnn_hidden,
            device=device,
        )

    # ------------------------------------------------------------------

    def forward_step(self, frame, instrument_id):
        """
        Streaming single-frame step.
        frame: (B, Freq)
        """
        B, Freq = frame.shape
        assert Freq == self.freq_bins, f"Expected {self.freq_bins}, got {Freq}"

        # CNN
        emb = self._encode_frames(frame)   # (B, cnn_out)
        emb = emb.unsqueeze(1)             # (B, 1, cnn_out)

        # GRU
        rnn_out, h = self.rnn(emb)
        ctx = rnn_out[:, -1, :]             # (B, rnn_feat_dim)

        # ----- Note classification -----
        octave_logits = self.octave_head(ctx)
        octave_probs = nn.functional.softmax(octave_logits, dim=-1)

        pitch_in = torch.cat([ctx, octave_probs], dim=-1)
        pitch_logits = self.pitch_head(pitch_in)
        pitch_probs = nn.functional.softmax(pitch_logits, dim=-1)

        # ----- Latent voicing -----
        instr_emb = self.instrument_emb(instrument_id)
        instr_in = torch.cat([instr_emb, ctx.detach()], dim=-1)
        voicing_instr = self.voicing_instr_head(instr_in)

        note_in = torch.cat([ctx, octave_probs, pitch_probs], dim=-1)
        voicing_note = self.voicing_note_head(note_in)

        voicing = voicing_instr + voicing_note

        # ----- Partials -----
        partials = self.partial_head(voicing)

        return octave_logits, pitch_logits, partials

