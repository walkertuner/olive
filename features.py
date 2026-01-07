import collections
import glob
import json
import math
import os
import typing

import librosa
import torch
import torchaudio
from nnAudio.features.cqt import CQT

import olive

class Sample(typing.NamedTuple):
    X: torch.Tensor
    octave: torch.Tensor
    pitch: torch.Tensor
    partials: torch.Tensor
    mask: torch.Tensor
    stream_id: int

class RealTimeNoiseFloor:
    def __init__(self, sample_rate, window_ms=50, hold_time_ms=500, margin_db=3.0):
        """
        margin_db: how far above noise floor counts as 'silence', in dB
        """
        self.sample_rate = sample_rate
        self.window_size = int(sample_rate * window_ms / 1000)
        self.hold_frames = int(hold_time_ms / window_ms)
        self.rms_history = collections.deque(maxlen=100)
        self.min_rms = None
        self.silence_counter = 0
        
        # Convert margin in dB to RMS ratio
        self.margin_rms = 10 ** (margin_db / 20.0)
    
    def update(self, frame):
        """
        frame: tensor (1, N) mono
        Returns: (noise_floor_rms, is_silence)
        """
        rms = math.sqrt(float((frame ** 2).mean()))
        self.rms_history.append(rms)
        
        # Slowly adapt noise floor
        if self.min_rms is None:
            self.min_rms = rms
        else:
            self.min_rms = min(self.min_rms * 0.99 + rms * 0.01, rms)

        # Silence detection (RMS)
        if rms < self.min_rms * self.margin_rms:
            self.silence_counter += 1
        else:
            self.silence_counter = 0

        is_silence = self.silence_counter >= self.hold_frames
        return self.min_rms, is_silence
    
    def noise_floor_db(self):
        """Optional: get current floor in dBFS"""
        return 20 * math.log10(max(self.min_rms or 1e-10, 1e-10))
    
# ------------------------------------------------------------

def load_features(args):
    cache_path = os.path.join(args.log, f"{args.run}.feat")

    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        feature_data = torch.load(cache_path, weights_only=False)
    else:
        feature_data = build_feature_data(args)
        print(f"Saving features to {cache_path}")
        torch.save(feature_data, cache_path)

    return feature_data

def build_feature_data(args):
    highest_note = 120  # C8 (108) + 12 semitones
    lowest_note = highest_note - (args.num_octaves * 12)

    cqt = CQT(
        sr=olive.SAMPLE_RATE,
        fmin=librosa.midi_to_hz(lowest_note),
        n_bins=args.num_octaves * 12,
        hop_length=args.hop_size,
        bins_per_octave=12,
        output_format="Magnitude",
        pad_mode="constant",
    ).to(args.device)

    noise_floor = RealTimeNoiseFloor(olive.SAMPLE_RATE, window_ms=50, hold_time_ms=300, margin_db=3.0)

    metadata = load_metadata(args.db)

    total_samples = sum(len(f["samples"]) for f in metadata)

    print("Extracting features...")
    
    feature_data = []
    k = 0

    for data in metadata:
        for i, sample_data in enumerate(data["samples"], start=1):
            if "file" not in sample_data:
                continue

            print(
                f"[{k + i}/{total_samples}] "
                f"Processing {os.path.basename(sample_data['file'])}",
                end="\r",
                flush=True,
            )

            sample = extract_features(sample_data, k + i, cqt, noise_floor, args)
            
            feature_data.extend(sample)

        k += len(data["samples"])

    print("\nDone.")
    return feature_data

def load_metadata(path):
    if not os.path.exists(path):
        raise Exception(f"Data directory not found: {path}")

    if os.path.isdir(path):
        all_metadata = []
        for filepath in glob.glob(os.path.join(path, "**", "*.json"), recursive=True):
            all_metadata.append(load_metadata_file(filepath))
        return all_metadata
    else:
        return [load_metadata(path)]

def load_metadata_file(json_path):
    base_dir = os.path.dirname(os.path.abspath(json_path))
    with open(json_path, "r") as f:
        data = json.load(f)

    # Fix all file paths to be absolute
    for sample_data in data["samples"]:
        if "file" in sample_data:
            rel_path = sample_data["file"]
            abs_path = os.path.join(base_dir, rel_path)
            sample_data["file"] = os.path.normpath(abs_path)

    return data

def extract_features(sample_data, stream_index, cqt, noise_floor, args):
    path = sample_data["file"]
    waveform, file_sr = torchaudio.load(path)
    
    # Resample if needed
    if file_sr != olive.SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(file_sr, olive.SAMPLE_RATE)(waveform)

    # Detect onset
    onset = detect_onset_spectral_flux(waveform, args.hop_size, args.num_hops)
    
    # Mono
    x = waveform[0:1]  # (1, num_samples)

    # Slice from onset to end
    x = x[:, onset:]

    # Step hop-by-hop until silence
    num_frames = x.shape[1]
    cut_point = num_frames  # default: keep full
    for hop_start in range(0, num_frames, args.hop_size):
        hop_end = min(hop_start + args.hop_size, num_frames)
        frame = x[:, hop_start:hop_end]

        _, is_silence = noise_floor.update(frame)
        if is_silence:
            cut_point = hop_start  # stop at this hop boundary
            break

    # Trim tail
    x = x[:, :cut_point]

    # Compute CQT magnitude -> dB
    x = x.to(device=args.device, dtype=torch.float32)
    with torch.no_grad():
        cqt_mag = cqt(x)  # (1, freq_bins, T)
    cqt_db = 20.0 * torch.log10(torch.clamp(cqt_mag, min=1e-10))  # (1, F, T)

    T = cqt_db.shape[-1]
    samples = []
    i = 0
    while True:
        end = i + args.num_hops
        if end > T:
            break
        segment = cqt_db[:, :, i:end]
        if segment.shape[-1] > args.num_hops:
            segment = segment[:, :, :args.num_hops]
        elif segment.shape[-1] < args.num_hops:
            segment = torch.nn.functional.pad(segment, (0, args.num_hops - segment.shape[-1]), value=float(cqt_db.min().item()))
        
        sample_features = segment.squeeze(0).permute(1, 0).contiguous()

        note = int(sample_data["note"])
        octave = note // 12
        pitch_class = note % 12

        partials_target = torch.zeros(args.num_partials, dtype=torch.float32)
        partials_mask = torch.zeros(args.num_partials, dtype=torch.bool)
        for k, v in enumerate(sample_data["partials"]):
            if k >= args.num_partials:
                break
            if v is None:
                continue
            partials_target[k] = v
            partials_mask[k] = True

        samples.append(Sample(sample_features.cpu(), octave, pitch_class, partials_target, partials_mask, stream_index))
        
        i += 1
        if i >= T:
            break

    return samples

def detect_onsets(
    waveform, 
    hop_size,     # CQT hop in samples
    num_hops=4,     # num hops in a window/frame
    flux_threshold=0.1,  # relative to max flux per file
    ignore_frames=0       # skip first N frames to avoid start artifacts
):
    # Mono
    waveform = waveform.mean(dim=0, keepdim=True)

    frame_size = hop_size * num_hops

    # STFT magnitude
    spec = torch.stft(
        waveform,
        n_fft=frame_size,
        hop_length=hop_size,
        win_length=frame_size,
        window=torch.hann_window(frame_size),
        return_complex=True
    ).abs()
    spec = spec[0]
    
    # Frame-to-frame diff, half-wave rectified (absolute magnitude change)
    flux = (spec[:, 1:] - spec[:, :-1]).clamp(min=0.0)
    # total flux
    flux = flux.pow(2).sum(dim=0).sqrt()
    # Normalize per file so threshold is in [0,1] range
    if flux.max() > 0:
        flux = flux / flux.max()

    # Find first frame above threshold after ignoring early frames
    onset_indices = (flux >= flux_threshold).nonzero(as_tuple=True)[0]
    onset_indices = onset_indices[onset_indices >= ignore_frames]

    return onset_indices

def detect_onset_spectral_flux(
    waveform, 
    hop_size,     # CQT hop in samples
    num_hops=4,     # num hops in a window/frame
    flux_threshold=0.1,  # relative to max flux per file
    ignore_frames=0       # skip first N frames to avoid start artifacts
):
    onset_indices = detect_onsets(waveform,hop_size,num_hops,flux_threshold,ignore_frames)

    if onset_indices.numel() == 0:
        return 0  # no onset detected

    onset_frame = onset_indices[0].item()
    onset_samples = onset_frame * hop_size

    # Clamp to valid range
    total_samples = waveform.shape[-1]
    onset_samples = min(onset_samples, total_samples - 1)

    return onset_samples

def split_on_onsets(waveform: torch.Tensor,
                    onset_indices,          # iterable of ints in *samples*
                    min_len_samples=0,      # drop segments shorter than this
                    include_tail=True):     # keep last tail after final onset
    """
    Returns a list of segments (each (C, T_i)) split at the given onset indices.
    """

    C, N = waveform.shape

    # 1) Normalize/validate indices: keep in (0, N), unique, sorted
    onsets = torch.tensor([int(i) for i in onset_indices], dtype=torch.long)
    onsets = onsets[(onsets > 0) & (onsets < N)].unique(sorted=True)

    # 2) Build cut boundaries: [0, *onsets, N] (include tail if requested)
    boundaries = [0]
    boundaries += onsets.tolist()
    if include_tail and boundaries[-1] != N:
        boundaries.append(N)

    merged_bounds = [boundaries[0]]
    skip_until = None
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        seg_len = b - a

        if seg_len < min_len_samples:
            # small segment → don't append boundary now,
            # will merge into the next segment
            continue
        else:
            # end of a long segment → append boundary
            merged_bounds.append(b)

    # Ensure last boundary is waveform end
    if merged_bounds[-1] != boundaries[-1]:
        merged_bounds.append(boundaries[-1])

    # 3) Convert boundaries -> segment sizes and use torch.split along time
    sizes = [b - a for a, b in zip(merged_bounds[:-1], merged_bounds[1:])]

    if not sizes:
        return []

    segments = list(torch.split(waveform, sizes, dim=1))
    return segments