import argparse
import json
import os
import struct
import sys

import torch
import torchaudio
import numpy as np

import plcli
import rct
import olive

def parse_paz(filename):
    basename, extension = os.path.splitext(filename)
    paz_filename = f"{basename}.paz"

    metadata = [{}] * 88
    if os.path.exists(paz_filename):
        paz = rct.load_paz(paz_filename)
        for i in range(88):
            partials = paz["offsets_harmonic"]["by_note"][i+10]
            partials_ET = paz["offsets_ET"][1]["by_note"][i+10]
            amplitude_data = paz["amplitude"]["by_note"][i+10]
            pcm_data = paz["pcm"][i]

            metadata[i] = {"note": i,
                           "partials": [],
            #               "amplitudes": [],
                           "pcm": []}
            metadata[i]["partials"] =  [None if v == 0.0 else v for v in partials["values"]]
            metadata[i]["partials_ET"] =  [None if v == 0.0 else v for v in partials_ET["values"]]
            metadata[i]["amplitudes"] = [None if v == 0.0 else v for v in amplitude_data["values"]]
            metadata[i]["pcm"] = pcm_data

    else:
        print(paz_filename+" doesn't exist")

    return metadata

def parse_args():
    parser = argparse.ArgumentParser(description="Train note classifier.")

    parser.add_argument("--db", type=str, default="db", help="Path to database directory")
    parser.add_argument("files", nargs="+", help="One or more input .paz files")

    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Output directory: {args.db}")
    os.makedirs(args.db, exist_ok=True)

    for input_path in args.files:
        basename, extension = os.path.splitext(os.path.basename(input_path))
        instrument, pass_number = basename.split('_')
        pass_number = int(pass_number)-1
        
        metadata = {"instrument": instrument,
                    "pass":       pass_number,
                    "samples":    parse_paz(input_path)}

        file_dir = os.path.join(args.db, instrument)
        os.makedirs(file_dir, exist_ok=True)

        for note, sample in enumerate(metadata["samples"]):
             output_name = f"{instrument}_{pass_number:02d}_{note:02d}.wav"
             output_filename = os.path.join(file_dir, output_name)
             pcm = sample.get("pcm", None)
             if len(pcm) > 0:
                waveform = torch.tensor(pcm, dtype=torch.int16)
                waveform = waveform.unsqueeze(0)  # (1, N) = mono
                torchaudio.save(output_filename, waveform, olive.SAMPLE_RATE)
                sample["file"] = output_name
             del sample["pcm"]

        num_samples = len(metadata["samples"])
        print(f"[✓] Extracted {basename}\tnotes={num_samples}")

        metadata_filename = os.path.join(file_dir, f"{instrument}_{pass_number:02d}.json")
        with open(metadata_filename, "w") as meta_file:
            json.dump(metadata, meta_file, indent=2)
        print(f"[✓] Wrote {metadata_filename}")