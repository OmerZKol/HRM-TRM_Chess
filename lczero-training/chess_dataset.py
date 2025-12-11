#!/usr/bin/env python3
"""PyTorch Dataset for chess training data from Leela Chess Zero format"""

import torch
from torch.utils.data import Dataset
import numpy as np
import struct
import gzip
import random
from typing import Tuple, List

# Training record version constants (from chunkparser.py)
V6_VERSION = struct.pack('i', 6)
V5_VERSION = struct.pack('i', 5)
V4_VERSION = struct.pack('i', 4)
V3_VERSION = struct.pack('i', 3)
CLASSICAL_INPUT = struct.pack('i', 1)

V6_STRUCT_STRING = '4si7432s832sBBBBBBBbfffffffffffffffIHH4H'
V5_STRUCT_STRING = '4si7432s832sBBBBBBBbfffffff'
V4_STRUCT_STRING = '4s7432s832sBBBBBBBbffff'
V3_STRUCT_STRING = '4s7432s832sBBBBBBBb'


def reverse_expand_bits(plane: int) -> bytes:
    """Convert single byte to expanded bit representation"""
    return np.unpackbits(np.array([plane], dtype=np.uint8))[::-1].astype(np.float32).tobytes()


class ChessDataset(Dataset):
    """PyTorch Dataset for chess training data"""

    def __init__(self, chunk_files: List[str], sample_rate: int = 1,
                 expected_input_format: int = None, shuffle_size: int = 8192):
        self.chunk_files = chunk_files
        self.expected_input_format = expected_input_format
        self.shuffle_size = shuffle_size

        # Initialize struct parsers
        self.v6_struct = struct.Struct(V6_STRUCT_STRING)
        self.v5_struct = struct.Struct(V5_STRUCT_STRING)
        self.v4_struct = struct.Struct(V4_STRUCT_STRING)
        self.v3_struct = struct.Struct(V3_STRUCT_STRING)

        # Pre-computed flat planes for efficiency (matching chunkparser.py)
        self.flat_planes = []
        for i in range(2):
            self.flat_planes.append(
                (np.zeros(64, dtype=np.float32) + i).tobytes())

        # Load all training records
        self.records = self._load_all_records()

    def _load_all_records(self) -> List[bytes]:
        """Load and parse all training records from chunk files"""
        records = []

        for idx, chunk_file in enumerate(self.chunk_files):
            try:
                if chunk_file.endswith('.gz'):
                    with gzip.open(chunk_file, 'rb') as f:
                        chunk_data = f.read()
                else:
                    with open(chunk_file, 'rb') as f:
                        chunk_data = f.read()

                sampled = self._sample_records(chunk_data)
                records.extend(sampled)

            except Exception as e:
                print(f"[ChessDataset] Error loading {chunk_file}: {e}")
        random.shuffle(records)
        return records

    def _sample_records(self, chunkdata: bytes) -> List[bytes]:
        """Sample records from chunk data with downsampling"""
        version = chunkdata[0:4]

        if version == V6_VERSION:
            record_size = self.v6_struct.size
        elif version == V5_VERSION:
            record_size = self.v5_struct.size
        elif version == V4_VERSION:
            record_size = self.v4_struct.size
        elif version == V3_VERSION:
            record_size = self.v3_struct.size
        else:
            return []

        records = []

        for i in range(0, len(chunkdata), record_size):

            record = chunkdata[i:i + record_size]

            # Pad older versions to V6 format
            if version == V3_VERSION:
                record += 16 * b'\x00'  # Add fake root_q, best_q, root_d, best_d
            if version == V3_VERSION or version == V4_VERSION:
                record += 12 * b'\x00'  # Add fake root_m, best_m, plies_left
                record = record[:4] + CLASSICAL_INPUT + record[4:]  # Insert input format
            if version in [V3_VERSION, V4_VERSION, V5_VERSION]:
                record += 48 * b'\x00'  # Add fake result_q, result_d etc

            records.append(record)

        return records

    def _create_dummy_sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a dummy sample when data parsing fails"""
        planes = torch.zeros(112, 8, 8, dtype=torch.float32)
        policy = torch.zeros(1858, dtype=torch.float32)
        policy[0] = 1.0  # Set first move as 100% probability
        value = torch.tensor([0.33, 0.34, 0.33], dtype=torch.float32)  # Balanced WDL
        best_q = torch.tensor([0.33, 0.34, 0.33], dtype=torch.float32)
        moves_left = torch.tensor([40.0], dtype=torch.float32)
        return planes, policy, value, best_q, moves_left

    def _convert_v6_to_tuple(self, content: bytes) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert V6 binary record to tensors (adapted from chunkparser.py)"""

        # Unpack the V6 content
        (ver, input_format, probs, planes, us_ooo, us_oo, them_ooo, them_oo,
         stm, rule50_count, invariance_info, dep_result, root_q, best_q,
         root_d, best_d, root_m, best_m, plies_left, result_q, result_d,
         played_q, played_d, played_m, orig_q, orig_d, orig_m, visits,
         played_idx, best_idx, reserved1, reserved2, reserved3, reserved4) = self.v6_struct.unpack(content)

        # Handle plies_left fallback
        if plies_left == 0:
            plies_left = invariance_info

        # Verify input format matches expectation (matching chunkparser.py behavior)
        if self.expected_input_format is None:
            self.expected_input_format = input_format
        else:
            assert input_format == self.expected_input_format, \
                f"Input format mismatch: expected {self.expected_input_format}, got {input_format}"

        # Unpack bit planes and cast to float32
        planes_array = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(np.float32)

        # Rule50 plane
        rule50_divisor = 100.0 if input_format > 3 else 99.0
        rule50_plane = struct.pack('f', rule50_count / rule50_divisor) * 64

        # Handle different input formats for middle planes
        if input_format == 1:
            # Classic input format - simpler plane structure using indices
            middle_planes = (self.flat_planes[us_ooo] +
                           self.flat_planes[us_oo] +
                           self.flat_planes[them_ooo] +
                           self.flat_planes[them_oo] +
                           self.flat_planes[stm])
        elif input_format == 2:
            # Each inner array has to be reversed as these fields are in opposite endian to the planes data
            them_ooo_bytes = reverse_expand_bits(them_ooo)
            us_ooo_bytes = reverse_expand_bits(us_ooo)
            them_oo_bytes = reverse_expand_bits(them_oo)
            us_oo_bytes = reverse_expand_bits(us_oo)
            middle_planes = (us_ooo_bytes + (6*8*4) * b'\x00' + them_ooo_bytes +
                           us_oo_bytes + (6*8*4) * b'\x00' + them_oo_bytes +
                           self.flat_planes[0] +
                           self.flat_planes[0] +
                           self.flat_planes[stm])
        elif input_format in [3, 4, 132, 5, 133]:
            # Each inner array has to be reversed as these fields are in opposite endian to the planes data
            them_ooo_bytes = reverse_expand_bits(them_ooo)
            us_ooo_bytes = reverse_expand_bits(us_ooo)
            them_oo_bytes = reverse_expand_bits(them_oo)
            us_oo_bytes = reverse_expand_bits(us_oo)
            enpassant_bytes = reverse_expand_bits(stm)

            middle_planes = (us_ooo_bytes + (6*8*4) * b'\x00' + them_ooo_bytes +
                           us_oo_bytes + (6*8*4) * b'\x00' + them_oo_bytes +
                           self.flat_planes[0] + self.flat_planes[0] +
                           (7*8*4) * b'\x00' + enpassant_bytes)

        # Edge detection plane
        aux_plus_6_plane = self.flat_planes[0]
        if (input_format in [132, 133]) and invariance_info >= 128:
            aux_plus_6_plane = self.flat_planes[1]

        # Concatenate all planes (matching chunkparser.py)
        # Make the last plane all 1's so the NN can detect edges of the board more easily
        all_planes = (planes_array.tobytes() + middle_planes +
                     rule50_plane + aux_plus_6_plane + self.flat_planes[1])

        # Verify expected length (matching chunkparser.py assertion)
        assert len(all_planes) == ((8 * 13 * 1 + 8 * 1 * 1) * 8 * 8 * 4), \
            f"Plane length mismatch: expected {(8 * 13 * 1 + 8 * 1 * 1) * 8 * 8 * 4}, got {len(all_planes)}"

        # Convert to tensor and reshape to (112, 8, 8)
        # Copy the buffer to make it writable and avoid PyTorch warnings
        planes_array_float = np.frombuffer(all_planes, dtype=np.float32).copy()
        planes_tensor = torch.from_numpy(planes_array_float).view(112, 8, 8)

        # Policy probabilities
        probs_array = np.frombuffer(probs, dtype=np.float32).copy()
        policy_tensor = torch.from_numpy(probs_array)

        # Value targets (WDL format - matching chunkparser.py)
        if ver == V6_VERSION:
            value_tensor = torch.tensor([
                0.5 * (1.0 - result_d + result_q),  # Win probability
                result_d,                            # Draw probability
                0.5 * (1.0 - result_d - result_q)   # Loss probability
            ], dtype=torch.float32)
        else:
            dep_result = float(dep_result)
            assert dep_result == 1.0 or dep_result == -1.0 or dep_result == 0.0, \
                f"Invalid dep_result: {dep_result}"
            value_tensor = torch.tensor([
                float(dep_result == 1.0),   # Win
                float(dep_result == 0.0),   # Draw
                float(dep_result == -1.0)   # Loss
            ], dtype=torch.float32)

        # Best Q value (also in WDL format - matching chunkparser.py)
        best_q_w = 0.5 * (1.0 - best_d + best_q)
        best_q_l = 0.5 * (1.0 - best_d - best_q)
        assert -1.0 <= best_q <= 1.0 and 0.0 <= best_d <= 1.0, \
            f"Invalid best_q or best_d: best_q={best_q}, best_d={best_d}"
        best_q_tensor = torch.tensor([best_q_w, best_d, best_q_l], dtype=torch.float32)

        # Moves left
        moves_left_tensor = torch.tensor([plies_left], dtype=torch.float32)

        return planes_tensor, policy_tensor, value_tensor, best_q_tensor, moves_left_tensor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        return self._convert_v6_to_tuple(record)
