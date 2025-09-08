"""
Chess puzzle dataset for HRM move prediction training.
Extends the base PuzzleDataset to handle move targets.
"""

import os
import json
import numpy as np
import pydantic
import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, _sample_batch, _sample_batch_simple


class ChessPuzzleDatasetConfig(PuzzleDatasetConfig):
    """Extended config for chess move prediction."""
    pass


class ChessPuzzleDataset(PuzzleDataset):
    """Chess dataset with move prediction targets."""
    
    def __init__(self, config: ChessPuzzleDatasetConfig, split: str = "train"):
        super().__init__(config, split)
        # Load dataset metadata to get num_actions for masking
        with open(os.path.join(config.dataset_path, "dataset.json"), 'r') as f:
            self.dataset_metadata = json.load(f)
    
    
    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "move_targets": "r",  # Move prediction targets
            "possible_moves": "r",  # Keep possible moves in memory for masking
            "value_targets": "r",  # Value prediction targets

            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Load data
        self._data = {}
        for set_name in self.metadata.sets:
            # Load subset
            self._data[set_name] = {
                field_name: np.load(os.path.join(self.config.dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                for field_name, mmap_mode in field_mmap_modes.items()
            }

    def _collate_batch(self, batch):
        # Convert dtype, handling possible_moves specially (they're already masks as float32)
        converted_batch = {}
        for k, v in batch.items():
            if k == "possible_moves":
                # possible_moves are already binary masks stored as float32
                converted_batch[k] = v.astype(np.float32)
            else:
                converted_batch[k] = v.astype(np.int32)
        batch = converted_batch

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            # Handle move targets ignore IDs too
            if "move_targets" in batch:
                batch["move_targets"][batch["move_targets"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size

            pad_values = {
                "inputs": self.metadata.pad_id,
                "move_targets": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id,
                "possible_moves": 0.0,  # Pad with zeros (no valid moves)
                "value_targets": IGNORE_LABEL_ID
            }
            batch = {k: np.pad(v, ((0, pad_size), ) + ((0, 0), ) * (v.ndim - 1), constant_values=pad_values.get(k, 0)) for k, v in batch.items()}

        # Convert to tensor and rename possible_moves to move_masks
        batch_tensors = {}
        for k, v in batch.items():
            if k == "possible_moves":
                batch_tensors["move_masks"] = torch.from_numpy(v)
            else:
                batch_tensors[k] = torch.from_numpy(v)
        
        return batch_tensors
    
    def _iter_test(self):
        for set_name, dataset in self._data.items():  # type: ignore
            total_examples = len(dataset["inputs"])

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)
                
                batch_data = {
                    "inputs": dataset["inputs"][local_start: local_end],
                    "move_targets": dataset["move_targets"][local_start: local_end],
                    "value_targets": dataset["value_targets"][local_start: local_end],
                    "possible_moves": dataset["possible_moves"][local_start: local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                }
                
                batch = self._collate_batch(batch_data)

                yield set_name, batch, end_index - start_index
                
                # Advance to next batch
                start_index += self.config.global_batch_size

    def _iter_train(self):
        for set_name, dataset in self._data.items():  # type: ignore
            # Increase epoch count
            self._iters += 1

            # Simple puzzle sampling without groups
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))
            
            # Calculate how many puzzles we have (puzzle_indices length - 1)
            num_puzzles = len(dataset["puzzle_indices"]) - 1
            
            # Calculate actual number of batches based on dataset size
            total_examples = len(dataset["inputs"])
            num_batches = (total_examples + self.config.global_batch_size - 1) // self.config.global_batch_size
            num_batches = num_batches * self.config.epochs_per_iter
            
            for batch_idx in range(num_batches):
                batch_indices, batch_puzzle_indices = _sample_batch_simple(
                    rng,
                    puzzle_indices=dataset["puzzle_indices"],
                    global_batch_size=self.config.global_batch_size,
                    num_puzzles=num_puzzles
                )
                
                # Select current rank and collate
                global_effective_batch_size = batch_puzzle_indices.size  # Global effective batch size, excluding pads
                
                # Drop last batch if too small
                if global_effective_batch_size < self.config.global_batch_size:
                    continue

                batch_indices        = batch_indices       [self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_puzzle_indices = batch_puzzle_indices[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                
                batch_data = {
                    "inputs": dataset["inputs"][batch_indices],
                    "move_targets": dataset["move_targets"][batch_indices],
                    "value_targets": dataset["value_targets"][batch_indices],
                    "possible_moves": dataset["possible_moves"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices]
                }
                batch = self._collate_batch(batch_data)
                yield set_name, batch, global_effective_batch_size