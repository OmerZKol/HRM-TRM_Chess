"""
Unified model wrapper for chess training.

This wrapper initializes the appropriate model based on config['model_type'].
Supports: HRM, TRM, TRM baseline, Transformer, and CNN models.
"""

import torch.nn as nn

from model.hrm.hrm_model import HierarchicalReasoningModel_ACTV1
from model.trm import TinyRecursiveModel_ACTV1, TinyRecursiveReasoningModel_ACTV1
from model.bridge import AlphaZeroBridge
from model.transformer_chess_nn import TransformerChessNet
from model.SimpleChessNet import SimpleChessNet


class ChessModelWrapper(nn.Module):
    """
    Unified wrapper for all chess model types.

    Initializes the appropriate model based on config['model_type']:
    - 'hrm': Hierarchical Reasoning Model
    - 'trm': Tiny Recursive Model
    - 'trm_baseline': TRM Baseline Model
    - 'transformer': Standard Transformer
    - 'CNN': Simple CNN baseline

    All models return:
        pi: Policy logits [batch_size, 1858]
        v: Value logits in WDL format [batch_size, 3]
        moves_left: Moves left predictions [batch_size, 1]
        q_info: Dictionary with additional information
    """

    def __init__(self, config):
        super(ChessModelWrapper, self).__init__()

        model_type = config.get('model_type')
        if model_type is None:
            raise ValueError("config must contain 'model_type' key")

        self.model_type = model_type

        if model_type == 'hrm':
            hrm_config = config.get('hrm_config')
            if hrm_config is None:
                raise ValueError("HRM model requires 'hrm_config' in config")
            hrm_model = HierarchicalReasoningModel_ACTV1(hrm_config)
            self.bridge = AlphaZeroBridge(hrm_model)
            self._use_bridge = True

        elif model_type == 'trm':
            trm_config = config.get('trm_config')
            if trm_config is None:
                raise ValueError("TRM model requires 'trm_config' in config")
            trm_model = TinyRecursiveModel_ACTV1(trm_config)
            self.bridge = AlphaZeroBridge(trm_model)
            self._use_bridge = True

        elif model_type == 'trm_baseline':
            trm_config = config.get('trm_config')
            if trm_config is None:
                raise ValueError("TRM baseline model requires 'trm_config' in config")
            trm_baseline_model = TinyRecursiveReasoningModel_ACTV1(trm_config)
            self.bridge = AlphaZeroBridge(trm_baseline_model)
            self._use_bridge = True

        elif model_type == 'transformer':
            self.model = TransformerChessNet(config)
            self._use_bridge = False

        elif model_type == 'CNN':
            self.model = SimpleChessNet()
            self._use_bridge = False

        else:
            raise ValueError(
                f"Unknown model_type: '{model_type}'. "
                f"Supported types: 'hrm', 'trm', 'trm_baseline', 'transformer', 'CNN'"
            )

    def forward(self, s):
        """
        Forward pass through the model.

        Args:
            s: Board state [batch_size, 112, 8, 8]

        Returns:
            pi: Policy logits [batch_size, 1858]
            v: Value logits in WDL format [batch_size, 3]
            moves_left: Moves left predictions [batch_size, 1]
            q_info: Dictionary with additional information
        """
        if self._use_bridge:
            return self.bridge(s)
        else:
            return self.model(s)
