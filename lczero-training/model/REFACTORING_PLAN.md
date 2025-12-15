# Model Directory Refactoring Plan

This document outlines suggested changes to reorganize the model directory for research publication.

## Current State Analysis

The codebase implements Hierarchical Reasoning Models (HRM) and Tiny Recursive Models (TRM) for chess, adapted to work with AlphaZero-style training. While functional, the structure has accumulated technical debt that makes it harder for others to understand and use.

## Major Issues Identified

### 1. Duplicate Bridge Files

**Location**: `HRMBridge.py` and `HRMBridge2.py`

**Problem**: These files are 99% identical. The only differences:
- `HRMBridge2.py` is missing the `SimpleHRMLoss` class
- Minor difference in device handling (one line in forward method)

**Impact**: Maintenance burden, confusion about which to use

**Suggested Fix**: Merge into a single `bridge.py` file with a unified `AlphaZeroBridge` class

---

### 2. Unnecessary Wrapper Classes

**Location**: `ChessNNet.py`, `ChessTRMNet.py`, `ChessTRMBaselineNet.py`

**Problem**: These are extremely thin wrappers (10-36 lines each) that only:
1. Instantiate the actual model
2. Wrap it in a bridge
3. Apply `torch.tanh()` to the value output

Example:
```python
class ChessNNet(nn.Module):
    def __init__(self, config):
        super(ChessNNet, self).__init__()
        hrm_model = HierarchicalReasoningModel_ACTV1(config)
        self.bridge = HRMAlphaZeroBridge(hrm_model)

    def forward(self, s):
        pi, v, moves_left, q_info = self.bridge(s)
        return pi, torch.tanh(v), moves_left, q_info  # Only adds tanh
```

**Impact**: Extra abstraction layer that provides minimal value, harder to trace code flow

**Suggested Fix**:
- Eliminate these wrappers entirely
- Move the `torch.tanh()` logic into the bridge as a configurable option
- Or handle value activation in the training code where it's more explicit

---

### 3. Code Duplication Between Model Directories

**Location**: `HRM_model/` and `trm_model/` subdirectories

**Problem**: Multiple files are duplicated:
- `common.py` - **Identical** in both directories (only contains `trunc_normal_init_`)
- Likely `layers.py`, `sparse_embedding.py`, `losses.py` are also duplicated

**Impact**:
- Bug fixes must be applied twice
- Unclear which version is "canonical"
- Wastes space and creates confusion

**Suggested Fix**: Create a shared `common/` directory for utilities used by both models

---

### 4. Loss Functions in Wrong Location

**Location**: `SimpleHRMLoss` class is in `HRMBridge.py` (lines 104-224)

**Problem**:
- Loss functions are defined in a model/bridge file
- Both `HRM_model/` and `trm_model/` have `losses.py` files
- Violates separation of concerns (model vs training)

**Impact**: Hard to find loss implementations, unclear organization

**Suggested Fix**: Move all loss functions to a single `losses.py` at the model root level

---

### 5. Messy Path Management

**Location**: Top of `ChessNNet.py`, `ChessTRMNet.py`, `ChessTRMBaselineNet.py`

**Problem**: Multiple `sys.path.append()` calls:
```python
sys.path.append('..')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append('../..')
```

**Impact**:
- Fragile import system
- Breaks when files are moved
- Not following Python best practices

**Suggested Fix**: Use proper package structure with `__init__.py` files and relative imports

---

### 6. Confusing Naming Conventions

**Problems**:
- Bridge named `HRMAlphaZeroBridge` but used for both HRM and TRM models
- No clear distinction between `HRMBridge` vs `HRMBridge2`
- File `hrm_act_v1.py` is verbose and unclear what v1 means
- `transformers_baseline.py` is unclear - baseline of what?

**Impact**: Harder for new readers to understand the codebase structure

**Suggested Fix**:
- Rename bridge to generic `AlphaZeroBridge` or `ChessModelBridge`
- Use clear, descriptive names: `hrm_model.py`, `trm_model.py`, `trm_baseline.py`

---

## Recommended New Structure

```
model/
├── __init__.py
├── README.md                      # Architecture overview and usage guide
│
├── common/                        # Shared utilities
│   ├── __init__.py
│   ├── initialization.py          # trunc_normal_init_ and other init functions
│   ├── layers.py                  # Shared layer implementations
│   └── sparse_embedding.py        # Shared embedding code
│
├── hrm/                          # HRM-specific code
│   ├── __init__.py
│   └── hrm_model.py              # Renamed from hrm/hrm_act_v1.py
│
├── trm/                          # TRM-specific code
│   ├── __init__.py
│   ├── trm_model.py              # Renamed from recursive_reasoning/trm.py
│   └── trm_baseline.py           # Renamed from recursive_reasoning/transformers_baseline.py
│
├── heads/                        # Output head implementations
│   ├── __init__.py
│   ├── attention_policy.py       # Renamed from attention_policy_map.py
│   └── value_heads.py            # Renamed from tensorflow_style_heads.py
│
├── bridge.py                     # Single unified bridge (merged HRMBridge + HRMBridge2)
├── losses.py                     # All loss functions (from bridge + model subdirs)
├── simple_baseline.py            # Renamed from SimpleChessNet.py
└── utils.py                      # Keep as is (AverageMeter, dotdict)
```

### Key Principles

1. **Flat is better than nested**: Reduced nesting levels (hrm/ instead of HRM_model/hrm/)
2. **Shared code in common/**: No duplication between model types
3. **Clear separation**: Models, heads, losses, and bridge are separate
4. **Descriptive names**: File names immediately indicate purpose

---

## Detailed Migration Plan

### Phase 1: Consolidate Common Code

1. Create `model/common/` directory
2. Move `trunc_normal_init_` to `common/initialization.py`
3. Deduplicate `layers.py` and `sparse_embedding.py`
4. Update all imports to use the common versions
5. Delete duplicate files from `HRM_model/` and `trm_model/`

### Phase 2: Reorganize Model Implementations

1. Create `model/hrm/` and `model/trm/` directories
2. Move and rename:
   - `HRM_model/hrm/hrm_act_v1.py` → `hrm/hrm_model.py`
   - `trm_model/recursive_reasoning/trm.py` → `trm/trm_model.py`
   - `trm_model/recursive_reasoning/transformers_baseline.py` → `trm/trm_baseline.py`
3. Update imports in moved files

### Phase 3: Consolidate Bridges

1. Create `model/bridge.py`
2. Merge `HRMBridge.py` and `HRMBridge2.py` into single `AlphaZeroBridge` class
3. Move value activation logic into bridge (make configurable)
4. Update all references

### Phase 4: Consolidate Losses

1. Create `model/losses.py`
2. Move `SimpleHRMLoss` from `HRMBridge.py`
3. Check `HRM_model/losses.py` and `trm_model/losses.py` for other losses
4. Consolidate all into single file
5. Delete old loss files

### Phase 5: Organize Heads

1. Create `model/heads/` directory
2. Move and rename:
   - `attention_policy_map.py` → `heads/attention_policy.py`
   - `tensorflow_style_heads.py` → `heads/value_heads.py`
3. Update imports

### Phase 6: Remove Wrappers

1. Delete `ChessNNet.py`, `ChessTRMNet.py`, `ChessTRMBaselineNet.py`
2. Update training code to instantiate models directly:
   ```python
   # Old way
   model = ChessNNet(config)

   # New way
   hrm_model = HierarchicalReasoningModel(hrm_config)
   model = AlphaZeroBridge(hrm_model, apply_tanh=True)
   ```

### Phase 7: Clean Up Imports

1. Add `__init__.py` files to all directories
2. Export main classes from `__init__.py`
3. Remove all `sys.path.append()` calls
4. Use relative imports:
   ```python
   from model.common.initialization import trunc_normal_init_
   from model.hrm import HierarchicalReasoningModel
   from model.bridge import AlphaZeroBridge
   ```

### Phase 8: Documentation

1. Create `model/README.md` with:
   - Architecture overview
   - Differences between HRM, TRM, and TRM-baseline
   - Usage examples
   - Citation information
2. Add docstrings to main classes
3. Create architecture diagrams if helpful

---

## Benefits of Refactoring

### For Research Publication

- **Clear structure**: Reviewers and readers can quickly understand the codebase
- **Easy to cite**: Well-organized code is more likely to be used and cited
- **Reproducibility**: Reduced complexity makes it easier to reproduce results
- **Professional appearance**: Shows attention to software engineering quality

### For Maintenance

- **No duplication**: Bug fixes only need to be applied once
- **Easier debugging**: Clear separation of concerns
- **Simpler testing**: Each component can be tested independently
- **Better extensibility**: Adding new models or heads is straightforward

### For Collaboration

- **Lower barrier to entry**: New contributors can understand the structure quickly
- **Clear conventions**: Consistent naming and organization
- **Self-documenting**: Structure itself communicates intent
- **Modular**: Contributors can work on different components independently

---

## Testing Strategy

After each phase:

1. Run existing unit tests (if any)
2. Verify model can be instantiated
3. Run a small training step to ensure forward/backward pass works
4. Compare model outputs before/after refactoring (should be identical)

---

## Notes and Considerations

### Backward Compatibility

- Old checkpoint paths will break
- Training scripts will need updates
- Consider keeping a `legacy/` directory for old code during transition

### Timing

- This refactoring is best done **before** major publication
- Easier to reference clean code in papers
- Can be done incrementally (one phase at a time)

### Risk Mitigation

- Keep old code in a branch until refactoring is validated
- Write tests for critical functionality before refactoring
- Do a test run on a small dataset after each phase

---

## Open Questions

1. Are there other loss functions in `HRM_model/losses.py` and `trm_model/losses.py`?
2. Are `layers.py` and `sparse_embedding.py` actually duplicated?
3. Is there existing training code that depends on the wrapper classes?
4. Are there any performance implications of the refactoring?
5. Should EMA functionality in `trm_model/ema.py` be moved to common?

---

## Next Steps

1. Review this plan and adjust based on project needs
2. Create a new git branch for refactoring work
3. Implement Phase 1 (consolidate common code)
4. Test thoroughly
5. Continue with subsequent phases
6. Update documentation throughout

---

**Created**: 2025-12-15
**Status**: Proposal - awaiting review and approval
