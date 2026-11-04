"""Explicit slice-level exclusion semantics; no change to legacy default AND."""
import numpy as np


def excluded_slices(labels, excluded_labels, mode='all'):
    if mode not in ('all', 'any'):
        raise ValueError('exclude_label_mode must be all (legacy AND) or any (OR)')
    if not excluded_labels:
        return np.zeros(labels.shape[0], dtype=bool)
    presence = np.stack([(labels == value).any(axis=(1, 2)) for value in excluded_labels])
    return presence.all(axis=0) if mode == 'all' else presence.any(axis=0)
