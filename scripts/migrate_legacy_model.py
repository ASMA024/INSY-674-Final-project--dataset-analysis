"""One-off: convert the notebook joblib (which pickled ``__main__.TelcoFeatures``)
into one that points at ``churn.features.TelcoFeatures``.

    python scripts/migrate_legacy_model.py old_churn_model.joblib models/churn_model.joblib

Only run this on model files you created yourself: unpickling runs code.
"""

import sys

import joblib

import __main__
from churn.features import TelcoFeatures

__main__.TelcoFeatures = TelcoFeatures  # lets pickle resolve the old reference

src, dst = sys.argv[1], sys.argv[2]
bundle = joblib.load(src)
joblib.dump(bundle, dst)  # re-pickled with module = churn.features
print(f"Migrated {src} -> {dst}")
