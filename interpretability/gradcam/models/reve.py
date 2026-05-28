import numpy as np
import torch

from interpretability.gradcam.cam import compute_gradcam
from interpretability.common.predict import extract_preds_and_confidence, unwrap_output


def run(model, X_test, batch_size, ch_names, channels,
        *, y_test=None, target_layer=-1, **_):
    """GradCAM for REVE.  ``channels`` ignored."""
    n_channels = len(ch_names)
    relevances, predictions, confidences = [], [], []
    for start in range(0, len(X_test), batch_size):
        x = torch.from_numpy(X_test[start : start + batch_size]).float().cuda()

        with torch.no_grad():
            out = unwrap_output(model(x.detach()))
            preds_batch, conf_batch = extract_preds_and_confidence(out)
            predictions.append(preds_batch)
            confidences.append(conf_batch)

        target = (
            torch.tensor(y_test[start : start + batch_size], dtype=torch.long, device=x.device)
            if y_test is not None else None
        )

        rel = compute_gradcam(
            "REVE", model, x,
            n_channels=n_channels,
            target_class=target,
            target_layer=target_layer,
            return_time_patches=True,
        )
        relevances.append(rel)

    return np.concatenate(relevances), np.concatenate(predictions), np.concatenate(confidences)
