from typing import Dict, List, Union, Any

from torch import Tensor
import torch
import numpy as np
from torchmetrics import Metric


class ObjectKeypointSimilarity(Metric):
    """
    Compute the Object Keypoint Similarity (OKS) metric for keypoint detection evaluation.

    OKS measures the similarity between predicted and ground truth keypoints, accounting
    for object scale and keypoint visibility.
    """

    def __init__(self):
        """
        Initialize the OKS metric.

        Args:
            sigma: Scale parameter controlling the falloff of the similarity measure
            thresholds: List of OKS thresholds for evaluation. Defaults to [0.5, 0.55, ..., 0.95]
        """
        super().__init__()
        self.sigma = 0.07
        self.thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.add_state("detection_keypoints", default=[], dist_reduce_fx=None)
        self.add_state("detection_scores", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_keypoints", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_areas", default=[], dist_reduce_fx=None)

    def update(
        self, preds: List[Dict[str, Tensor]], targets: List[Dict[str, Tensor]]
    ) -> None:
        """
        Update the metric state with predictions and targets.

        Args:
            preds: List of dictionaries containing predicted keypoints and scores
                  Each dict should have 'keypoints' (N_pred, K, 3) and 'scores' (N_pred,)
            targets: List of dictionaries containing ground truth keypoints
                    Each dict should have 'keypoints' (N_gt, K, 3), optional 'area' or 'boxes'
        """
        for pred in preds:
            self.detection_keypoints.append(pred["keypoints"])
            self.detection_scores.append(pred["scores"])

        for target in targets:
            self.groundtruth_keypoints.append(target["keypoints"])

            if "area" in target:
                areas = target["area"]
            elif "boxes" in target:
                boxes = target["boxes"]
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            else:
                areas = torch.tensor(
                    [self._estimate_area(kpts) for kpts in target["keypoints"]],
                    device=target["keypoints"].device,
                    dtype=torch.float32,
                )
            self.groundtruth_areas.append(areas)

    def _estimate_area(self, keypoints: Tensor) -> float:
        """
        Estimate object area based on the bounding box of visible keypoints.

        Args:
            keypoints: Tensor of shape (K, 3) with keypoints in x, y, visibility format

        Returns:
            Estimated area of the keypoints' bounding box. Returns 1.0 if no visible keypoints.
        """
        vis = keypoints[:, 2] > 0
        if not vis.any():
            return 1.0

        visible_pts = keypoints[vis, :2]
        min_xy = torch.min(visible_pts, dim=0)[0]
        max_xy = torch.max(visible_pts, dim=0)[0]
        area = float(torch.prod(max_xy - min_xy))
        return max(area, 1.0)  # Ensure area is never zero

    def _compute_oks_matrix(
        self, pred_kpts: Tensor, gt_kpts: Tensor, gt_areas: Tensor
    ) -> Tensor:
        """
        Compute OKS matrix between all predictions and ground truths.

        Args:
            pred_kpts: Tensor of shape (N_pred, K, 3) with predicted keypoints
            gt_kpts: Tensor of shape (N_gt, K, 3) with ground truth keypoints
            gt_areas: Tensor of shape (N_gt,) with ground truth areas

        Returns:
            Tensor of shape (N_pred, N_gt) with OKS values
        """
        N_pred, K, _ = pred_kpts.shape
        N_gt = gt_kpts.shape[0]

        pred_xy = pred_kpts[:, None, :, :2]  # (N_pred, 1, K, 2)
        gt_xy = gt_kpts[None, :, :, :2]  # (1, N_gt, K, 2)
        vis = gt_kpts[None, :, :, 2] > 0  # (1, N_gt, K)

        # Compute squared distances between all predictions and ground truths
        squared_dist = ((pred_xy - gt_xy) ** 2).sum(dim=-1)  # (N_pred, N_gt, K)

        # Compute OKS per keypoint
        sigma_squared = (self.sigma**2) * 2
        oks_per_keypoint = torch.exp(
            -squared_dist / (gt_areas[None, :, None] * sigma_squared)
        )  # (N_pred, N_gt, K)

        # Apply visibility mask and compute mean OKS
        oks_matrix = torch.zeros((N_pred, N_gt), device=pred_kpts.device)

        for i in range(N_pred):
            for j in range(N_gt):
                keypoint_vis = vis[0, j]  # (K,)
                if not keypoint_vis.any():
                    oks_matrix[i, j] = 0.0
                    continue
                valid_oks = oks_per_keypoint[i, j][keypoint_vis]
                oks_matrix[i, j] = valid_oks.mean()

        return oks_matrix

    def _compute_matches(
        self,
        threshold: float,
        oks_matrix: Tensor,
        pred_scores: Tensor,
        num_preds: int,
        num_gts: int,
    ) -> List[Dict[str, Any]]:
        """
        Compute matches between predictions and ground truths at given threshold.

        Args:
            threshold: OKS threshold for considering a match
            oks_matrix: Tensor of shape (N_pred, N_gt) with OKS values
            pred_scores: Tensor of shape (N_pred,) with prediction confidence scores
            num_preds: Number of predictions
            num_gts: Number of ground truths

        Returns:
            List of match dictionaries with match information
        """
        matches = []
        unmatched_preds = set(range(num_preds))
        unmatched_gts = set(range(num_gts))

        oks_copy = oks_matrix.clone()
        oks_copy[oks_copy < threshold] = 0.0

        # Greedy matching
        while unmatched_preds and unmatched_gts:
            valid_oks = oks_copy[list(unmatched_preds), :][:, list(unmatched_gts)]
            if valid_oks.numel() == 0 or valid_oks.max() <= 0:
                break

            # Get indices in the subset, then convert to original indices
            max_val, max_idx = valid_oks.reshape(-1).max(0)
            max_idx = max_idx.item()
            pred_idx = list(unmatched_preds)[max_idx // len(unmatched_gts)]
            gt_idx = list(unmatched_gts)[max_idx % len(unmatched_gts)]

            matches.append(
                {
                    "pred_idx": pred_idx,
                    "gt_idx": gt_idx,
                    "oks": float(oks_matrix[pred_idx, gt_idx]),
                    "score": float(pred_scores[pred_idx]),
                }
            )

            unmatched_preds.remove(pred_idx)
            unmatched_gts.remove(gt_idx)
            oks_copy[pred_idx, :] = 0
            oks_copy[:, gt_idx] = 0

        for pred_idx in unmatched_preds:
            matches.append(
                {
                    "pred_idx": pred_idx,
                    "gt_idx": -1,
                    "oks": 0.0,
                    "score": float(pred_scores[pred_idx]),
                }
            )

        for gt_idx in unmatched_gts:
            matches.append({"pred_idx": -1, "gt_idx": gt_idx, "oks": 0.0, "score": 0.0})

        return matches

    def compute(self) -> Dict[str, Union[float, Tensor]]:
        total_gt = sum(g.shape[0] for g in self.groundtruth_keypoints)
        if total_gt == 0:
            return {"AP": 0.0, "AP@0.5": 0.0, "AP@0.75": 0.0, "AR": 0.0}

        # Process each threshold separately
        results_by_threshold = {}
        for threshold in self.thresholds:
            all_matches = []
            for i in range(len(self.detection_keypoints)):
                pred_kpts = self.detection_keypoints[i]  # (N_pred, K, 3)
                pred_scores = self.detection_scores[i]  # (N_pred,)
                gt_kpts = self.groundtruth_keypoints[i]  # (N_gt, K, 3)
                gt_areas = self.groundtruth_areas[i]  # (N_gt,)

                num_preds = pred_kpts.shape[0]
                num_gts = gt_kpts.shape[0]

                if num_preds == 0 or num_gts == 0:
                    continue

                # Sort predictions by score in descending order
                order = torch.argsort(pred_scores, descending=True)
                pred_kpts = pred_kpts[order]
                pred_scores = pred_scores[order]

                oks_matrix = self._compute_oks_matrix(pred_kpts, gt_kpts, gt_areas)
                matches = self._compute_matches(
                    threshold, oks_matrix, pred_scores, num_preds, num_gts
                )
                all_matches.extend(matches)
            all_matches.sort(key=lambda x: x["score"], reverse=True)
            results_by_threshold[threshold] = all_matches

        ap_list = []
        ar_list = []
        for threshold, matches in results_by_threshold.items():
            if not matches:
                ap_list.append(0.0)
                ar_list.append(0.0)
                continue

            # Compute precision-recall curve
            tp = 0
            fp = 0
            precisions = []
            recalls = []

            for match in matches:
                if match["pred_idx"] != -1:  # Actual prediction (not an unmatched GT)
                    if match["gt_idx"] != -1:
                        tp += 1
                    else:
                        fp += 1

                    precisions.append(tp / (tp + fp))
                    recalls.append(tp / total_gt)

            if not precisions:
                ap_list.append(0.0)
                ar_list.append(0.0)
                continue

            # Compute AP using 11-point interpolation
            ap = 0.0
            for recall_level in np.linspace(0, 1, 11):
                # Find max precision at recall >= recall_level
                max_precision = 0.0
                for i in range(len(recalls)):
                    if recalls[i] >= recall_level:
                        max_precision = max(max_precision, precisions[i])
                ap += max_precision / 11

            ap_list.append(ap)
            ar_list.append(tp / total_gt if total_gt > 0 else 0.0)

        metrics = {"mAP": np.mean(ap_list), "mAR": np.mean(ar_list)}
        if 0.5 in self.thresholds:
            metrics["AP@0.5"] = ap_list[self.thresholds.index(0.5)]
        if 0.75 in self.thresholds:
            metrics["AP@0.75"] = ap_list[self.thresholds.index(0.75)]

        return metrics

    def reset(self) -> None:
        self.detection_keypoints.clear()
        self.detection_scores.clear()
        self.groundtruth_keypoints.clear()
        self.groundtruth_areas.clear()
