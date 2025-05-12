from typing import Dict, List, Union, Any

from torch import Tensor
import torch
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
        self.thresholds = torch.tensor(
            [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        )
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
                areas = torch.stack(
                    [self._estimate_area(kpts) for kpts in target["keypoints"]]
                ).to(device=target["keypoints"].device)
            self.groundtruth_areas.append(areas)

    def _estimate_area(self, keypoints: Tensor) -> Tensor:
        """
        Estimate object area based on the bounding box of visible keypoints.

        Args:
            keypoints: Tensor of shape (K, 3) with keypoints in x, y, visibility format

        Returns:
            Estimated area of the keypoints' bounding box. Returns 1.0 if no visible keypoints.
        """
        vis = keypoints[:, 2] > 0
        if not vis.any():
            return torch.tensor(1.0, device=keypoints.device)

        visible_pts = keypoints[vis, :2]
        min_xy = torch.min(visible_pts, dim=0)[0]
        max_xy = torch.max(visible_pts, dim=0)[0]
        area = torch.prod(max_xy - min_xy)
        return torch.max(
            area, torch.tensor(1.0, device=keypoints.device)
        )  # Ensure area is never zero

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
        vis_expanded = vis.expand(N_pred, -1, -1)  # (N_pred, N_gt, K)
        valid_oks = oks_per_keypoint * vis_expanded.float()
        valid_counts = vis_expanded.sum(dim=-1)  # (N_pred, N_gt)

        # Avoid division by zero - set OKS to 0 where no visible keypoints
        oks_matrix = torch.where(
            valid_counts > 0,
            valid_oks.sum(dim=-1) / valid_counts,
            torch.zeros_like(valid_counts, dtype=torch.float32),
        )

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
        device = oks_matrix.device

        # Initialize unmatched lists
        unmatched_preds = torch.arange(num_preds, device=device)
        unmatched_gts = torch.arange(num_gts, device=device)

        # Create a mask for valid matches
        valid_mask = oks_matrix >= threshold

        while len(unmatched_preds) > 0 and len(unmatched_gts) > 0:
            # Get the submatrix for unmatched predictions and ground truths
            submatrix = oks_matrix[unmatched_preds][:, unmatched_gts]
            sub_valid_mask = valid_mask[unmatched_preds][:, unmatched_gts]

            if not sub_valid_mask.any():
                break

            # Find the maximum OKS in the submatrix
            # First flatten the matrix to find global max
            flat_submatrix = (submatrix * sub_valid_mask.float()).flatten()
            max_val, flat_max_idx = torch.max(flat_submatrix, dim=0)
            max_val = max_val.item()

            if max_val <= 0:
                break

            # Convert flat index to 2D indices
            gt_sub_size = len(unmatched_gts)
            pred_sub_idx = flat_max_idx // gt_sub_size
            gt_sub_idx = flat_max_idx % gt_sub_size

            # Get original indices
            pred_idx = unmatched_preds[pred_sub_idx].item()
            gt_idx = unmatched_gts[gt_sub_idx].item()

            matches.append(
                {
                    "pred_idx": pred_idx,
                    "gt_idx": gt_idx,
                    "oks": oks_matrix[pred_idx, gt_idx].item(),
                    "score": pred_scores[pred_idx].item(),
                }
            )

            # Remove matched indices
            unmatched_preds = unmatched_preds[unmatched_preds != pred_idx]
            unmatched_gts = unmatched_gts[unmatched_gts != gt_idx]

        # Add unmatched predictions
        for pred_idx in unmatched_preds.tolist():
            matches.append(
                {
                    "pred_idx": pred_idx,
                    "gt_idx": -1,
                    "oks": 0.0,
                    "score": pred_scores[pred_idx].item(),
                }
            )

        # Add unmatched ground truths
        for gt_idx in unmatched_gts.tolist():
            matches.append(
                {
                    "pred_idx": -1,
                    "gt_idx": gt_idx,
                    "oks": 0.0,
                    "score": 0.0,
                }
            )

        return matches

    def compute(self) -> Dict[str, Union[float, Tensor]]:
        total_gt = sum(g.shape[0] for g in self.groundtruth_keypoints)
        if total_gt == 0:
            return {
                "mAP": torch.tensor(0.0),
                "AP@0.5": torch.tensor(0.0),
                "AP@0.75": torch.tensor(0.0),
                "mAR": torch.tensor(0.0),
            }

        # Convert thresholds to tensor on the correct device
        if len(self.detection_keypoints) > 0:
            device = self.detection_keypoints[0].device
            thresholds = self.thresholds.to(device)
        else:
            device = (
                self.groundtruth_keypoints[0].device
                if len(self.groundtruth_keypoints) > 0
                else "cpu"
            )
            thresholds = self.thresholds.to(device)

        # Process each threshold separately
        ap_by_threshold = {}
        ar_by_threshold = {}

        for threshold in thresholds:
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

            # Filter out unmatched ground truths and sort by score
            all_matches = [m for m in all_matches if m["pred_idx"] != -1]
            all_matches.sort(key=lambda x: x["score"], reverse=True)

            # Calculate precision-recall curve
            tp = 0
            fp = 0
            precisions = []
            recalls = []

            for match in all_matches:
                if match["gt_idx"] != -1 and match["oks"] >= threshold:
                    tp += 1
                else:
                    fp += 1

                precisions.append(tp / (tp + fp))
                recalls.append(tp / total_gt)

            # Calculate AP using precision-recall curve
            if not precisions:
                ap_by_threshold[threshold.item()] = 0.0
                ar_by_threshold[threshold.item()] = 0.0
                continue

            # Convert to tensors for computation
            precisions = torch.tensor(precisions, device=device)
            recalls = torch.tensor(recalls, device=device)

            # Ensure precision is monotonically decreasing
            for i in range(len(precisions) - 1, 0, -1):
                precisions[i - 1] = torch.max(precisions[i - 1], precisions[i])

            # Calculate AP as area under precision-recall curve
            ap = torch.tensor(0.0, device=device)
            for i in range(1, len(precisions)):
                if recalls[i] != recalls[i - 1]:
                    ap += (recalls[i] - recalls[i - 1]) * precisions[i]

            ap_by_threshold[threshold.item()] = ap.item()
            ar_by_threshold[threshold.item()] = (
                recalls[-1].item() if len(recalls) > 0 else 0.0
            )

        # Calculate final metrics
        map_value = torch.tensor(
            sum(ap_by_threshold.values()) / len(ap_by_threshold), device=device
        )
        mar_value = torch.tensor(
            sum(ar_by_threshold.values()) / len(ar_by_threshold), device=device
        )

        metrics = {
            "mAP": map_value,
            "mAR": mar_value,
            "AP@0.50": torch.tensor(ap_by_threshold[0.5], device=device),
            "AP@0.75": torch.tensor(ap_by_threshold[0.75], device=device),
        }

        return metrics

    def reset(self) -> None:
        self.detection_keypoints.clear()
        self.detection_scores.clear()
        self.groundtruth_keypoints.clear()
        self.groundtruth_areas.clear()
