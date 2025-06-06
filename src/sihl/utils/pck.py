from typing import Dict, List, Tuple

from torch import Tensor
from torchmetrics import Metric
import torch


class PercentageOfCorrectKeypoints(Metric):
    """
    Compute the Percentage of Correct Keypoints (PCK) metric for keypoint detection evaluation.

    PCK measures the percentage of predicted keypoints that are within a threshold distance
    from the ground truth keypoints. This implementation handles scenarios where there are
    N predictions and M ground truths that need to be optimally matched.
    """

    def __init__(self, threshold: float = 0.05) -> None:
        """
        Initialize the PCK metric.

        Args:
            threshold: Distance threshold for considering a keypoint as correct.
        """
        super().__init__()
        self.threshold = threshold
        self.add_state(
            "correct_keypoints", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("total_keypoints", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        pred_keypoints: Tensor,
        pred_presence: Tensor,
        gt_keypoints: Tensor,
        gt_presence: Tensor,
    ) -> None:
        """
        Update the metric state with predictions and targets.

        Args:
            pred_keypoints: Predicted keypoint coordinates of shape (N_pred, K, 2)
            pred_presence: Predicted keypoint presence of shape (N_pred, K)
            gt_keypoints: Ground truth keypoint coordinates of shape (N_gt, K, 2)
            gt_presence: Ground truth keypoint presence of shape (N_gt, K)
        """
        if pred_keypoints.shape[0] == 0 or gt_keypoints.shape[0] == 0:
            # Count all visible GT keypoints as missed if no predictions
            if gt_keypoints.shape[0] > 0:
                self.total_keypoints += gt_presence.sum()
            return

        N_pred, K, _ = pred_keypoints.shape
        N_gt = gt_keypoints.shape[0]

        # Find greedy assignment
        assignment = self._find_greedy_assignment(
            pred_keypoints, pred_presence, gt_keypoints, gt_presence
        )

        # Evaluate PCK for matched pairs
        for pred_idx, gt_idx in assignment:
            if pred_idx < N_pred and gt_idx < N_gt:
                pred_kpts = pred_keypoints[pred_idx]  # (K, 2)
                pred_vis = pred_presence[pred_idx]  # (K,)
                gt_kpts = gt_keypoints[gt_idx]  # (K, 2)
                gt_vis = gt_presence[gt_idx]  # (K,)

                # Only evaluate keypoints visible in ground truth
                visible_mask = gt_vis > 0

                if visible_mask.any():
                    # Compute distances for visible keypoints
                    distances = torch.norm(
                        pred_kpts[visible_mask] - gt_kpts[visible_mask], dim=-1
                    )

                    # Count correct predictions
                    correct = (distances <= self.threshold).sum()
                    total = visible_mask.sum()

                    self.correct_keypoints += correct
                    self.total_keypoints += total

        # Count unmatched ground truths as missed detections
        matched_gts = set(gt_idx for _, gt_idx in assignment if gt_idx < N_gt)
        for gt_idx in range(N_gt):
            if gt_idx not in matched_gts:
                unmatched_visible = (gt_presence[gt_idx] > 0).sum()
                self.total_keypoints += unmatched_visible

    def _find_greedy_assignment(
        self,
        pred_keypoints: Tensor,
        pred_presence: Tensor,
        gt_keypoints: Tensor,
        gt_presence: Tensor,
    ) -> List[Tuple[int, int]]:
        """
        Find greedy assignment between predictions and ground truths.

        Iteratively assigns the prediction-gt pair with the lowest average distance,
        ensuring each prediction and gt is used at most once.

        Returns:
            List of (pred_idx, gt_idx) tuples representing the assignment
        """
        N_pred = pred_keypoints.shape[0]
        N_gt = gt_keypoints.shape[0]
        device = pred_keypoints.device

        # Compute cost matrix based on average keypoint distances
        cost_matrix = torch.full((N_pred, N_gt), float("inf"), device=device)

        # Vectorized computation of cost matrix
        for i in range(N_pred):
            pred_vis = pred_presence[i] > 0  # (K,)
            pred_kpts = pred_keypoints[i]  # (K, 2)

            for j in range(N_gt):
                gt_vis = gt_presence[j] > 0  # (K,)
                gt_kpts = gt_keypoints[j]  # (K, 2)

                # Only consider keypoints visible in both pred and gt
                mutual_vis = pred_vis & gt_vis

                if mutual_vis.any():
                    # Compute average distance for mutually visible keypoints
                    distances = torch.norm(
                        pred_kpts[mutual_vis] - gt_kpts[mutual_vis], dim=-1
                    )
                    cost_matrix[i, j] = distances.mean()

        # Greedy assignment
        assignments = []
        used_preds = torch.zeros(N_pred, dtype=torch.bool, device=device)
        used_gts = torch.zeros(N_gt, dtype=torch.bool, device=device)

        while True:
            # Mask out already used predictions and ground truths
            available_costs = cost_matrix.clone()
            available_costs[used_preds, :] = float("inf")
            available_costs[:, used_gts] = float("inf")

            # Find minimum cost assignment
            min_cost = available_costs.min()

            if torch.isinf(min_cost):
                break  # No more valid assignments

            # Find indices of minimum cost
            min_indices = (available_costs == min_cost).nonzero(as_tuple=False)
            pred_idx, gt_idx = min_indices[0].tolist()  # Take first if multiple mins

            # Record assignment and mark as used
            assignments.append((pred_idx, gt_idx))
            used_preds[pred_idx] = True
            used_gts[gt_idx] = True

        return assignments

    def compute(self) -> Dict[str, float]:
        if self.total_keypoints == 0:
            return {"PCK": 0.0}

        pck = self.correct_keypoints.float() / self.total_keypoints.float()
        return {"PCK": pck.item()}

    def reset(self) -> None:
        self.correct_keypoints = torch.tensor(0)
        self.total_keypoints = torch.tensor(0)
