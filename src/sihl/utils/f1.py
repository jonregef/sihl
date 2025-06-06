import torch
from torchmetrics import Metric
from torchvision.ops import box_iou
from typing import Union, Dict


class OptimalF1Threshold(Metric):
    """
    A TorchMetrics-style class that efficiently computes the optimal score threshold
    that maximizes the F1 score for object detection.

    Memory-optimized implementation that processes data in chunks to avoid OOM errors.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        class_metrics: bool = False,
        threshold_granularity: int = 10,
        chunk_size: int = 512,  # Process predictions in chunks to reduce memory
        dist_sync_on_step: bool = False,
    ):
        """
        Initialize the OptimalF1Threshold metric.

        Args:
            iou_threshold: IoU threshold to consider a prediction as a match for a ground truth box
            class_metrics: Whether to compute per-class metrics
            threshold_granularity: Number of threshold values to test between 0 and 1
            chunk_size: Number of predictions to process at once to reduce memory usage
            dist_sync_on_step: Synchronize metric state across processes at each forward() call
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.iou_threshold = iou_threshold
        self.class_metrics = class_metrics
        self.threshold_granularity = threshold_granularity
        self.chunk_size = chunk_size

        # Store data as lists instead of concatenating tensors which can lead to OOM
        self.add_state("preds_classes_list", default=[], dist_reduce_fx=None)
        self.add_state("preds_scores_list", default=[], dist_reduce_fx=None)
        self.add_state("preds_boxes_list", default=[], dist_reduce_fx=None)
        self.add_state("target_classes_list", default=[], dist_reduce_fx=None)
        self.add_state("target_boxes_list", default=[], dist_reduce_fx=None)

    def update(
        self,
        preds_classes: torch.Tensor,
        preds_scores: torch.Tensor,
        preds_boxes: torch.Tensor,
        target_classes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> None:
        """
        Update state with new predictions and targets.

        Args:
            preds_classes: (n,) tensor of predicted class IDs
            preds_scores: (n,) tensor of predicted confidence scores
            preds_boxes: (n, 4) tensor of predicted bounding boxes in xyxy format
            target_classes: (m,) tensor of ground truth class IDs
            target_boxes: (m, 4) tensor of ground truth bounding boxes in xyxy format
        """
        # Validate input shapes
        assert preds_classes.dim() == 1, "preds_classes should be a 1D tensor"
        assert preds_scores.dim() == 1, "preds_scores should be a 1D tensor"
        assert preds_boxes.dim() == 2 and preds_boxes.shape[1] == 4, (
            "preds_boxes should be a 2D tensor of shape (n, 4)"
        )
        assert target_classes.dim() == 1, "target_classes should be a 1D tensor"
        assert target_boxes.dim() == 2 and target_boxes.shape[1] == 4, (
            "target_boxes should be a 2D tensor of shape (m, 4)"
        )

        # Verify lengths match
        assert (
            preds_classes.shape[0] == preds_scores.shape[0] == preds_boxes.shape[0]
        ), (
            "Number of predictions must match across preds_classes, preds_scores, and preds_boxes"
        )
        assert target_classes.shape[0] == target_boxes.shape[0], (
            "Number of targets must match across target_classes and target_boxes"
        )

        # Store inputs as CPU tensors to save GPU memory
        self.preds_classes_list.append(preds_classes.detach().cpu())
        self.preds_scores_list.append(preds_scores.detach().cpu())
        self.preds_boxes_list.append(preds_boxes.detach().cpu())
        self.target_classes_list.append(target_classes.detach().cpu())
        self.target_boxes_list.append(target_boxes.detach().cpu())

    def _prepare_data(self):
        """
        Prepare and preprocess the data for computation.

        Returns:
            Dictionary with preprocessed data
        """
        # Concatenate all predictions and targets
        if not self.preds_classes_list or len(self.preds_classes_list) == 0:
            return None

        all_preds_classes = torch.cat(self.preds_classes_list)
        all_preds_scores = torch.cat(self.preds_scores_list)
        all_preds_boxes = torch.cat(self.preds_boxes_list)
        all_target_classes = torch.cat(self.target_classes_list)
        all_target_boxes = torch.cat(self.target_boxes_list)

        # Handle no predictions or targets case
        if all_preds_classes.numel() == 0 or all_target_classes.numel() == 0:
            return None

        # Get unique class IDs
        unique_classes = (
            torch.cat([all_preds_classes, all_target_classes]).unique().tolist()
        )

        # Sort predictions by score in descending order for efficient threshold processing
        sorted_indices = torch.argsort(all_preds_scores, descending=True)
        sorted_scores = all_preds_scores[sorted_indices]
        sorted_classes = all_preds_classes[sorted_indices]
        sorted_boxes = all_preds_boxes[sorted_indices]

        # Generate thresholds
        if self.threshold_granularity < sorted_scores.size(0):
            # Use a subset of thresholds
            indices = torch.linspace(
                0, sorted_scores.size(0) - 1, self.threshold_granularity
            ).long()
            thresholds = sorted_scores[indices]
        else:
            # Use all unique scores as thresholds
            thresholds = torch.unique(sorted_scores)

        # Count ground truths per class if needed
        class_total_gt = {}
        if self.class_metrics:
            for cls in unique_classes:
                class_total_gt[cls] = (all_target_classes == cls).sum().item()

        return {
            "sorted_scores": sorted_scores,
            "sorted_classes": sorted_classes,
            "sorted_boxes": sorted_boxes,
            "target_classes": all_target_classes,
            "target_boxes": all_target_boxes,
            "unique_classes": unique_classes,
            "thresholds": thresholds,
            "class_total_gt": class_total_gt,
            "total_gt": all_target_classes.size(0),
        }

    def compute(self) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Efficiently compute the optimal threshold and corresponding F1 score.

        This implementation processes predictions in chunks to reduce memory usage.

        Returns:
            A dictionary containing:
            - optimal_threshold: The score threshold that maximizes F1
            - best_f1: The maximum F1 score
            - precision: Precision at the optimal threshold
            - recall: Recall at the optimal threshold
            - class_metrics: (Optional) Per-class metrics if class_metrics=True
        """
        # Prepare data
        data = self._prepare_data()

        # Handle empty case
        if data is None:
            return {
                "optimal_threshold": 0.0,
                "best_f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "class_metrics": {} if self.class_metrics else None,
            }

        # Variables for tracking best results
        best_f1 = 0.0
        optimal_threshold = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_class_metrics = {}

        # Extract data
        sorted_scores = data["sorted_scores"]
        sorted_classes = data["sorted_classes"]
        sorted_boxes = data["sorted_boxes"]
        target_classes = data["target_classes"]
        target_boxes = data["target_boxes"]
        unique_classes = data["unique_classes"]
        thresholds = data["thresholds"]
        class_total_gt = data["class_total_gt"]
        total_gt = data["total_gt"]

        # Get device to use
        device = sorted_scores.device

        # Process each threshold efficiently
        for threshold in thresholds:
            # All predictions with sorted_scores >= threshold
            # This uses the fact that scores are already sorted in descending order
            k = (sorted_scores >= threshold).sum().item()

            if k == 0:  # No predictions above threshold
                continue

            # Initialize counters
            tp = 0
            class_tp = {cls: 0 for cls in unique_classes} if self.class_metrics else {}
            class_preds = (
                {cls: 0 for cls in unique_classes} if self.class_metrics else {}
            )

            # Compute class prediction counts above threshold
            if self.class_metrics:
                for cls in sorted_classes[:k].tolist():
                    class_preds[cls] = class_preds.get(cls, 0) + 1

            # Track which GT boxes have been matched
            gt_matched = torch.zeros(total_gt, dtype=torch.bool, device=device)

            # Process predictions in chunks to save memory
            for chunk_start in range(0, k, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, k)
                chunk_pred_boxes = sorted_boxes[chunk_start:chunk_end]
                chunk_pred_classes = sorted_classes[chunk_start:chunk_end]

                # Calculate IoU matrix for this chunk
                # Shape: [chunk_size, num_targets]
                iou_matrix = box_iou(chunk_pred_boxes, target_boxes)

                # Create class matching matrix for this chunk
                # Shape: [chunk_size, num_targets]
                class_match_matrix = chunk_pred_classes.unsqueeze(
                    1
                ) == target_classes.unsqueeze(0)

                # Mask out IoUs where classes don't match
                iou_matrix = iou_matrix * class_match_matrix.float()

                # Apply IoU threshold mask
                valid_iou_mask = iou_matrix >= self.iou_threshold

                # Process each prediction in the chunk in order of confidence
                for pred_idx in range(chunk_end - chunk_start):
                    global_pred_idx = chunk_start + pred_idx
                    pred_cls = chunk_pred_classes[pred_idx].item()

                    # Find unmatched GTs with valid IoUs for this prediction
                    valid_ious = valid_iou_mask[pred_idx]
                    valid_and_unmatched = valid_ious & ~gt_matched
                    valid_gt_indices = torch.where(valid_and_unmatched)[0]

                    if len(valid_gt_indices) > 0:
                        # Find the GT with the highest IoU
                        if len(valid_gt_indices) == 1:
                            # Optimization: if only one valid GT, no need for argmax
                            max_iou_idx = valid_gt_indices[0]
                        else:
                            # Get IoUs for valid GTs
                            valid_gt_ious = iou_matrix[pred_idx, valid_gt_indices]
                            max_iou_pos = valid_gt_ious.argmax()
                            max_iou_idx = valid_gt_indices[max_iou_pos]

                        # Mark this GT as matched
                        gt_matched[max_iou_idx] = True

                        # Increment TP counts
                        tp += 1
                        if self.class_metrics:
                            class_tp[pred_cls] += 1

                # Free memory
                del iou_matrix, class_match_matrix, valid_iou_mask

            # Calculate metrics
            precision = tp / k if k > 0 else 0
            recall = tp / total_gt if total_gt > 0 else 0

            # Calculate F1 score with safe division
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)

            # Update best F1 and threshold
            if f1 > best_f1:
                best_f1 = f1
                optimal_threshold = threshold.item()
                best_precision = precision
                best_recall = recall

                # Compute per-class metrics at the best threshold if needed
                if self.class_metrics:
                    best_class_metrics = {}
                    for cls in unique_classes:
                        cls_precision = class_tp.get(cls, 0) / max(
                            class_preds.get(cls, 0), 1
                        )
                        cls_recall = class_tp.get(cls, 0) / max(
                            class_total_gt.get(cls, 0), 1
                        )
                        cls_f1 = (
                            2
                            * cls_precision
                            * cls_recall
                            / max(cls_precision + cls_recall, 1e-10)
                        )

                        best_class_metrics[cls] = {
                            "precision": cls_precision,
                            "recall": cls_recall,
                            "f1": cls_f1,
                        }

        # Prepare result dictionary
        result = {
            "optimal_threshold": optimal_threshold,
            "best_f1": best_f1,
            "precision": best_precision,
            "recall": best_recall,
        }

        if self.class_metrics:
            result["class_metrics"] = best_class_metrics

        return result

    def reset(self) -> None:
        """Reset metric states."""
        super().reset()
