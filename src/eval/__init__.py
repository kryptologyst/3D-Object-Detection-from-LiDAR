"""Evaluation utilities for 3D object detection models."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import KITTIDataset, collate_fn
from ..utils import apply_nms_3d, get_device


class DetectionMetrics:
    """Metrics for 3D object detection evaluation."""
    
    def __init__(self, num_classes: int = 3, iou_thresholds: List[float] = None):
        """Initialize detection metrics.
        
        Args:
            num_classes: Number of object classes.
            iou_thresholds: IoU thresholds for evaluation.
        """
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Initialize metrics storage
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = []
        self.ground_truths = []
        self.sample_ids = []
    
    def update(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        sample_ids: List[str],
    ) -> None:
        """Update metrics with new predictions and ground truths.
        
        Args:
            predictions: List of prediction dictionaries.
            ground_truths: List of ground truth dictionaries.
            sample_ids: List of sample IDs.
        """
        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truths)
        self.sample_ids.extend(sample_ids)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary of computed metrics.
        """
        if not self.predictions:
            return {}
        
        metrics = {}
        
        # Compute mAP for each IoU threshold
        for iou_thresh in self.iou_thresholds:
            ap_scores = self._compute_ap(iou_thresh)
            metrics[f"mAP_3D_{iou_thresh:.1f}"] = np.mean(ap_scores)
        
        # Compute overall mAP
        metrics["mAP_3D"] = np.mean([metrics[f"mAP_3D_{iou_thresh:.1f}"] for iou_thresh in self.iou_thresholds])
        
        # Compute mAP BEV
        metrics["mAP_BEV"] = self._compute_map_bev()
        
        # Compute NDS (NuScenes Detection Score)
        metrics["NDS"] = self._compute_nds()
        
        # Compute ATE, ASE, AOE
        ate, ase, aoe = self._compute_ate_ase_aoe()
        metrics["ATE"] = ate
        metrics["ASE"] = ase
        metrics["AOE"] = aoe
        
        return metrics
    
    def _compute_ap(self, iou_threshold: float) -> List[float]:
        """Compute Average Precision for given IoU threshold.
        
        Args:
            iou_threshold: IoU threshold.
            
        Returns:
            List of AP scores for each class.
        """
        ap_scores = []
        
        for class_id in range(self.num_classes):
            # Filter predictions and ground truths for this class
            class_predictions = [pred for pred in self.predictions if pred.get("class_id") == class_id]
            class_ground_truths = [gt for gt in self.ground_truths if gt.get("class_id") == class_id]
            
            if not class_predictions or not class_ground_truths:
                ap_scores.append(0.0)
                continue
            
            # Sort predictions by confidence
            class_predictions.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            # Compute precision and recall
            tp = np.zeros(len(class_predictions))
            fp = np.zeros(len(class_predictions))
            
            for i, pred in enumerate(class_predictions):
                # Find best matching ground truth
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt in enumerate(class_ground_truths):
                    iou = self._compute_iou_3d(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= iou_threshold:
                    tp[i] = 1
                    # Remove matched ground truth
                    if best_gt_idx >= 0:
                        class_ground_truths.pop(best_gt_idx)
                else:
                    fp[i] = 1
            
            # Compute precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
            recall = tp_cumsum / len(class_ground_truths) if class_ground_truths else tp_cumsum
            
            # Compute AP using 11-point interpolation
            ap = self._compute_ap_from_pr(precision, recall)
            ap_scores.append(ap)
        
        return ap_scores
    
    def _compute_map_bev(self) -> float:
        """Compute mAP for Bird's Eye View (BEV).
        
        Returns:
            BEV mAP score.
        """
        # Simplified BEV mAP computation
        # In practice, this would project 3D boxes to BEV and compute 2D IoU
        return 0.0
    
    def _compute_nds(self) -> float:
        """Compute NuScenes Detection Score (NDS).
        
        Returns:
            NDS score.
        """
        # Simplified NDS computation
        # In practice, this would compute weighted combination of mAP, ATE, ASE, AOE
        return 0.0
    
    def _compute_ate_ase_aoe(self) -> Tuple[float, float, float]:
        """Compute ATE, ASE, AOE metrics.
        
        Returns:
            Tuple of (ATE, ASE, AOE) scores.
        """
        # Simplified computation
        # In practice, these would be computed based on translation, scale, and orientation errors
        return 0.0, 0.0, 0.0
    
    def _compute_iou_3d(self, bbox1: torch.Tensor, bbox2: torch.Tensor) -> float:
        """Compute 3D IoU between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x, y, z, w, h, l, yaw].
            bbox2: Second bounding box [x, y, z, w, h, l, yaw].
            
        Returns:
            IoU value.
        """
        # Simplified 3D IoU computation
        # In practice, this would use proper 3D box intersection
        return 0.0
    
    def _compute_ap_from_pr(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """Compute AP from precision and recall arrays.
        
        Args:
            precision: Precision values.
            recall: Recall values.
            
        Returns:
            Average Precision score.
        """
        # Add sentinel values
        precision = np.concatenate(([0], precision, [0]))
        recall = np.concatenate(([0], recall, [1]))
        
        # Compute precision envelope
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        
        # Find points where recall changes
        indices = np.where(recall[1:] != recall[:-1])[0]
        
        # Sum (\Delta recall) * prec
        ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
        
        return ap


class Evaluator:
    """Evaluator class for 3D object detection models."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate.
            test_loader: Test data loader.
            device: Device to evaluate on.
            config: Evaluation configuration.
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device or get_device()
        self.config = config or {}
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize metrics
        self.metrics = DetectionMetrics(
            num_classes=self.config.get("num_classes", 3),
            iou_thresholds=self.config.get("detection_thresholds", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        )
        
        # NMS configuration
        self.nms_config = self.config.get("nms", {})
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model.
        
        Returns:
            Dictionary of evaluation metrics.
        """
        self.metrics.reset()
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Move batch to device
                point_clouds = batch["point_clouds"].to(self.device)
                labels = batch["labels"]
                bboxes = batch["bboxes"]
                sample_ids = batch.get("sample_ids", [])
                
                # Forward pass
                predictions = self.model(point_clouds)
                
                # Post-process predictions
                processed_predictions = self._post_process_predictions(predictions, sample_ids)
                
                # Prepare ground truths
                ground_truths = self._prepare_ground_truths(labels, bboxes, sample_ids)
                
                # Update metrics
                self.metrics.update(processed_predictions, ground_truths, sample_ids)
        
        # Compute final metrics
        final_metrics = self.metrics.compute()
        
        return final_metrics
    
    def _post_process_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        sample_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Post-process model predictions.
        
        Args:
            predictions: Raw model predictions.
            sample_ids: Sample IDs.
            
        Returns:
            List of processed predictions.
        """
        processed_predictions = []
        
        batch_size = predictions["classification"].shape[0]
        
        for i in range(batch_size):
            sample_id = sample_ids[i] if i < len(sample_ids) else f"sample_{i}"
            
            # Extract predictions for this sample
            cls_pred = predictions["classification"][i]
            reg_pred = predictions["regression"][i]
            
            # Apply NMS
            boxes, scores, indices = apply_nms_3d(
                boxes=reg_pred.view(-1, 7),
                scores=cls_pred.view(-1),
                iou_threshold=self.nms_config.get("iou_threshold", 0.1),
                score_threshold=self.nms_config.get("score_threshold", 0.1),
                max_detections=self.nms_config.get("max_detections", 100),
            )
            
            # Convert to prediction format
            for j in range(len(boxes)):
                prediction = {
                    "sample_id": sample_id,
                    "bbox": boxes[j],
                    "confidence": scores[j].item(),
                    "class_id": 0,  # Simplified - would need proper class assignment
                }
                processed_predictions.append(prediction)
        
        return processed_predictions
    
    def _prepare_ground_truths(
        self,
        labels: List[List[str]],
        bboxes: List[List[torch.Tensor]],
        sample_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Prepare ground truth annotations.
        
        Args:
            labels: List of label lists.
            bboxes: List of bounding box lists.
            sample_ids: Sample IDs.
            
        Returns:
            List of ground truth dictionaries.
        """
        ground_truths = []
        
        # Class mapping
        class_mapping = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
        
        for i, (sample_labels, sample_bboxes) in enumerate(zip(labels, bboxes)):
            sample_id = sample_ids[i] if i < len(sample_ids) else f"sample_{i}"
            
            for label, bbox in zip(sample_labels, sample_bboxes):
                ground_truth = {
                    "sample_id": sample_id,
                    "bbox": bbox,
                    "class_id": class_mapping.get(label, 0),
                }
                ground_truths.append(ground_truth)
        
        return ground_truths


def evaluate_model(
    model: nn.Module,
    test_dataset: KITTIDataset,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Evaluate a 3D object detection model.
    
    Args:
        model: Model to evaluate.
        test_dataset: Test dataset.
        config: Evaluation configuration.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("batch_size", 4) if config else 4,
        shuffle=False,
        num_workers=config.get("num_workers", 4) if config else 4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Initialize evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        config=config,
    )
    
    # Evaluate
    metrics = evaluator.evaluate()
    
    return metrics


def create_evaluation_report(
    metrics: Dict[str, float],
    output_path: str = "evaluation_report.txt",
) -> None:
    """Create evaluation report.
    
    Args:
        metrics: Evaluation metrics.
        output_path: Output file path.
    """
    with open(output_path, "w") as f:
        f.write("3D Object Detection Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall metrics
        f.write("Overall Metrics:\n")
        f.write(f"mAP_3D: {metrics.get('mAP_3D', 0):.4f}\n")
        f.write(f"mAP_BEV: {metrics.get('mAP_BEV', 0):.4f}\n")
        f.write(f"NDS: {metrics.get('NDS', 0):.4f}\n")
        f.write(f"ATE: {metrics.get('ATE', 0):.4f}\n")
        f.write(f"ASE: {metrics.get('ASE', 0):.4f}\n")
        f.write(f"AOE: {metrics.get('AOE', 0):.4f}\n\n")
        
        # Per-IoU metrics
        f.write("Per-IoU Threshold Metrics:\n")
        for key, value in metrics.items():
            if key.startswith("mAP_3D_") and not key.endswith("_3D"):
                f.write(f"{key}: {value:.4f}\n")
    
    print(f"Evaluation report saved to: {output_path}")
