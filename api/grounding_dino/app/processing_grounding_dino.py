from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from transformers.image_transforms import center_to_corners_format
from transformers.models.grounding_dino.processing_grounding_dino import (
    DictWithDeprecationWarning,
    GroundingDinoProcessor,
    get_phrases_from_posmap,
)
from transformers.utils import TensorType, is_torch_available
from transformers.utils.deprecation import deprecate_kwarg

if is_torch_available():
    import torch

if TYPE_CHECKING:
    from transformers.models.grounding_dino.modeling_grounding_dino import (
        GroundingDinoObjectDetectionOutput,
    )


AnnotationType = Dict[str, Union[int, str, List[Dict]]]


class BatchGroundingDinoProcessor(GroundingDinoProcessor):
    @deprecate_kwarg("box_thresholds", new_name="threshold", version="4.51.0")
    def post_process_grounded_object_detection(
        self,
        outputs: "GroundingDinoObjectDetectionOutput",
        input_ids: Optional[TensorType] = None,
        thresholds: Union[float, List[float], TensorType] = 0.25,
        text_thresholds: Union[float, List[float], TensorType] = 0.25,
        target_sizes: Optional[Union[TensorType, List[Tuple]]] = None,
        text_labels: Optional[List[List[str]]] = None,
    ):
        """
        Converts the raw output of [`GroundingDinoForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format and get the associated text label.

        Args:
            outputs ([`GroundingDinoObjectDetectionOutput`]):
                Raw outputs of the model.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The token ids of the input text. If not provided will be taken from the model output.
            thresholds (`float`, `List[float]`, or `torch.Tensor`, *optional*, defaults to 0.25):
                Threshold(s) to keep object detection predictions based on confidence score. Can be a single float,
                a list of floats (one per batch), or a tensor of shape `(batch_size,)`.
            text_thresholds (`float`, `List[float]`, or `torch.Tensor`, *optional*, defaults to 0.25):
                Score threshold(s) to keep text detection predictions. Can be a single float, a list of floats
                (one per batch), or a tensor of shape `(batch_size,)`.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
            text_labels (`List[List[str]]`, *optional*):
                List of candidate labels to be detected on each image. At the moment it's *NOT used*, but required
                to be in signature for the zero-shot object detection pipeline. Text labels are instead extracted
                from the `input_ids` tensor provided in `outputs`.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the
                - **scores**: tensor of confidence scores for detected objects
                - **boxes**: tensor of bounding boxes in [x0, y0, x1, y1] format
                - **labels**: list of text labels for each detected object (will be replaced with integer ids in v4.51.0)
                - **text_labels**: list of text labels for detected objects
        """
        batch_logits, batch_boxes = outputs.logits, outputs.pred_boxes
        input_ids = input_ids if input_ids is not None else outputs.input_ids
        # if thresholds is a single value, convert it to a list of the same value
        if isinstance(thresholds, float):
            thresholds = [thresholds] * batch_logits.shape[0]
        if isinstance(text_thresholds, float):
            text_thresholds = [text_thresholds] * batch_logits.shape[0]

        if target_sizes is not None and len(target_sizes) != len(batch_logits):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")

        batch_probs = torch.sigmoid(batch_logits)  # (batch_size, num_queries, 256)
        batch_scores = torch.max(batch_probs, dim=-1)[0]  # (batch_size, num_queries)

        # Convert to [x0, y0, x1, y1] format
        batch_boxes = center_to_corners_format(batch_boxes)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(batch_boxes.device)
            batch_boxes = batch_boxes * scale_fct[:, None, :]

        results = []
        for idx, (scores, boxes, probs, threshold, text_threshold) in enumerate(
            zip(batch_scores, batch_boxes, batch_probs, thresholds, text_thresholds, strict=False)
        ):
            keep = scores > threshold
            scores = scores[keep]
            boxes = boxes[keep]

            # extract text labels
            prob = probs[keep]
            label_ids = get_phrases_from_posmap(prob > text_threshold, input_ids[idx])
            objects_text_labels = self.batch_decode(label_ids)

            result = DictWithDeprecationWarning(
                {
                    "scores": scores,
                    "boxes": boxes,
                    "text_labels": objects_text_labels,
                    # TODO: @pavel, set labels to None since v4.51.0 or find a way to extract ids
                    "labels": objects_text_labels,
                }
            )
            results.append(result)

        return results
