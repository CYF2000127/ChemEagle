import os
import contextlib
import copy
import numpy as np

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from .data import ImageData, ReactionImageData, CorefImageData


class CocoEvaluator(object):

    def __init__(self, coco_gt):
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

    def evaluate(self, predictions):
        img_ids, results = self.prepare(predictions, 'bbox')
        if len(results) == 0:
            return np.zeros((12,))
        coco_dt = self.coco_gt.loadRes(results)
        cocoEval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        cocoEval.params.imgIds = img_ids
        cocoEval.params.catIds = [1]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        self.cocoEval = cocoEval
        return cocoEval.stats

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        img_ids = []
        coco_results = []
        for idx, prediction in enumerate(predictions):
            if len(prediction) == 0:
                continue

            image = self.coco_gt.dataset['images'][idx]
            img_ids.append(image['id'])
            width = image['width']
            height = image['height']
            coco_results.extend(
                [
                    {
                        "image_id": image['id'],
                        "category_id": pred['category_id'],
                        "bbox": convert_to_xywh(pred['bbox'], width, height),
                        "score": pred['score'],
                    }
                    for pred in prediction
                ]
            )
        return img_ids, coco_results


def convert_to_xywh(box, width, height):
    xmin, ymin, xmax, ymax = box
    return [xmin * width, ymin * height, (xmax - xmin) * width, (ymax - ymin) * height]


EMPTY_STATS = {'gold_hits': 0, 'gold_total': 0, 'pred_hits': 0, 'pred_total': 0, 'image': 0}


class ReactionEvaluator(object):

    def evaluate_image(self, gold_image, pred_image, **kwargs):
        data = ReactionImageData(gold_image, pred_image)


        return data.evaluate(**kwargs)

    def compute_metrics(self, gold_hits, gold_total, pred_hits, pred_total):
        precision = pred_hits / max(pred_total, 1)
        recall = gold_hits / max(gold_total, 1)
        f1 = precision * recall * 2 / max(precision + recall, 1e-6)
        return {'precision': precision, 'recall': recall, 'f1': f1}

    def evaluate(self, groundtruths, predictions, **kwargs):
        gold_hits, gold_total, pred_hits, pred_total = 0, 0, 0, 0
        for gold_image, pred_image in zip(groundtruths, predictions):
            gh, ph = self.evaluate_image(gold_image, pred_image, **kwargs)
            gold_hits += sum(gh)
            gold_total += len(gh)
            pred_hits += sum(ph)
            pred_total += len(ph)
        return self.compute_metrics(gold_hits, gold_total, pred_hits, pred_total)

    def evaluate_by_size(self, groundtruths, predictions, **kwargs):
        group_stats = {}
        for gold_image, pred_image in zip(groundtruths, predictions):
            gh, ph = self.evaluate_image(gold_image, pred_image, **kwargs)
            gtotal = len(gh)
            if gtotal not in group_stats:
                group_stats[gtotal] = copy.deepcopy(EMPTY_STATS)
            group_stats[gtotal]['gold_hits'] += sum(gh)
            group_stats[gtotal]['gold_total'] += len(gh)
            group_stats[gtotal]['pred_hits'] += sum(ph)
            group_stats[gtotal]['pred_total'] += len(ph)
            group_stats[gtotal]['image'] += 1
        group_scores = {}
        for gtotal, stats in group_stats.items():
            group_scores[gtotal] = self.compute_metrics(
                stats['gold_hits'], stats['gold_total'], stats['pred_hits'], stats['pred_total'])
        return group_scores, group_stats

    def evaluate_by_group(self, groundtruths, predictions, **kwargs):
        group_stats = {}
        for gold_image, pred_image in zip(groundtruths, predictions):
            gh, ph = self.evaluate_image(gold_image, pred_image, **kwargs)
            diagram_type = gold_image['diagram_type']
            if diagram_type not in group_stats:
                group_stats[diagram_type] = copy.deepcopy(EMPTY_STATS)
            group_stats[diagram_type]['gold_hits'] += sum(gh)
            group_stats[diagram_type]['gold_total'] += len(gh)
            group_stats[diagram_type]['pred_hits'] += sum(ph)
            group_stats[diagram_type]['pred_total'] += len(ph)
            group_stats[diagram_type]['image'] += 1
        group_scores = {}
        for group, stats in group_stats.items():
            group_scores[group] = self.compute_metrics(
                stats['gold_hits'], stats['gold_total'], stats['pred_hits'], stats['pred_total'])
        return group_scores, group_stats

    def evaluate_summarize(self, groundtruths, predictions, **kwargs):
        size_scores, size_stats = self.evaluate_by_size(groundtruths, predictions, **kwargs)
        summarize = {
            'overall': copy.deepcopy(EMPTY_STATS),
            # 'single': copy.deepcopy(EMPTY_STATS),
            # 'multiple': copy.deepcopy(EMPTY_STATS)
        }
        for size, stats in size_stats.items():
            if type(size) is int:
                # output = summarize['single'] if size <= 1 else summarize['multiple']
                for key in stats:
                    # output[key] += stats[key]
                    summarize['overall'][key] += stats[key]
        scores = {}
        for key, val in summarize.items():
            scores[key] = self.compute_metrics(val['gold_hits'], val['gold_total'], val['pred_hits'], val['pred_total'])
        return scores, summarize, size_stats

class CorefEvaluator(object):

    def evaluate_image(self, gold_image, pred_image, **kwargs):
        data = CorefImageData(gold_image, predictions = pred_image)
        return data.evaluate()
    
    def evaluate(self, groundtruths, predictions):
        hits, gold_total, pred_total = 0, 0, 0
        counter = 0 
        print(len(predictions))
        for gold_image, pred_image in zip(groundtruths, predictions):
            
            try: hit, gold_pairs, pred_pairs = self.evaluate_image(gold_image, pred_image)
            except: print(counter)
            hits += hit
            gold_total += gold_pairs
            pred_total += pred_pairs
            counter += 1
        return hits, gold_total, pred_total
    
    def evaluate_summarize(self, groundtruths, predictions):
        hits, gold_total, pred_total = self.evaluate(groundtruths, predictions)
        precision = hits/max(pred_total, 1)
        recall = hits/max(gold_total, 1)
        f1 = precision * recall * 2 / max(precision + recall, 1e-6)
        return (precision, recall, f1)
        