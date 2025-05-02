import os
import cv2
import numpy as np
import matplotlib.colors as colors
import matplotlib.patches as patches
from PIL import Image


class BBox(object):

    def __init__(self, bbox, image_data=None, xyxy=False, normalized=False):
        """
        :param bbox: {'catrgory_id', 'bbox'}
        :param input_image: ImageData object
        :param xyxy:
        :param normalized:
        """
        self.data = bbox
        self.image_data = image_data
        if image_data is not None:
            self.width = image_data.width
            self.height = image_data.height
        self.category_id = bbox['category_id']
        if xyxy:
            x1, y1, x2, y2 = bbox['bbox']
        else:
            x1, y1, w, h = bbox['bbox']
            x2, y2 = x1 + w, y1 + h
        if not normalized:
            x1, y1, x2, y2 = x1 / self.width, y1 / self.height, x2 / self.width, y2 / self.height
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    @property
    def is_mol(self):
        return self.category_id == 1

    @property
    def is_idt(self):
        return self.category_id == 3

    @property
    def is_empty(self):
        return abs(self.x2 - self.x1) <= 0.01 or abs(self.y2 - self.y1) <= 0.01

    def unnormalize(self):
        return self.x1 * self.width, self.y1 * self.height, self.x2 * self.width, self.y2 * self.height

    def image(self):
        x1, y1, x2, y2 = self.unnormalize()
        x1, y1, x2, y2 = max(int(x1), 0), max(int(y1), 0), min(int(x2), self.width), min(int(y2), self.height)
        return self.image_data.image[y1:y2, x1:x2]

    COLOR = {1: 'r', 2: 'g', 3: 'b', 4: 'y'}
    CATEGORY = {1: 'Mol', 2: 'Txt', 3: 'Idt', 4: 'Sup'}

    def draw(self, ax, color='r', text = None):
        x1, y1, x2, y2 = self.unnormalize()
        if color is None:
            color = self.COLOR[self.category_id]
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor=color, facecolor=colors.to_rgba(color, 0.2))
        text = f'{self.CATEGORY[self.category_id]}'
        if text == 'Mol':
            ax.text(x1 , y1-15, text, fontsize=10, bbox=dict(linewidth=0, facecolor='yellow', alpha=0.5))
        else:
            ax.text(x1-45 , y1+10, text, fontsize=10, bbox=dict(linewidth=0, facecolor='yellow', alpha=0.5))
        ax.add_patch(rect)
        return

    def set_smiles(self, smiles, symbols,coords,edges,  molfile=None, atoms=None, bonds=None):
        self.data['smiles'] = smiles
        self.data['symbols'] = symbols
        self.data['coords'] = coords
        self.data['edges'] = edges
        if molfile:
            self.data['molfile'] = molfile
        
        if atoms:
            self.data['atoms'] = atoms
        if bonds:
            self.data['bonds'] = bonds

    def set_text(self, text):
        self.data['text'] = text

    def to_json(self):
        return self.data


class Reaction(object):

    def __init__(self, reaction=None, bboxes=None, image_data=None):
        '''
        if image_data is None, create from prediction
        if image_data is not None, create from groundtruth
        '''
        self.reactants = []
        self.conditions = []
        self.products = []
        self.bboxes = []
        if reaction is not None:
            for x in reaction['reactants']:
                bbox = bboxes[x] if type(x) is int else BBox(x, image_data, xyxy=True, normalized=True)
                self.bboxes.append(bbox)
                self.reactants.append(len(self.bboxes) - 1)
            for x in reaction['conditions']:
                bbox = bboxes[x] if type(x) is int else BBox(x, image_data, xyxy=True, normalized=True)
                self.bboxes.append(bbox)
                self.conditions.append(len(self.bboxes) - 1)
            for x in reaction['products']:
                bbox = bboxes[x] if type(x) is int else BBox(x, image_data, xyxy=True, normalized=True)
                self.bboxes.append(bbox)
                self.products.append(len(self.bboxes) - 1)

    def to_json(self):
        return {
            'reactants': [self.bboxes[i].to_json() for i in self.reactants],
            'conditions': [self.bboxes[i].to_json() for i in self.conditions],
            'products': [self.bboxes[i].to_json() for i in self.products]
        }

    def _deduplicate_bboxes(self, indices):
        results = []
        for i, idx_i in enumerate(indices):
            duplicate = False
            for j, idx_j in enumerate(indices[:i]):
                if get_iou(self.bboxes[idx_i], self.bboxes[idx_j]) > 0.6:
                    duplicate = True
                    break
            if not duplicate:
                results.append(idx_i)
        return results

    def deduplicate(self):
        flags = [False] * len(self.bboxes)
        bbox_list = self.reactants + self.products + self.conditions
        for i, idx_i in enumerate(bbox_list):
            if self.bboxes[idx_i].is_empty:
                flags[idx_i] = True
                continue
            for idx_j in bbox_list[:i]:
                if flags[idx_j] is False and get_iou(self.bboxes[idx_i], self.bboxes[idx_j]) > 0.6:
                    flags[idx_i] = True
                    break
        self.reactants = [i for i in self.reactants if not flags[i]]
        self.conditions = [i for i in self.conditions if not flags[i]]
        self.products = [i for i in self.products if not flags[i]]

    def schema(self, mol_only=False):
        # Return reactants, conditions, and products. If mol_only is True, only include bboxes that are mol structures.
        if mol_only:
            reactants, conditions, products = [[idx for idx in indices if self.bboxes[idx].is_mol]
                                               for indices in [self.reactants, self.conditions, self.products]]
            # It would be unfair to compare two reactions if their reactants or products are empty after filtering.
            # Setting them to the original ones in this case.
            if len(reactants) == 0:
                reactants = self.reactants
            if len(products) == 0:
                products = self.products
            return reactants, conditions, products
        else:
            return self.reactants, self.conditions, self.products

    def compare(self, other, mol_only=False, merge_condition=False, debug=False):
        reactants1, conditions1, products1 = self.schema(mol_only)
        reactants2, conditions2, products2 = other.schema(mol_only)
        if debug:
            print(reactants1, conditions1, products1, ';', reactants2, conditions2, products2)
        if len(reactants1) + len(conditions1) + len(products1) == 0:
            # schema is empty, always return False
            return False
        if len(reactants1) + len(conditions1) + len(products1) != len(reactants2) + len(conditions2) + len(products2):
            return False
        # Match use original index
        match1, match2, scores = get_bboxes_match(self.bboxes, other.bboxes, iou_thres=0.5)
        m_reactants, m_conditions, m_products = [[match1[i] for i in x] for x in [reactants1, conditions1, products1]]
        if any([m == -1 for m in m_reactants + m_conditions + m_products]):
            return False
        if debug:
            print(m_reactants, m_conditions, m_products, ';', reactants2, conditions2, products2)
        if merge_condition:
            return sorted(m_reactants + m_conditions) == sorted(reactants2 + conditions2) \
                   and sorted(m_products) == sorted(products2)
        else:
            return sorted(m_reactants) == sorted(reactants2) and sorted(m_conditions) == sorted(conditions2) \
                   and sorted(m_products) == sorted(products2)

    def __eq__(self, other):
        # Exact matching of two reactions
        return self.compare(other)

    def draw(self, ax):
        for i in self.reactants:
            self.bboxes[i].draw(ax, color='r')
        for i in self.conditions:
            self.bboxes[i].draw(ax, color='g')
        for i in self.products:
            self.bboxes[i].draw(ax, color='b')
        return


class ReactionSet(object):

    def __init__(self, reactions, bboxes=None, image_data=None):
        self.reactions = [Reaction(reaction, bboxes, image_data) for reaction in reactions]

    def __len__(self):
        return len(self.reactions)

    def __iter__(self):
        return iter(self.reactions)

    def __getitem__(self, item):
        return self.reactions[item]

    def deduplicate(self):
        results = []
        for reaction in self.reactions:
            if any(r == reaction for r in results):
                continue
            if len(reaction.reactants) < 1 or len(reaction.products) < 1:
                continue
            results.append(reaction)
        self.reactions = results

    def to_json(self):
        return [r.to_json() for r in self.reactions]


class ImageData(object):

    def __init__(self, data=None, predictions=None, image_file=None, image=None):
        self.width, self.height = None, None
        if data:
            self.file_name = data['file_name']
            self.width = data['width']
            self.height = data['height']
        if image_file:
            self.image = cv2.imread(image_file)
            self.height, self.width, _ = self.image.shape
        if image is not None:
            if not isinstance(image, np.ndarray):
                image = np.asarray(image)
            self.image = image
            self.height, self.width, _ = self.image.shape
        if data and 'bboxes' in data:
            self.gold_bboxes = [BBox(bbox, self, xyxy=False, normalized=False) for bbox in data['bboxes']]
        if predictions is not None:
            self.pred_bboxes = [BBox(bbox, self, xyxy=True, normalized=True) for bbox in predictions]

    def draw_gold(self, ax, image=None):
        if image is not None:
            ax.imshow(image)
        for i, b in enumerate(self.gold_bboxes):
            b.draw(ax, color = None)

    def draw_prediction(self, ax, image=None):
        if image is not None:
            ax.imshow(image)
        for i, b in enumerate(self.pred_bboxes):
            b.draw(ax, color = None)


class ReactionImageData(ImageData):

    def __init__(self, data=None, predictions=None, image_file=None, image=None):
        super().__init__(data=data, image_file=image_file, image=image)
        if data and 'reactions' in data:
            self.gold_reactions = ReactionSet(data['reactions'], self.gold_bboxes, image_data=self)
        if predictions is not None:
            self.pred_reactions = ReactionSet(predictions, image_data=self)
            self.pred_reactions.deduplicate()

    def evaluate(self, mol_only=False, merge_condition=False, debug=False):
        gold_total = len(self.gold_reactions)
        gold_hit = [False] * gold_total
        pred_total = len(self.pred_reactions)
        pred_hit = [False] * pred_total
        for i, ri in enumerate(self.gold_reactions):
            for j, rj in enumerate(self.pred_reactions):
                if gold_hit[i] and pred_hit[j]:
                    continue
                if ri.compare(rj, mol_only, merge_condition, debug):
                    gold_hit[i] = True
                    pred_hit[j] = True
        return gold_hit, pred_hit

class CorefImageData(ImageData):

    def __init__(self, data=None, predictions=None, image_file=None, image=None):
        super().__init__(data=data, predictions = predictions, image_file=image_file, image=image)
        if data and 'corefs' in data:
            self.gold_corefs = data['corefs']
    
    def evaluate(self):
        #for every bbox in self.gold_bboxes, match with highest iou in self.pred_bboxes
        #a true hit is defined as follows: suppose a pair (i, j) is a coref. then if highest_iou(j) follows
        #highest_iou(i) in pred_bboxes, it is a hit. 
        #total number of predictions is number of bboxes in pred/2.
        #precision = TP/number of predictions
        #recall = TP/number of gt pairs

        if hasattr(self, "pred_bboxes"):
            hits = 0
            num_preds = 0
            for pred in self.pred_bboxes:
                if pred.category_id == 3:
                    num_preds+=1
            matches = {}
            for gold in self.gold_bboxes:
                highest_iou = 0
                highest_index = -1
                for i, pred in enumerate(self.pred_bboxes):
                    iou = get_iou(gold, pred)
                    if iou> highest_iou:
                        highest_iou = iou
                        highest_index = i
                if highest_iou > 0.3 and gold.category_id == 1:
                    matches[gold] = highest_index
                else:
                    matches[gold]=highest_index
            for coref_pair in self.gold_corefs:
                mol = self.gold_bboxes[coref_pair[0]]
                idx = self.gold_bboxes[coref_pair[1]]

                if mol in matches and idx in matches: 
                    all_ids = True
                    if matches[mol] < matches[idx]:
                        for counter in range(matches[mol]+1, matches[idx], 1):
                            if self.pred_bboxes[counter].category_id != 3:
                                all_ids = False
                        if all_ids:
                            hits+=1
            return hits, len(self.gold_corefs), num_preds
        
        return 0, 0, 0

    def draw_gold(self, ax, image=None):
        if image is not None:
            ax.imshow(image)
        counter_dict = {}
        counter = 0

        for pair in self.gold_corefs:
            mol, idt = pair 
            if mol in counter_dict:
                xmin, ymin, xmax, ymax = self.gold_bboxes[idt].unnormalize()
                ax.text(xmin - 50, ymin+ 60, str(counter_dict[mol]), fontsize=20, bbox=dict(facecolor='purple', alpha=0.5))
                
            else:
                counter+=1
                counter_dict[mol] = counter
                xmin, ymin, xmax, ymax = self.gold_bboxes[mol].unnormalize() 
                ax.text(xmin - 50, ymin+ 60, str(counter), fontsize=20, bbox=dict(facecolor='purple', alpha=0.5))
                xmin, ymin, xmax, ymax = self.gold_bboxes[idt].unnormalize() 
                ax.text(xmin - 50, ymin+ 60, str(counter), fontsize=20, bbox=dict(facecolor='purple', alpha=0.5)) 
        for b in self.gold_bboxes:
            b.draw(ax)

    def draw_prediction(self, ax, image=None):
        if image is not None:
            ax.imshow(image)
        counter = 0
        colours = ['#648fff', '#785ef0','#dc267f', '#fe6100','#ffb000','r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y']
        colorcounter = -1
        for i, b in enumerate(self.pred_bboxes):
            if (b.category_id == 1 or b.category_id == 2):
                counter += 1
                colorcounter += 1
                b.draw(ax, color = colours[colorcounter%len(colours)]) 
            elif b.category_id == 3:
                b.draw(ax, color = colours[colorcounter%len(colours)]) 



def deduplicate_bboxes(bboxes):
    results = []
    for i in range(len(bboxes)):
        duplicate = False
        for j in range(i):
            if get_iou(bboxes[i], bboxes[j]) > 0.9:
                duplicate = True
                break
        if not duplicate:
            results.append(bboxes[i])
    return results

def get_iou(bb1, bb2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    bb1 = {'x1': bb1.x1, 'y1': bb1.y1, 'x2': bb1.x2, 'y2': bb1.y2}
    bb2 = {'x1': bb2.x1, 'y1': bb2.y1, 'x2': bb2.x2, 'y2': bb2.y2}

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_bboxes_match(bboxes1, bboxes2, iou_thres=0.5, match_category=False):
    """Find the match between two sets of bboxes. Each bbox is matched with a bbox with maximum overlap
    (at least above iou_thres). -1 if a bbox does not have a match."""
    scores = np.zeros((len(bboxes1), len(bboxes2)))
    for i, bbox1 in enumerate(bboxes1):
        for j, bbox2 in enumerate(bboxes2):
            if match_category and bbox1.category_id != bbox2.category_id:
                scores[i, j] = 0
            else:
                scores[i, j] = get_iou(bbox1, bbox2)
    match1 = scores.argmax(axis=1)
    for i in range(len(match1)):
        if scores[i, match1[i]] < iou_thres:
            match1[i] = -1
    match2 = scores.argmax(axis=0)
    for j in range(len(match2)):
        if scores[match2[j], j] < iou_thres:
            match2[j] = -1
    return match1, match2, scores


def deduplicate_reactions(reactions):
    pred_reactions = ReactionSet(reactions)
    for r in pred_reactions:
        r.deduplicate()
    pred_reactions.deduplicate()
    return pred_reactions.to_json()


def postprocess_reactions(reactions, image_file=None, image=None, molscribe=None, ocr=None, batch_size=32):
    image_data = ReactionImageData(predictions=reactions, image_file=image_file, image=image)
    pred_reactions = image_data.pred_reactions
    for r in pred_reactions:
        r.deduplicate()
    pred_reactions.deduplicate()
    if molscribe:
        bbox_images, bbox_indices = [], []
        for i, reaction in enumerate(pred_reactions):
            for j, bbox in enumerate(reaction.bboxes):
                if bbox.is_mol:
                    bbox_images.append(bbox.image())
                    bbox_indices.append((i, j))
        if len(bbox_images) > 0:
            predictions = molscribe.predict_images(bbox_images, return_atoms_bonds=True, batch_size=batch_size)

            for (i, j), pred in zip(bbox_indices, predictions):
                pred_reactions[i].bboxes[j].set_smiles(pred['smiles'],pred["symbols"], pred["coords"],pred["edges"],pred['molfile'],pred['atoms'], pred['bonds'])
                #deduplicated[i].set_smiles(pred['smiles'],pred['oringinal_coords'],pred['original_symbols'],pred['orignal_edges'])
    if ocr:
        for reaction in pred_reactions:
            for bbox in reaction.bboxes:
                if not bbox.is_mol:
                    text = ocr.readtext(bbox.image(), detail=0)
                    bbox.set_text(text)
    return pred_reactions.to_json()

def postprocess_bboxes(bboxes, image = None, molscribe = None, batch_size = 32):
    image_d = ImageData(image = image)
    bbox_objects = [BBox(bbox = bbox, image_data = image_d, xyxy = True, normalized = True) for bbox in bboxes]
    bbox_objects_no_empty = [bbox for bbox in bbox_objects if not bbox.is_empty]
    #deduplicate
    deduplicated = deduplicate_bboxes(bbox_objects_no_empty)

    if molscribe:
        bbox_images, bbox_indices = [], []

        for i, bbox in enumerate(deduplicated):
            if bbox.is_mol:
                bbox_images.append(bbox.image())
                bbox_indices.append(i)
        
        if len(bbox_images) > 0:
            predictions = molscribe.predict_images(bbox_images, return_atoms_bonds=True, batch_size = batch_size)

            for i, pred in zip(bbox_indices, predictions):
                #deduplicated[i].set_smiles(pred['smiles'], pred["original_symbols"],pred['molfile'],pred['atoms'], pred['bonds'])
                deduplicated[i].set_smiles(pred['smiles'],pred["symbols"], pred["coords"],pred["edges"],pred['molfile'],pred['atoms'], pred['bonds'])
    return [bbox.to_json() for bbox in deduplicated]

def postprocess_coref_results(bboxes, image, molscribe = None, ocr = None, batch_size = 32):
    image_d = ImageData(image = cv2.resize(np.asarray(image), None, fx=3, fy=3))
    bbox_objects = [BBox(bbox = bbox, image_data = image_d, xyxy = True, normalized = True) for bbox in bboxes['bboxes']]
    if molscribe:
        
        bbox_images, bbox_indices = [], []

        for i, bbox in enumerate(bbox_objects):
            if bbox.is_mol:
                bbox_images.append(bbox.image())
                bbox_indices.append(i)
        
        if len(bbox_images) > 0:
            predictions = molscribe.predict_images(bbox_images, return_atoms_bonds=True, batch_size = batch_size)

            for i, pred in zip(bbox_indices, predictions):
                bbox_objects[i].set_smiles(pred['smiles'],pred["symbols"], pred["coords"],pred["edges"],pred['molfile'],pred['atoms'], pred['bonds'])
    if ocr: 
        for bbox in bbox_objects:
            if bbox.is_idt:
                text = ocr.readtext(bbox.image(), detail = 0)
                bbox.set_text(text)
    
    return {'bboxes': [bbox.to_json() for bbox in bbox_objects], 'corefs': bboxes['corefs']}
        




    
