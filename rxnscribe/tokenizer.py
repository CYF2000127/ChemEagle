import json
import copy
import random
import numpy as np


PAD = '<pad>'
SOS = '<sos>'
EOS = '<eos>'
UNK = '<unk>'
MASK = '<mask>'

Rxn = '[Rxn]'  # Reaction
Rct = '[Rct]'  # Reactant
Prd = '[Prd]'  # Product
Cnd = '[Cnd]'  # Condition
Idt = '[Idt]'  # Identifier
Mol = '[Mol]'  # Molecule
Txt = '[Txt]'  # Text
Sup = '[Sup]'  # Supplement
Noise = '[Nos]'


class ReactionTokenizer(object):

    def __init__(self, input_size=100, sep_xy=True, pix2seq=False):
        self.stoi = {}
        self.itos = {}
        self.pix2seq = pix2seq
        self.maxx = input_size  # height
        self.maxy = input_size  # width
        self.sep_xy = sep_xy
        self.special_tokens = [PAD, SOS, EOS, UNK, MASK]
        self.tokens = [Rxn, Rct, Prd, Cnd, Idt, Mol, Txt, Sup, Noise]
        self.fit_tokens(self.tokens)

    def __len__(self):
        if self.pix2seq:
            return 2094
        if self.sep_xy:
            return self.offset + self.maxx + self.maxy
        else:
            return self.offset + max(self.maxx, self.maxy)

    @property
    def max_len(self):
        return 256

    @property
    def PAD_ID(self):
        return self.stoi[PAD]

    @property
    def SOS_ID(self):
        return self.stoi[SOS]

    @property
    def EOS_ID(self):
        return self.stoi[EOS]

    @property
    def UNK_ID(self):
        return self.stoi[UNK]

    @property
    def NOISE_ID(self):
        return self.stoi[Noise]

    @property
    def offset(self):
        return 0 if self.pix2seq else len(self.stoi)

    @property
    def output_constraint(self):
        return True

    def fit_tokens(self, tokens):
        vocab = self.special_tokens + tokens
        if self.pix2seq:
            for i, s in enumerate(vocab):
                self.stoi[s] = 2001 + i
            self.stoi[EOS] = len(self) - 2
            # self.stoi[Noise] = len(self) - 1
        else:
            for i, s in enumerate(vocab):
                self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        self.bbox_category_to_token = {1: Mol, 2: Txt, 3: Idt, 4: Sup}
        self.token_to_bbox_category = {item[1]: item[0] for item in self.bbox_category_to_token.items()}

    def is_x(self, x):
        return 0 <= x - self.offset < self.maxx

    def is_y(self, y):
        if self.sep_xy:
            return self.maxx <= y - self.offset < self.maxx + self.maxy
        return 0 <= y - self.offset < self.maxy

    def x_to_id(self, x):
        if x < -0.001 or x > 1.001:
            print(x)
        else:
            x = min(max(x, 0), 1)
        assert 0 <= x <= 1
        return self.offset + round(x * (self.maxx - 1))

    def y_to_id(self, y):
        if y < -0.001 or y > 1.001:
            print(y)
        else:
            y = min(max(y, 0), 1)
        assert 0 <= y <= 1
        if self.sep_xy:
            return self.offset + self.maxx + round(y * (self.maxy - 1))
        return self.offset + round(y * (self.maxy - 1))

    def id_to_x(self, id, scale=1):
        if not self.is_x(id):
            return -1
        return (id - self.offset) / (self.maxx - 1) / scale

    def id_to_y(self, id, scale=1):
        if not self.is_y(id):
            return -1
        if self.sep_xy:
            return (id - self.offset - self.maxx) / (self.maxy - 1) * scale
        return (id - self.offset) / (self.maxy - 1) / scale

    def update_state(self, state, idx):
        if state is None:
            new_state = (Rxn, 'e')
        else:
            if state[1] == 'x1':
                new_state = (state[0], 'y1')
            elif state[1] == 'y1':
                new_state = (state[0], 'x2')
            elif state[1] == 'x2':
                new_state = (state[0], 'y2')
            elif state[1] == 'y2':
                new_state = (state[0], 'c')
            elif state[1] == 'c':
                if self.is_x(idx):
                    new_state = (state[0], 'x1')
                else:
                    new_state = (state[0], 'e')
            else:
                if state[0] == Rct:
                    if self.is_x(idx):
                        new_state = (Cnd, 'x1')
                    else:
                        new_state = (Cnd, 'e')
                elif state[0] == Cnd:
                    new_state = (Prd, 'x1')
                elif state[0] == Prd:
                    new_state = (Rxn, 'e')
                elif state[0] == Rxn:
                    if self.is_x(idx):
                        new_state = (Rct, 'x1')
                    else:
                        new_state = (EOS, 'e')
                else:
                    new_state = (EOS, 'e')
        return new_state

    def output_mask(self, state):
        # mask: True means forbidden
        mask = np.array([True] * len(self))
        if state[1] in ['y1', 'c']:
            mask[self.offset:self.offset+self.maxx] = False
        if state[1] in ['x1', 'x2']:
            if self.sep_xy:
                mask[self.offset+self.maxx:self.offset+self.maxx+self.maxy] = False
            else:
                mask[self.offset:self.offset+self.maxy] = False
        if state[1] == 'y2':
            for token in [Idt, Mol, Txt, Sup]:
                mask[self.stoi[token]] = False
        if state[1] == 'c':
            mask[self.stoi[state[0]]] = False
        if state[1] == 'e':
            if state[0] in [Rct, Cnd, Rxn]:
                mask[self.offset:self.offset + self.maxx] = False
            if state[0] == Rct:
                mask[self.stoi[Cnd]] = False
            if state[0] == Prd:
                mask[self.stoi[Rxn]] = False
                mask[self.stoi[Noise]] = False
            if state[0] in [Rxn, EOS]:
                mask[self.EOS_ID] = False
        return mask

    def update_states_and_masks(self, states, ids):
        new_states = [self.update_state(state, idx) for state, idx in zip(states, ids)]
        masks = np.array([self.output_mask(state) for state in new_states])
        return new_states, masks

    def bbox_to_sequence(self, bbox, category):
        sequence = []
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            return []
        sequence.append(self.x_to_id(x1))
        sequence.append(self.y_to_id(y1))
        sequence.append(self.x_to_id(x2))
        sequence.append(self.y_to_id(y2))
        if category in self.bbox_category_to_token:
            sequence.append(self.stoi[self.bbox_category_to_token[category]])
        else:
            sequence.append(self.stoi[Noise])
        return sequence

    def sequence_to_bbox(self, sequence, scale=[1, 1]):
        if len(sequence) < 5:
            return None
        x1, y1 = self.id_to_x(sequence[0], scale[0]), self.id_to_y(sequence[1], scale[1])
        x2, y2 = self.id_to_x(sequence[2], scale[0]), self.id_to_y(sequence[3], scale[1])
        if x1 == -1 or y1 == -1 or x2 == -1 or y2 == -1 or x1 >= x2 or y1 >= y2 or sequence[4] not in self.itos:
            return None
        category = self.itos[sequence[4]]
        if category not in [Mol, Txt, Idt, Sup]:
            return None
        return {'category': category, 'bbox': (round(x1,3), round(y1,3), round(x2,3), round(y2,3)), 'category_id': self.token_to_bbox_category[category]}

    def perturb_reaction(self, reaction, boxes):
        reaction = copy.deepcopy(reaction)
        options = []
        options.append(0)  # Option 0: add
        if not(len(reaction['reactants']) == 1 and len(reaction['conditions']) == 0 and len(reaction['products']) == 1):
            options.append(1)  # Option 1: delete
            options.append(2)  # Option 2: move
        choice = random.choice(options)
        if choice == 0:
            key = random.choice(['reactants', 'conditions', 'products'])
            # TODO: insert to a random position
            # We simply add a random box, which may be a duplicate box in this reaction
            reaction[key].append(random.randrange(len(boxes)))
        if choice == 1 or choice == 2:
            options = []
            for key, val in [('reactants', 1), ('conditions', 0), ('products', 1)]:
                if len(reaction[key]) > val:
                    options.append(key)
            key = random.choice(options)
            idx = random.randrange(len(reaction[key]))
            del_box = reaction[key][idx]
            reaction[key] = reaction[key][:idx] + reaction[key][idx+1:]
            if choice == 2:
                options = ['reactants', 'conditions', 'products']
                options.remove(key)
                newkey = random.choice(options)
                reaction[newkey].append(del_box)
        return reaction

    def augment_reaction(self, reactions, data):
        area, boxes, labels = data['area'], data['boxes'], data['labels']
        nonempty_boxes = [i for i in range(len(area)) if area[i] > 0]
        if len(nonempty_boxes) == 0:
            return None
        if len(reactions) == 0 or random.randrange(100) < 20:
            num_reactants = random.randint(1, 3)
            num_conditions = random.randint(0, 3)
            num_products = random.randint(1, 3)
            reaction = {
                'reactants': random.choices(nonempty_boxes, k=num_reactants),
                'conditions': random.choices(nonempty_boxes, k=num_conditions),
                'products': random.choices(nonempty_boxes, k=num_products)
            }
        else:
            assert len(reactions) > 0
            reaction = self.perturb_reaction(random.choice(reactions), boxes)
        return reaction

    def reaction_to_sequence(self, reaction, data, shuffle_bbox=False):
        reaction = copy.deepcopy(reaction)
        area, boxes, labels = data['area'], data['boxes'], data['labels']
        # If reactants or products are empty (because of image cropping), skip the reaction
        if all([area[i] == 0 for i in reaction['reactants']]) or all([area[i] == 0 for i in reaction['products']]):
            return []
        if shuffle_bbox:
            random.shuffle(reaction['reactants'])
            random.shuffle(reaction['conditions'])
            random.shuffle(reaction['products'])
        sequence = []
        for idx in reaction['reactants']:
            if area[idx] == 0:
                continue
            sequence += self.bbox_to_sequence(boxes[idx].tolist(), labels[idx].item())
        sequence.append(self.stoi[Rct])
        for idx in reaction['conditions']:
            if area[idx] == 0:
                continue
            sequence += self.bbox_to_sequence(boxes[idx].tolist(), labels[idx].item())
        sequence.append(self.stoi[Cnd])
        for idx in reaction['products']:
            if area[idx] == 0:
                continue
            sequence += self.bbox_to_sequence(boxes[idx].tolist(), labels[idx].item())
        sequence.append(self.stoi[Prd])
        sequence.append(self.stoi[Rxn])
        return sequence

    def data_to_sequence(self, data, rand_order=False, shuffle_bbox=False, add_noise=False, mix_noise=False):
        sequence = [self.SOS_ID]
        sequence_out = [self.SOS_ID]
        reactions = copy.deepcopy(data['reactions'])
        reactions_seqs = []
        for reaction in reactions:
            seq = self.reaction_to_sequence(reaction, data, shuffle_bbox=shuffle_bbox)
            reactions_seqs.append([seq, seq])
        noise_seqs = []
        if add_noise:
            total_len = sum(len(seq) for seq, seq_out in reactions_seqs)
            while total_len < self.max_len:
                reaction = self.augment_reaction(reactions, data)
                if reaction is None:
                    break
                seq = self.reaction_to_sequence(reaction, data)
                if len(seq) == 0:
                    continue
                if mix_noise:
                    seq[-1] = self.NOISE_ID
                    seq_out = [self.PAD_ID] * (len(seq) - 1) + [self.NOISE_ID]
                else:
                    seq_out = [self.PAD_ID] * (len(seq) - 1) + [self.NOISE_ID]
                noise_seqs.append([seq, seq_out])
                total_len += len(seq)
        if rand_order:
            random.shuffle(reactions_seqs)
        reactions_seqs += noise_seqs
        if mix_noise:
            random.shuffle(reactions_seqs)
        for seq, seq_out in reactions_seqs:
            sequence += seq
            sequence_out += seq_out
        sequence.append(self.EOS_ID)
        sequence_out.append(self.EOS_ID)
        return sequence, sequence_out

    def sequence_to_data(self, sequence, scores=None, scale=None):
        reactions = []
        i = 0
        cur_reaction = {'reactants': [], 'conditions': [], 'products': []}
        flag = 'reactants'
        if len(sequence) > 0 and sequence[0] == self.SOS_ID:
            i += 1
        while i < len(sequence):
            if sequence[i] == self.EOS_ID:
                break
            if sequence[i] in self.itos:
                if self.itos[sequence[i]] in [Rxn, Noise]:
                    cur_reaction['label'] = self.itos[sequence[i]]
                    if len(cur_reaction['reactants']) > 0 and len(cur_reaction['products']) > 0:
                        reactions.append(cur_reaction)
                    cur_reaction = {'reactants': [], 'conditions': [], 'products': []}
                    flag = 'reactants'
                elif self.itos[sequence[i]] == Rct:
                    flag = 'conditions'
                elif self.itos[sequence[i]] == Cnd:
                    flag = 'products'
                elif self.itos[sequence[i]] == Prd:
                    flag = None
            elif i+5 <= len(sequence) and flag is not None:
                bbox = self.sequence_to_bbox(sequence[i:i+5], scale)
                if bbox is not None:
                    cur_reaction[flag].append(bbox)
                    i += 4
            i += 1
        return reactions

    def sequence_to_tokens(self, sequence):
        return [self.itos[x] if x in self.itos else x for x in sequence]


class BboxTokenizer(ReactionTokenizer):

    def __init__(self, input_size=100, sep_xy=True, pix2seq=False):
        super(BboxTokenizer, self).__init__(input_size, sep_xy, pix2seq)

    @property
    def max_len(self):
        return 500

    @property
    def output_constraint(self):
        return False

    def random_category(self):
        return random.choice(list(self.bbox_category_to_token.keys()))
        # return random.choice([random.choice(list(self.bbox_category_to_token.keys())), self.NOISE_ID])

    def random_bbox(self):
        _x1, _y1, _x2, _y2 = random.random(), random.random(), random.random(), random.random()
        x1, y1, x2, y2 = min(_x1, _x2), min(_y1, _y2), max(_x1, _x2), max(_y1, _y2)
        category = self.random_category()
        return [x1, y1, x2, y2], category

    def jitter_bbox(self, bbox, ratio=0.2):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        _x1 = x1 + random.uniform(-w*ratio, w*ratio)
        _y1 = y1 + random.uniform(-h*ratio, h*ratio)
        _x2 = x2 + random.uniform(-w * ratio, w * ratio)
        _y2 = y2 + random.uniform(-h * ratio, h * ratio)
        x1, y1, x2, y2 = min(_x1, _x2), min(_y1, _y2), max(_x1, _x2), max(_y1, _y2)
        category = self.random_category()
        return np.clip([x1, y1, x2, y2], 0, 1), category

    def augment_box(self, bboxes):
        if len(bboxes) == 0:
            return self.random_bbox()
        if random.random() < 0.5:
            return self.random_bbox()
        else:
            return self.jitter_bbox(random.choice(bboxes))

    def split_heuristic_helper(self, toprocess):
        maxy = 0 
        for pair in toprocess:
            if pair[0][1]>maxy:
                maxy = pair[0][1]
        numbuckets = int(maxy//500 + 1)

        buckets = {}
        for i in range(numbuckets):
            buckets[i] = []

        for pair in toprocess:
            buckets[int(pair[0][1]//500)].append(pair)

        for bucket in buckets:
            buckets[bucket] = sorted(buckets[bucket], key = lambda x: x[0][0])
        toreturn = []

        for bucket in buckets:
            toreturn+=buckets[bucket]

        return toreturn

    def data_to_sequence(self, data, add_noise=False, rand_order=False, split_heuristic=False):
        sequence = [self.SOS_ID]
        sequence_out = [self.SOS_ID]
        if rand_order:
            perm = np.random.permutation(len(data['boxes']))
            boxes = data['boxes'][perm].tolist()
            labels = data['labels'][perm].tolist()
        elif split_heuristic:
            to_process = list(zip(data['boxes'].tolist(), data['labels'].tolist()))
            processed = self.split_heuristic_helper(to_process)
            boxes = [item[0] for item in processed]
            labels = [item[1] for item in processed]
        else:
            boxes = data['boxes'].tolist()
            labels = data['labels'].tolist()
        for bbox, category in zip(boxes, labels):
            seq = self.bbox_to_sequence(bbox, category)
            sequence += seq
            # sequence[-1] = self.random_category()
            sequence_out += seq
        if add_noise:
            while len(sequence) < self.max_len:
                bbox, category = self.augment_box(boxes)
                sequence += self.bbox_to_sequence(bbox, category)
                sequence_out += [self.PAD_ID] * 4 + [self.NOISE_ID]
        sequence.append(self.EOS_ID)
        sequence_out.append(self.EOS_ID)
        return sequence, sequence_out

    def sequence_to_data(self, sequence, scores=None, scale=None):
        bboxes = []
        i = 0
        #print(sequence)
        if len(sequence) > 0 and sequence[0] == self.SOS_ID:
            i += 1
        while i < len(sequence):
            if sequence[i] == self.EOS_ID:
                break
            if i+4 < len(sequence):
                bbox = self.sequence_to_bbox(sequence[i:i+5], scale)
                if bbox is not None:
                    if scores is not None:
                        bbox['score'] = scores[i + 4]
                    bboxes.append(bbox)
                    i += 4
            i += 1
        return bboxes

class CorefTokenizer(ReactionTokenizer):

    def __init__(self, input_size=100, sep_xy=True, pix2seq=False):
        super(CorefTokenizer, self).__init__(input_size, sep_xy, pix2seq)

    @property
    def max_len(self):
        return 500

    @property
    def output_constraint(self):
        return False

    def split_heuristic_helper(self, toprocess):
        maxy = 0 
        compress = []
        for pair in toprocess:
            if pair[1] == 1 or pair[1] == 2:
                compress.append([pair])
            else:
                compress[-1].append(pair)

        for pair in toprocess:
            if pair[0][1] > maxy and (pair[1] == 1 or pair[1] ==2):
                maxy = pair[0][1]
        numbuckets = int(maxy//500 + 1)

        buckets = {}
        for i in range(numbuckets):
            buckets[i] = []

        for bbox_group in compress:
            buckets[int(bbox_group[0][0][1]//500)].append(bbox_group)

        for bucket in buckets:
            buckets[bucket] = sorted(buckets[bucket], key = lambda x: x[0][0][0])
        toreturn = []

        for bucket in buckets:
            for bbox_group in buckets[bucket]:
                toreturn+=bbox_group

        return toreturn

    def coref_tokenize(self, boxes, labels, corefs, split_heuristic = False):
        coref_dict = {}
        for pair in corefs:
            if pair[0] in coref_dict:
                coref_dict[pair[0]].append(pair[1])
            else:
                coref_dict[pair[0]] = [pair[1]]
        #coref_dict = {pair[0]: pair[1] for pair in corefs}
        toreturn_boxes = []
        toreturn_labels = []
        
        for i, label in enumerate(labels):
            if i in coref_dict:
                toreturn_boxes.append(boxes[i])
                toreturn_labels.append(labels[i])
                for index in coref_dict[i]:
                    
                    toreturn_boxes.append(boxes[index])
                    toreturn_labels.append(labels[index])
            elif label == 1:
                toreturn_boxes.append(boxes[i])
                toreturn_labels.append(labels[i])  
        '''
        for pair in corefs:
            for entry in pair:
                toreturn_boxes.append(boxes[entry])
                toreturn_labels.append(labels[entry])
        '''
        if split_heuristic:
            returned = self.split_heuristic_helper(list(zip(toreturn_boxes, toreturn_labels)))
            toreturn_boxes = [r[0] for r in returned]
            toreturn_labels = [r[1] for r in returned]
        '''
        if True:
            for i, label in enumerate(labels): 
                if label == 2:
                    toreturn_boxes.append(boxes[i])
                    toreturn_labels.append(labels[i])
        '''
        return toreturn_boxes, toreturn_labels

    def data_to_sequence(self, data, add_noise = False, rand_order = False, split_heuristic = False):
        sequence = [self.SOS_ID]
        sequence_out = [self.SOS_ID]
        if rand_order:
            #TODO
            pass
        else:
            boxes, labels = self.coref_tokenize(data['boxes'].tolist(), data['labels'].tolist(), data['corefs'], split_heuristic)
        for bbox, category in zip(boxes, labels):
   
            seq = self.bbox_to_sequence(bbox, category)
            sequence += seq
            # sequence[-1] = self.random_category()
            sequence_out += seq
        if add_noise:
            pass
            #TODO
            '''
            while len(sequence) < self.max_len:
                bbox, category = self.augment_box(boxes)
                sequence += self.bbox_to_sequence(bbox, category)
                sequence_out += [self.PAD_ID] * 4 + [self.NOISE_ID]
            '''
        
        #sequence = sequence[:6]
        #sequence_out = sequence_out[:6]
        sequence.append(self.EOS_ID)
        sequence_out.append(self.EOS_ID)
        return sequence, sequence_out
    
    def sequence_to_data(self, sequence, scores=None, scale=None):
        bboxes = []
        i = 0
        if len(sequence) > 0 and sequence[0] == self.SOS_ID:
            i += 1
        while i < len(sequence):
            if sequence[i] == self.EOS_ID:
                break
            if i+4 < len(sequence):
                bbox = self.sequence_to_bbox(sequence[i:i+5], scale)
                if bbox is not None:
                    if scores is not None:
                        bbox['score'] = scores[i + 4]
                    bboxes.append(bbox)
                    i += 4
            i += 1
        return {'bboxes': bboxes, 'corefs': self.bbox_to_coref(bboxes)}
    
    def bbox_to_coref(self, bboxes):
        corefs = []

        for i in range(len(bboxes) - 1):
            if bboxes[i]['category_id'] == 1 or bboxes[i]['category_id'] == 2:
                j = i + 1
                while j < len(bboxes) and bboxes[j]['category_id'] == 3:
                    corefs.append([i, j])
                    j += 1

        return corefs 
            
class CocoTokenizer(BboxTokenizer):

    def __init__(self, input_size=100, sep_xy=True, pix2seq=False):
        super(CocoTokenizer, self).__init__(input_size, sep_xy, pix2seq)
        self.index_to_class = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
        self.class_to_index = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}

    @property
    def max_len(self):
        return 700

    def random_category(self):
        return random.choice(list(self.class_to_index.keys()))

    
    def bbox_to_sequence(self, bbox, category):
        sequence = []
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            return []
        sequence.append(self.x_to_id(x1))
        sequence.append(self.y_to_id(y1))
        sequence.append(self.x_to_id(x2))
        sequence.append(self.y_to_id(y2))

        sequence.append(2006+self.class_to_index[category])


        return sequence

    def sequence_to_bbox(self, sequence, scale=[1, 1]):
        if len(sequence) < 5:
            return None
        x1, y1 = self.id_to_x(sequence[0], scale[0]), self.id_to_y(sequence[1], scale[1])
        x2, y2 = self.id_to_x(sequence[2], scale[0]), self.id_to_y(sequence[3], scale[1])
        if x1 == -1 or y1 == -1 or x2 == -1 or y2 == -1 or x1 >= x2 or y1 >= y2:
            return None
        if sequence[4] - 2006 in self.index_to_class:
            category = self.index_to_class[sequence[4] - 2006]
        else:
            category = -1
        return { 'bbox': (x1, y1, x2, y2), 'category_id': category}

        

def get_tokenizer(args):
    tokenizer = {}
    if args.pix2seq:
        args.coord_bins = 2000
        args.sep_xy = False
    format = args.format
    if format == 'reaction':
        tokenizer[format] = ReactionTokenizer(args.coord_bins, args.sep_xy, args.pix2seq)
    if format == 'bbox':
        if args.is_coco:
            tokenizer[format] = CocoTokenizer(args.coord_bins, args.sep_xy, args.pix2seq)
        else:
            tokenizer[format] = BboxTokenizer(args.coord_bins, args.sep_xy, args.pix2seq)
    if format == 'coref':
        tokenizer[format] = CorefTokenizer(args.coord_bins, args.sep_xy, args.pix2seq)
    return tokenizer