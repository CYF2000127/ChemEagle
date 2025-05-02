import pdf2image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import layoutparser as lp
import cv2

from PyPDF2 import PdfReader, PdfWriter
import pandas as pd

import pdfminer.high_level
import pdfminer.layout
from operator import itemgetter

# inputs: pdf_file, page #, bounding box (optional) (llur or ullr), output_bbox
class TableExtractor(object):
    def __init__(self, output_bbox=True):
        self.pdf_file = ""
        self.page = ""
        self.image_dpi = 200
        self.pdf_dpi = 72
        self.output_bbox = output_bbox
        self.blocks = {}
        self.title_y = 0
        self.column_header_y = 0
        self.model = None
        self.img = None
        self.output_image = True
        self.tagging = {
            'substance': ['compound', 'salt', 'base', 'solvent', 'CBr4', 'collidine', 'InX3', 'substrate', 'ligand', 'PPh3', 'PdL2', 'Cu', 'compd', 'reagent', 'reagant', 'acid', 'aldehyde', 'amine', 'Ln', 'H2O', 'enzyme', 'cofactor', 'oxidant', 'Pt(COD)Cl2', 'CuBr2', 'additive'],
            'ratio': [':'],
            'measurement': ['μM', 'nM', 'IC50', 'CI', 'excitation', 'emission', 'Φ', 'φ', 'shift', 'ee', 'ΔG', 'ΔH', 'TΔS', 'Δ', 'distance', 'trajectory', 'V', 'eV'],
            'temperature': ['temp', 'temperature', 'T', '°C'],
            'time': ['time', 't(', 't ('],
            'result': ['yield', 'aa', 'result', 'product', 'conversion', '(%)'],
            'alkyl group': ['R', 'Ar', 'X', 'Y'],
            'solvent': ['solvent'],
            'counter': ['entry', 'no.'],
            'catalyst': ['catalyst', 'cat.'],
            'conditions': ['condition'],
            'reactant': ['reactant'],
        }
        
    def set_output_image(self, oi):
        self.output_image = oi
    
    def set_pdf_file(self, pdf):
        self.pdf_file = pdf
    
    def set_page_num(self, pn):
        self.page = pn
        
    def set_output_bbox(self, ob):
        self.output_bbox = ob
        
    def run_model(self, page_info):
        #img = np.asarray(pdf2image.convert_from_path(self.pdf_file, dpi=self.image_dpi)[self.page])

        #model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config', extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5], label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
        
        img = np.asarray(page_info)
        self.img = img
        
        layout_result = self.model.detect(img)
        
        text_blocks = lp.Layout([b for b in layout_result if b.type == 'Text'])
        title_blocks = lp.Layout([b for b in layout_result if b.type == 'Title'])
        list_blocks = lp.Layout([b for b in layout_result if b.type == 'List'])
        table_blocks = lp.Layout([b for b in layout_result if b.type == 'Table'])
        figure_blocks = lp.Layout([b for b in layout_result if b.type == 'Figure'])
        
        self.blocks.update({'text': text_blocks})
        self.blocks.update({'title': title_blocks})
        self.blocks.update({'list': list_blocks})
        self.blocks.update({'table': table_blocks})
        self.blocks.update({'figure': figure_blocks})
    
    # type is what coordinates you want to get. it comes in text, title, list, table, and figure
    def convert_to_pdf_coordinates(self, type):
        # scale coordinates
        
        blocks = self.blocks[type]
        coordinates =  [blocks[a].scale(self.pdf_dpi/self.image_dpi) for a in range(len(blocks))]
        
        reader = PdfReader(self.pdf_file)

        writer = PdfWriter()
        p = reader.pages[self.page]
        a = p.mediabox.upper_left
        new_coords = []
        for new_block in coordinates:
            new_coords.append((new_block.block.x_1, pd.to_numeric(a[1]) - new_block.block.y_2, new_block.block.x_2, pd.to_numeric(a[1]) - new_block.block.y_1))
        
        return new_coords
    # output: list of bounding boxes for tables but in pdf coordinates
    
    # input: new_coords is singular table bounding box in pdf coordinates
    def extract_singular_table(self, new_coords):
        for page_layout in pdfminer.high_level.extract_pages(self.pdf_file, page_numbers=[self.page]):
            elements = []
            for element in page_layout:
                if isinstance(element, pdfminer.layout.LTTextBox):
                    for e in element._objs:
                        temp = e.bbox
                        if temp[0] > min(new_coords[0], new_coords[2]) and temp[0] < max(new_coords[0], new_coords[2]) and temp[1] > min(new_coords[1], new_coords[3]) and temp[1] < max(new_coords[1], new_coords[3]) and temp[2] > min(new_coords[0], new_coords[2]) and temp[2] < max(new_coords[0], new_coords[2]) and temp[3] > min(new_coords[1], new_coords[3]) and temp[3] < max(new_coords[1], new_coords[3]) and isinstance(e, pdfminer.layout.LTTextLineHorizontal):
                            elements.append([e.bbox[0], e.bbox[1], e.bbox[2], e.bbox[3], e.get_text()])
                            
            elements = sorted(elements, key=itemgetter(0))
            w = sorted(elements, key=itemgetter(3), reverse=True)
            if len(w) <= 1:
                continue

            ret = {}
            i = 1
            g = [w[0]]

            while i < len(w) and w[i][3] > w[i-1][1]:
                g.append(w[i])
                i += 1
            g = sorted(g, key=itemgetter(0))
            # check for overlaps
            for a in range(len(g)-1, 0, -1):
                if g[a][0] < g[a-1][2]:
                    g[a-1][0] = min(g[a][0], g[a-1][0])
                    g[a-1][1] = min(g[a][1], g[a-1][1])
                    g[a-1][2] = max(g[a][2], g[a-1][2])
                    g[a-1][3] = max(g[a][3], g[a-1][3])
                    g[a-1][4] = g[a-1][4].strip() + " " + g[a][4]
                    g.pop(a)
            
            
            ret.update({"columns":[]})
            for t in g:
                temp_bbox = t[:4]
                
                column_text = t[4].strip()
                tag = 'unknown'
                tagged = False
                for key in self.tagging.keys():
                    for word in self.tagging[key]:
                        if word in column_text:
                            tag = key
                            tagged = True
                            break
                    if tagged:
                        break
                
                if self.output_bbox:
                    ret["columns"].append({'text':column_text,'tag': tag, 'bbox':temp_bbox})
                else:
                    ret["columns"].append({'text':column_text,'tag': tag})
                self.column_header_y = max(t[1], t[3])
            ret.update({"rows":[]})

            g.insert(0, [0, 0, new_coords[0], 0, ''])
            g.append([new_coords[2], 0, 0, 0, ''])
            while i < len(w):
                group = [w[i]]
                i += 1
                while i < len(w) and w[i][3] > w[i-1][1]:
                    group.append(w[i])
                    i += 1
                group = sorted(group, key=itemgetter(0))

                for a in range(len(group)-1, 0, -1):
                    if group[a][0] < group[a-1][2]:
                        group[a-1][0] = min(group[a][0], group[a-1][0])
                        group[a-1][1] = min(group[a][1], group[a-1][1])
                        group[a-1][2] = max(group[a][2], group[a-1][2])
                        group[a-1][3] = max(group[a][3], group[a-1][3])
                        group[a-1][4] = group[a-1][4].strip() + " " + group[a][4]
                        group.pop(a)
                
                a = 1
                while a < len(g) - 1:
                    if a > len(group):
                        group.append([0, 0, 0, 0, '\n'])
                        a += 1
                        continue
                    if group[a-1][0] >= g[a-1][2] and group[a-1][2] <= g[a+1][0]:
                        pass
                        """
                        if a < len(group) and group[a][0] >= g[a-1][2] and group[a][2] <= g[a+1][0]:
                            g.insert(1, [g[0][2], 0, group[a-1][2], 0, ''])
                            #ret["columns"].insert(0, '')
                        else:
                            a += 1
                            continue
                        """
                    else: group.insert(a-1, [0, 0, 0, 0, '\n'])
                    a += 1
                
                
                added_row = []
                for t in group:
                    temp_bbox = t[:4]
                    if self.output_bbox:
                        added_row.append({'text':t[4].strip(), 'bbox':temp_bbox})
                    else:
                        added_row.append(t[4].strip())
                ret["rows"].append(added_row)
            if ret["rows"] and len(ret["rows"][0]) != len(ret["columns"]):
                ret["columns"] = ret["rows"][0]
                ret["rows"] = ret["rows"][1:]
                for col in ret['columns']:
                    tag = 'unknown'
                    tagged = False
                    for key in self.tagging.keys():
                        for word in self.tagging[key]:
                            if word in col['text']:
                                tag = key
                                tagged = True
                                break
                        if tagged:
                            break
                    col['tag'] = tag
            
            return ret
            
    def get_title_and_footnotes(self, tb_coords):
    
        for page_layout in pdfminer.high_level.extract_pages(self.pdf_file, page_numbers=[self.page]):
            title = (0, 0, 0, 0, '')
            footnote = (0, 0, 0, 0, '')
            title_gap = 30
            footnote_gap = 30
            for element in page_layout:
                if isinstance(element, pdfminer.layout.LTTextBoxHorizontal):
                    if (element.bbox[0] >= tb_coords[0] and element.bbox[0] <= tb_coords[2]) or (element.bbox[2] >= tb_coords[0] and element.bbox[2] <= tb_coords[2]) or (tb_coords[0] >= element.bbox[0] and tb_coords[0] <= element.bbox[2]) or (tb_coords[2] >= element.bbox[0] and tb_coords[2] <= element.bbox[2]):
                        #print(element)
                        if 'Table' in element.get_text():
                            if abs(element.bbox[1] - tb_coords[3]) < title_gap:
                                title = tuple(element.bbox) + (element.get_text()[element.get_text().index('Table'):].replace('\n', ' '),)
                                title_gap = abs(element.bbox[1] - tb_coords[3])
                        if 'Scheme' in element.get_text():
                            if abs(element.bbox[1] - tb_coords[3]) < title_gap:
                                title = tuple(element.bbox) + (element.get_text()[element.get_text().index('Scheme'):].replace('\n', ' '),)
                                title_gap = abs(element.bbox[1] - tb_coords[3])
                        if element.bbox[1] >= tb_coords[1] and element.bbox[3] <= tb_coords[3]: continue
                        #print(element)
                        temp = ['aA', 'aB', 'aC', 'aD', 'aE', 'aF', 'aG', 'aH', 'aI', 'aJ', 'aK', 'aL', 'aM', 'aN', 'aO', 'aP', 'aQ', 'aR', 'aS', 'aT', 'aU', 'aV', 'aW', 'aX', 'aY', 'aZ', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a0']
                        for segment in temp:
                            if segment in element.get_text():
                                if abs(element.bbox[3] - tb_coords[1]) < footnote_gap:
                                    footnote = tuple(element.bbox) + (element.get_text()[element.get_text().index(segment):].replace('\n', ' '),)
                                    footnote_gap = abs(element.bbox[3] - tb_coords[1])
                                break
            self.title_y = min(title[1], title[3])
            if self.output_bbox:
                return ({'text': title[4], 'bbox': list(title[:4])}, {'text': footnote[4], 'bbox': list(footnote[:4])})
            else:
                return (title[4], footnote[4])
            
    def extract_table_information(self):
        #self.run_model(page_info) # changed
        table_coordinates = self.blocks['table'] #should return a list of layout objects
        table_coordinates_in_pdf = self.convert_to_pdf_coordinates('table') #should return a list of lists

        ans = []
        i = 0
        for coordinate in table_coordinates_in_pdf:
            ret = {}
            pad = 20
            coordinate = [coordinate[0] - pad, coordinate[1], coordinate[2] + pad, coordinate[3]]
            ullr_coord = [coordinate[0], coordinate[3], coordinate[2], coordinate[1]]
        
            table_results = self.extract_singular_table(coordinate)
            tf = self.get_title_and_footnotes(coordinate)
            figure = Image.fromarray(table_coordinates[i].crop_image(self.img))
            ret.update({'title': tf[0]})
            ret.update({'figure': {
                'image': None,
                'bbox': []
                       }})
            if self.output_image:
                ret['figure']['image'] = figure
            ret.update({'table': {'bbox': list(coordinate), 'content': table_results}})
            ret.update({'footnote': tf[1]})
            if abs(self.title_y - self.column_header_y) > 50:
                ret['figure']['bbox'] = list(coordinate)
            
            ret.update({'page':self.page})
            
            ans.append(ret)
            i += 1
        
        return ans
        
    def extract_figure_information(self):
        figure_coordinates = self.blocks['figure']
        figure_coordinates_in_pdf = self.convert_to_pdf_coordinates('figure')
        
        ans = []
        for i in range(len(figure_coordinates)):
            ret = {}
            coordinate = figure_coordinates_in_pdf[i]
            ullr_coord = [coordinate[0], coordinate[3], coordinate[2], coordinate[1]]
            
            tf = self.get_title_and_footnotes(coordinate)
            figure = Image.fromarray(figure_coordinates[i].crop_image(self.img))
            ret.update({'title':tf[0]})
            ret.update({'figure': {
                'image': None,
                'bbox': []
                       }})
            if self.output_image:
                ret['figure']['image'] = figure
            ret.update({'table': {
                'bbox': [],
                'content': None
                       }})
            ret.update({'footnote': tf[1]})
            ret['figure']['bbox'] = list(coordinate)
                
            ret.update({'page':self.page})
            
            ans.append(ret)
        
        return ans
            
        
    def extract_all_tables_and_figures(self, pages, pdfparser, content=None):
        self.model = pdfparser
        ret = []
        for i in range(len(pages)):
            self.set_page_num(i)
            self.run_model(pages[i])
            table_info = self.extract_table_information()
            figure_info = self.extract_figure_information()
            if content == 'tables':
                ret += table_info
            elif content == 'figures':
                ret += figure_info
                for table in table_info:
                    if table['figure']['bbox'] != []:
                        ret.append(table)
            else:
                ret += table_info
                ret += figure_info
        return ret
