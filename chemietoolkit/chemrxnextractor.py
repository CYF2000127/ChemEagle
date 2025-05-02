from PyPDF2 import PdfReader, PdfWriter
import pdfminer.high_level
import pdfminer.layout
from operator import itemgetter
import os
import pdftotext
from chemrxnextractor import RxnExtractor

class ChemRxnExtractor(object):
    def __init__(self, pdf, pn, model_dir, device):
        self.pdf_file = pdf
        self.pages = pn
        self.model_dir = os.path.join(model_dir, "cre_models_v0.1") # directory saving both prod and role models
        use_cuda = (device == 'cuda')
        self.rxn_extractor = RxnExtractor(self.model_dir, use_cuda=use_cuda)
        self.text_file = "info.txt"
        self.pdf_text = ""
        if len(self.pdf_file) > 0:
            with open(self.pdf_file, "rb") as f:
                self.pdf_text = pdftotext.PDF(f)
        
    def set_pdf_file(self, pdf):
        self.pdf_file = pdf
        with open(self.pdf_file, "rb") as f:
            self.pdf_text = pdftotext.PDF(f)
    
    def set_pages(self, pn):
        self.pages = pn
    
    def set_model_dir(self, md):
        self.model_dir = md
        self.rxn_extractor = RxnExtractor(self.model_dir)
    
    def set_text_file(self, tf):
        self.text_file = tf
    
    def extract_reactions_from_text(self):
        if self.pages is None:
            return self.extract_all(len(self.pdf_text))
        else:
            return self.extract_all(self.pages)
    
    def extract_all(self, pages):
        ans = []
        text = self.get_paragraphs_from_pdf(pages)
        for data in text:
            L = [sent for paragraph in data['paragraphs'] for sent in paragraph]
            reactions = self.get_reactions(L, page_number=data['page'])
            ans.append(reactions)
        return ans
    
    def get_reactions(self, sents, page_number=None):
        rxns = self.rxn_extractor.get_reactions(sents)
        
        ret = []
        for r in rxns:
            if len(r['reactions']) != 0: ret.append(r)
        ans = {}
        ans.update({'page' : page_number})
        ans.update({'reactions' : ret})
        return ans


    def get_paragraphs_from_pdf(self, pages):
        current_page_num = 1
        if pages is None:
            pages = len(self.pdf_text)
        result = []
        for page in range(pages):
            content = self.pdf_text[page]
            pg = content.split("\n\n")
            L = []
            for line in pg:
                paragraph = []
                if '\x0c' in line:
                    continue
                text = line
                text = text.replace("\n", " ")
                text = text.replace("- ", "-")
                curind = 0
                i = 0
                while i < len(text):
                    if text[i] == '.':
                        if i != 0 and not text[i-1].isdigit() or i != len(text) - 1 and (text[i+1] == " " or text[i+1] == "\n"):
                            paragraph.append(text[curind:i+1] + "\n")
                            while(i < len(text) and text[i] != " "):
                                i += 1
                            curind = i + 1
                    i += 1
                if curind != i:
                    if text[i - 1] == " ":
                        if i != 1:
                            i -= 1
                        else:
                            break
                    if text[i - 1] != '.':
                        paragraph.append(text[curind:i] + ".\n")
                    else:
                        paragraph.append(text[curind:i] + "\n")
                L.append(paragraph)

            result.append({
                'paragraphs': L,
                'page': current_page_num
            })
            current_page_num += 1
        return result
