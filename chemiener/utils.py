
def merge_predictions(results):
    if len(results) == 0:
        return []
    predictions = {}
    for batch_preds in results:
        for idx, preds in enumerate(batch_preds):
            predictions[idx] = preds
    predictions = [predictions[i] for i in range(len(predictions))]

    return predictions

def get_class_to_index(corpus):
    if corpus == "chemu":
        return {'B-EXAMPLE_LABEL': 1, 'B-REACTION_PRODUCT': 2, 'B-STARTING_MATERIAL': 3, 'B-REAGENT_CATALYST': 4, 'B-SOLVENT': 5, 'B-OTHER_COMPOUND': 6, 'B-TIME': 7, 'B-TEMPERATURE': 8, 'B-YIELD_OTHER': 9, 'B-YIELD_PERCENT': 10, 'O': 0,
            'I-EXAMPLE_LABEL': 11, 'I-REACTION_PRODUCT': 12, 'I-STARTING_MATERIAL': 13, 'I-REAGENT_CATALYST': 14, 'I-SOLVENT': 15, 'I-OTHER_COMPOUND': 16, 'I-TIME': 17, 'I-TEMPERATURE': 18, 'I-YIELD_OTHER': 19, 'I-YIELD_PERCENT': 20}
    elif corpus == "chemdner":
        return {'O': 0, 'B-ABBREVIATION': 1, 'B-FAMILY': 2,  'B-FORMULA': 3, 'B-IDENTIFIER': 4, 'B-MULTIPLE': 5, 'B-SYSTEMATIC': 6, 'B-TRIVIAL': 7, 'B-NO CLASS': 8, 'I-ABBREVIATION': 9, 'I-FAMILY': 10,  'I-FORMULA': 11, 'I-IDENTIFIER': 12, 'I-MULTIPLE': 13, 'I-SYSTEMATIC': 14, 'I-TRIVIAL': 15, 'I-NO CLASS': 16}
    elif corpus == "chemdner-mol":
        return {'O': 0, 'B-MOL': 1, 'I-MOL': 2}



