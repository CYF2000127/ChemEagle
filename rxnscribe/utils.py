import json


def merge_predictions(results):
    if len(results) == 0:
        return {}
    formats = results[0][1].keys()
    predictions = {format_: {} for format_ in formats}
    for format_ in formats:
        for indices, batch_preds in results:
            for idx, preds in zip(indices, batch_preds[format_]):
                predictions[format_][idx] = preds
        predictions[format_] = [predictions[format_][i] for i in range(len(predictions[format_]))]
    return predictions
