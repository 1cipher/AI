from sklearn import metrics

def report(target, pred, target_names):
    cm = metrics.confusion_matrix(target, pred)
    print('Contingency matrix: (rows: examples of a given true class)')
    print(cm)
    print('Classification report')
    print(metrics.classification_report(target, pred,
                                        target_names=target_names))
