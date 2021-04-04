
def calculatePrior(label):
    pos = 0
    neg = 0
    for i in range(len(label)):
        if label[i] == 1:
            pos += 1
        else:
            neg += 1
    Ppos = pos/len(label)
    Pneg = neg/len(label)
    return Ppos,Pneg,pos,neg
