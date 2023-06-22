from SmilesPE.pretokenizer import atomwise_tokenizer

class tokenize(object):

    def smileTokenizer(self, smi):
        return atomwise_tokenizer(smi)

    def proteinTokenizer(self,
                         protSequence):
        encoding_data = [ protSequence[i:i + 3] for i in range(len(protSequence) - 2) ]
        return encoding_data




