import os
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import argparse

DATA_DIR = "E:\GIT_ROOT\Pytorch-Deep-Learning-models\LatentVarseq2seq\data\Cornell"





parser = argparse.ArgumentParser()
parser.add_argument("DATA_DIR", help="Choose folder where files are stored",type=str)
parser.add_argument("mode", help="Choose train or val",type=str)


args = parser.parse_args()

class CornellParse:

    def getlines(self,filepath):
        """
        Input parameters:
           filename/file location - Must be present in the same location, else change the code to find the file
        """
        with open(filepath, mode='r') as doc:
            content = doc.read()
            sentences = content.split("\n")
            doc.close()
        return sentences

    def join(self,qsent,asent):
        """
        Input parameters:
           List of sentences from question and answer files
        """

        if len(qsent)==len(asent):
            writesent=[a_+" "+ b_ for a_, b_ in zip(qsent, asent)]
            return writesent
        else:
            raise "File lengths must match"



    def parsecorpus(self):
        """
        Input parameters:

        """
        DIR = args.DATA_DIR
        print(DIR)
        if args.mode=="train":
            outfile = args.DATA_DIR+"\\"+args.mode+".txt"
            qfile= args.DATA_DIR+"\\"+"train.enc"
            afile = args.DATA_DIR +"\\"+  "train.dec"
        elif args.mode=="val":
            outfile = args.DATA_DIR+"\\"+args.mode+".txt"
            qfile = args.DATA_DIR +"\\"+ "test.enc"
            afile = args.DATA_DIR + "\\" + "test.dec"
        print("Starting....")
        print("Combining questions and responses into one corpus file for % s " % (args.mode))
        qsent=self.getlines(qfile)
        print("Done with Questions")
        asent=self.getlines(afile)
        print("Done with Answers")
        writesent=self.join(qsent,asent)
        print("Done combining")
        with open(outfile, mode="w", encoding="utf-8") as target:
            print("Writing to %s"%(outfile))
            for s in writesent:
                target.write(s)
                target.write("\n")
            target.close()
        print("Finished. Use the script to process now")






if __name__ == '__main__':
    cp = CornellParse()
    cp.parsecorpus()
