import os
import random
import numpy as np
import pandas as pd

class VTool():

    @classmethod
    def makeDirs(cls, files=[], folders=[]):
        # os.path.abspath(__file__)
        for file in files:
            folder = os.path.dirname(file)
            if not os.path.exists(folder):
                os.makedirs(folder)

        for folder in folders:           
            if not os.path.exists(folder):
                os.makedirs(folder)

    @classmethod
    def makeCsvRandom(cls, basic_path=None, input_file=None, random_file=None, chunk_size=0):
        if input_file is None or random_file is None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(basic_path, input_file)
        random_path = os.path.join(basic_path, random_file)
        cls.makeDirs(files=[random_path])
        
        # time.strftime("%Y-%m-%d %H:%M:%S")
        if chunk_size <= 0:
            data = pd.read_csv(input_path)
            data = np.array(data).tolist()
            random.shuffle(data)
            if len(data) > 0:
                col = len(data[0])
            cols = {}
            for i in range(col):
                cols[i] = []
            pd.DataFrame(cols).to_csv(random_path, index=False)
            pd.DataFrame(data).to_csv(random_path, index=False, header=False, mode="a")
        else:
            lines = 0
            reader = pd.read_csv(input_path, chunksize=100)
            for sentences in reader:
                lines += len(sentences)
            reader.close()
            line_ids = []
            for i in range(lines):
                line_ids.append(i)
            random.shuffle(line_ids)

            reader = pd.read_csv(input_path, iterator = True)
            temp = reader.get_chunk(5)
            reader.close()
            temp = np.array(temp).tolist()
            if len(temp) > 0:
                col = len(temp[0])
            cols = {}
            for i in range(col):
                cols[i] = []
            pd.DataFrame(cols).to_csv(random_path, index=False)

            num = int(lines / chunk_size)
            rest = lines % chunk_size
            if rest != 0:
                num += 1
            for r in range(num):
                data = {}
                lines_start = r*chunk_size
                lines_end = (r+1)*chunk_size
                if lines_end > lines:
                    lines_end = lines
                for i in range(lines_start, lines_end):
                    data[line_ids[i]] = []

                inum = 0
                rsize = 100
                rnum = lines_end - lines_start
                rend = False
                reader = pd.read_csv(input_path, chunksize=rsize)
                for sentences in reader:
                    sentences = np.array(sentences).tolist()
                    for i in range(len(sentences)):
                        k = inum * rsize + i
                        if k in data:
                            data[k] = sentences[i]
                            rnum -= 1
                        if rnum == 0:
                            rend = True
                            break
                    if rend == True:
                        break
                    inum += 1               
                reader.close()

                temp = []
                for i in range(lines_start, lines_end):
                    temp.append(data[line_ids[i]])
                pd.DataFrame(temp).to_csv(random_path, index=False, header=False, mode="a")
                del data, temp

    @classmethod
    def initCsvTrainAndTest(cls, basic_path=None, input_file=None, batch_size=10, test_part=0.1):
        if input_file is None:
            return None
        if basic_path is None:
            basic_path = os.path.dirname(os.path.abspath(__file__))
        if test_part <= 0 or test_part > 1:
            test_part = 0.1
        input_path = os.path.join(basic_path, input_file)

        lines = 0
        reader = pd.read_csv(input_path, chunksize=100)
        for sentences in reader:
            lines += len(sentences)
        reader.close()

        train = int(lines * (1-test_part))
        train = train - train % batch_size
        test = lines - train
        return train, test

# basic_path = "G:\\ml_workspace\\Mine\\run_data"
# VTool.makeCsvRandom(basic_path=basic_path, input_file="store\\lstm\\stock_origin_data.csv", random_file="as.csv", chunk_size=100)
# train, test = VTool.initCsvTrainAndTest(basic_path=basic_path, input_file="store\\lstm\\stock_origin_data.csv", batch_size=30, test_part=0.1)