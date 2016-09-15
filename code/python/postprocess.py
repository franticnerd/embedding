import sys
from zutils import parameter
from io_utils import IO
import numpy as np
import heapq
from scipy.spatial.distance import cosine
import operator

class Postprocess:

    def __init__(self, para_file):
        self.para = parameter.yaml_loader().load(para_file)
        self.io = IO(para_file)


    def query_embed_results(self):
        # self.load_tweet_text()
        self.embed_vectors = []
        for embed_file in self.io.embed_files:
            embed_dict = self.load_embed_vectors()
            self.embed_vectors.append(embed_dict)
        self.query()


    def load_tweet_text(self):
        self.tweet_text_dict = {}
        with open(self.io.clean_text_file, 'r') as fin:
            for line in fin:
                items = line.split()
                line_id = int(items[0].lstrip('_*'))
                text = ' '.join(items[1:])
                self.tweet_text_dict[line_id] = text
                # if len(self.tweet_text_dict) > 10:
                #     break

    def load_embed_vectors(self, embed_file):
        embed_dict = {}
        with open(embed_file, 'r') as fin:
            for line in fin:
                items = line.split()
                line_id = items[0]
                vector = np.array([float(e) for e in items[1:]])
                embed_dict[line_id] = vector
        return embed_dict

    def query(self):
        while True:
            input_line = raw_input('\nPlease input the line id: ')
            if input_line == 'E':
                break
            items = input_line.split()
            type_id, query_string = int(items[0]), items[1]
            query_vector = self.embed_vectors[type_id][query_string]
            for embed_dict in self.embed_vectors:
                most_similar = self.find_most_similar(embed_dict, query_vector)
                for sim, entity in most_similar:
                    # text = self.tweet_text_dict[id]
                    print '{:>8} {:<}'.format(str(sim), entity)

    def find_most_similar(self, embed_dict, query_vector):
        heap = []
        for key in embed_dict:
            vector = embed_dict[key]
            sim = 1.0 - cosine(query_vector, vector)
            # print vector, query_vector, sim
            if len(heap) < 20:
                heapq.heappush(heap, [sim, key])
            elif sim > heap[0][0]:
                heapq.heappop(heap)
                heapq.heappush(heap, [sim, key])
        heap.sort(key = operator.itemgetter(0), reverse=True)
        return heap


if __name__ == '__main__':
    p = Postprocess(sys.argv[1])
    p.query_embed_results()
