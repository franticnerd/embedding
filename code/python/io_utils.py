from zutils import parameter

class IO:

    def __init__(self, para_file='../run/la.yaml'):
        self.init_para(para_file)
        self.init_files()

    def init_para(self, para_file):
        self.para = parameter.yaml_loader().load(para_file)
        # mongo parameters
        self.dns = self.para['mongo']['dns']
        self.port = self.para['mongo']['port']
        self.db = self.para['mongo']['db']
        self.tweet = self.para['mongo']['tweet']
        self.index = self.para['mongo']['index']
        self.exp = self.para['mongo']['exp']
        self.num_bins = self.para['grid']

    def init_files(self):
        self.models_dir = self.para['file']['models']['dir']
        self.output_dir = self.para['file']['output']['dir']
        self.line_dir = self.para['file']['line']['dir']
        # raw files
        self.raw_tweet_file = self.para['file']['raw']['tweets']
        self.clean_text_file = self.para['file']['input']['text']
        self.segmented_text_file = self.para['file']['input']['segmented']
        # entropy file
        self.entropy_file = self.para['file']['output']['entropy']
        self.doc2vec_file = self.para['file']['output']['doc2vec']
        # activity file
        self.activity_file = self.para['file']['output']['activity']
        self.nonactivity_file = self.para['file']['output']['nonactivity']
        # embedding file
        self.embed_file = self.para['file']['output']['embed']
        # evaluation logging file
        self.eval_file = self.para['file']['output']['eval']
