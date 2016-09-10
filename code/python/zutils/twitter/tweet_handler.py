import json
from zutils import text_processor, timestamp

class Tweet:

    # parse a tweet from an input line (GIS data format)
    def load(self, line):
        items = line.split('\x01')
        if len(items) != 28 or items[9] != 'en':
            raise IOError
        self.id = long(items[0])
        self.uid = long(items[10])
        self.created_at = items[6]
        location_items = items[2].split(',')
        self.lat = float(location_items[0])
        self.lng = float(location_items[1])
        self.text = items[1]
        self.clean_text()
        self.calc_timestamp()

    # load a clean tweet from a mongo database object
    def load_from_mongo(self, d):
        self.id = d['id']
        self.uid = d['uid']
        self.created_at = d['time']
        self.ts = d['timestamp']
        self.lat = d['lat']
        self.lng = d['lng']
        self.text = d['text']
        # self.words = d['words']
        self.words = d['phrases']

    # parse the raw tweet message to return a list of words
    def clean_text(self):
        message = self.text.split('http')[0]
        tp = text_processor.TextProcessor()
        self.words = tp.parse_words(message)

    # calc timestamp in second, the start time is 2000-01-01 00:00:00
    def calc_timestamp(self):
        t = timestamp.Timestamp()
        self.ts = t.get_timestamp(self.created_at)


    # parse a clean tweet from a line
    def parse_clean_tweet(self, line):
        items = line.split(',')
        self.id = long(items[0])
        self.user_id = long(items[1])
        self.created_at = items[2]
        self.lat = float(items[3])
        self.lng = float(items[4])
        self.words = items[5].split()


    def get_id(self):
        return self.id

    def get_uid(self):
        return self.uid

    # return the number of words for a clean tweet
    def num_words(self):
        return len(self.words)

    def get_clean_words_string(self):
        return ' '.join(self.words)

    def to_dict(self):
        return {'id': self.id,
                'uid': self.uid,
                'time': self.created_at,
                'timestamp': self.ts,
                'lat': self.lat,
                'lng': self.lng,
                'text': self.text,
                'words': self.words}

    def to_json_string(self):
        dict_data = self.to_dict()
        return json.dumps(dict_data, sort_keys=True)

    def to_string(self):
        data = [str(self.id), str(self.uid), str(self.lat), str(self.lng),
                self.created_at, ' '.join(self.words)]
        return ','.join(data)
