import urllib2
from sys import platform
from dateutil import parser
from dataset import read_pois, read_tweets, get_voca
from paras import load_params
from embed import *


class QualitativeEvaluator:
    def __init__(self, predictor, predictor_name, pd):
        self.predictor = predictor
        self.predictor_name = predictor_name
        self.pd = pd
        self.pois = np.random.choice(read_pois(pd['poi_file']), 10000)

    def plot_locations_on_google_map(self, locations, output_path):
        request ='https://maps.googleapis.com/maps/api/staticmap?zoom=10&size=600x600&maptype=roadmap&'
        for lat, lng in locations:
            request += 'markers=color:red%7C' + '%f,%f&' % (lat, lng)
        if platform == 'darwin':
            proxy = urllib2.ProxyHandler({'https': '127.0.0.1:1087'}) # VPN via shadowsocks
        else:
            proxy = urllib2.ProxyHandler({})
        opener = urllib2.build_opener(proxy)
        response = opener.open(request).read()
        with open(output_path, 'wb') as f:
            f.write(response)
            f.close()
        time.sleep(3)

    def scribe(self, directory, ls, ps, ts, ws, show_ls):
        if not os.path.isdir(directory):
            os.makedirs(directory)
        for nbs, file_name in [(ps, 'pois.txt'), (ts, 'times.txt'), (ws, 'words.txt')]:
            output_file = open(directory+file_name,'w')
            for nb in nbs:
                output_file.write(str(nb)+'\n')
        if show_ls:
            self.plot_locations_on_google_map(ls[:10], directory+'locations.png')
        else:
            self.plot_locations_on_google_map(ls[:1], directory+'queried_location.png')

    def getNbs1(self, query):
        if type(query)==str and query not in self.pd['category_list'] and query.lower() not in self.predictor.nt2vecs['w']:
            print query, 'not in voca'
            return
        directory = self.pd['result_dir']+str(query)+'/'+self.predictor_name+'/'
        ls, ps, ts, ws = [self.predictor.get_nbs1(self.pois, query, nt) for nt in ['l', 'p', 't', 'w']]
        self.scribe(directory, ls, ps, ts, ws, type(query)!=list)

    def getNbs2(self, query1, query2, func=lambda a, b:a+b):
        if type(query1)==str and query1 not in self.pd['category_list'] and query1.lower() not in self.predictor.nt2vecs['w']:
            return
        directory = self.pd['result_dir']+str(query1)+'-'+str(query2)+'/'+self.predictor_name+'/'
        ls, ps, ts, ws = [self.predictor.get_nbs2(self.pois, query1, query2, func, nt) for nt in ['l', 'p', 't', 'w']]
        self.scribe(directory, ls, ps, ts, ws, type(query1)!=list)



if __name__ == '__main__':
    para_file = None if len(sys.argv) <= 1 else sys.argv[1]
    pd = load_params(para_file)
    tweets = read_tweets(pd['tweet_file'])
    voca = get_voca(tweets, pd['voca_min'], pd['voca_max'])
    periods = [(datetime.datetime(2014, 8, 25), datetime.datetime(2014, 8, 30))]
    periods += [(datetime.datetime(2014, 9, 29), datetime.datetime(2014, 10, 3))]
    periods += [(datetime.datetime(2014, 11, 25), datetime.datetime(2014, 11, 30))]
    for start_date, end_date in periods:
        tweets_train = [tweet for tweet in tweets if start_date<=parser.parse(tweet.datetime).replace(tzinfo=None)<=end_date]
        print start_date, end_date, 'tweets num:', len(tweets_train)
        predictor = EmbedPredictor(pd)
        predictor.fit(tweets_train, voca)
        evaluator = QualitativeEvaluator(predictor, str(start_date.date())+' - '+str(end_date.date()), pd)
        # for category in pd['category_list']:
        # 	evaluator.getNbs1(category)
        for word in ['food', 'restaurant', 'beach', 'weather', 'clothes', 'nba', 'basketball', 'thanksgiving', 'outdoor', 'staple', 'dodgers', 'stadium']:
            evaluator.getNbs1(word)
        # for location in [[34.043021,-118.2690243], [33.9424, -118.4137], [34.008, -118.4961], [34.0711, -118.4434], [34.1017, -118.3270], [34.073851,-118.242147], [34.1381168,-118.3555723]]:
        # 	evaluator.getNbs1(location)
        for location in [[40.6824952,-73.9772289], [40.7505045,-73.9956327], [40.8075355,-73.9647667], [40.8296426,-73.9283685], [40.8128397,-74.0764031]]:
            evaluator.getNbs1(location)
        # evaluator.getNbs2('outdoor', 'friend')
        # evaluator.getNbs2('outdoor', 'weekend')
        # evaluator.getNbs2('Outdoors & Recreation', 'friend')
        # evaluator.getNbs2('Outdoors & Recreation', 'weekend')

			