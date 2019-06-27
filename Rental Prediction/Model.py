import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing
from lightgbm import LGBMClassifier as LGB
import xgboost as xgb
import itertools
import re
import nltk
import math
import pickle

NY_LAT = 40.785091
NY_LON = -73.968285
Stopwords = nltk.corpus.stopwords.words('english')
Stemmer = nltk.PorterStemmer()

# Haversine equation for calculating distance
def get_distance_from_centre(lat_srs, lon_srs):
    p = np.pi / 180
    dist_srs = 0.5 - (np.cos((lat_srs - NY_LAT) * p) / 2) + ((np.cos(lat_srs * p) * math.cos(NY_LAT * p) * (1 - np.cos((lon_srs - NY_LON) * p))) / 2)
    return 12742 * np.arcsin(np.sqrt(dist_srs))

# Functions that should be called on either or train or test data exclusively are suffixed appropriately
class FeatureSelector:
    def __init__(self, data):
        self.data = data
        self.attributes = []
        self.features = []

    def select_numeric(self):
        numeric = [
            'bathrooms',
            'bedrooms',
            'latitude',
            'longitude',
            'price'
        ]

        self.features.extend(numeric)
        self.attributes.extend(numeric)

    def select_num_features(self):
        self.features.append("num_features")

        self.data["num_features"] = self.data.features.apply(len)
        self.attributes.append("num_features")

    def select_num_photos(self):
        self.features.append("num_photos")

        self.data["num_photos"] = self.data.photos.apply(len)
        self.attributes.append("num_photos")

    def select_price_per_bedrooms(self):
        self.features.append("price_per_bedrooms")

        self.data["price_per_bedrooms"] = self.data.price / (self.data.bedrooms + 1)
        self.attributes.append("price_per_bedrooms")

    def select_price_per_bathrooms(self):
        self.features.append("price_per_bathrooms")

        self.data["price_per_bathrooms"] = self.data.price / (self.data.bathrooms + 1)
        self.attributes.append("price_per_bathrooms")

    def select_total_rooms(self):
        self.features.append("total_rooms")

        self.data["total_rooms"] = self.data.bathrooms + self.data.bedrooms
        self.attributes.append("total_rooms")

    def select_price_per_total_rooms(self):
        self.features.append("price_per_total_rooms")

        self.data["price_per_total_rooms"] = self.data.price / (self.data.bathrooms + self.data.bedrooms + 1)

        self.attributes.append("price_per_total_rooms")

    def select_num_description(self):
        self.features.append("num_description")

        self.data["num_description"] = self.data.description.apply(lambda x: len(x.split(" ")))
        self.attributes.append("num_description")

    def select_created(self, day=True, month=True, year=True):
        temp = pd.to_datetime(self.data.created)
        if day:
            self.data["created_day"] = temp.dt.day
            self.attributes.append("created_day")
            self.features.append("created_day")
        if month:
            self.data["created_month"] = temp.dt.month
            self.attributes.append("created_month")
            self.features.append("created_month")
        if year:
            self.data["created_year"] = temp.dt.year
            self.attributes.append("created_year")
            self.features.append("created_year")

    def select_distance(self):
        self.data["distance"] = get_distance_from_centre(self.data.latitude, self.data.longitude)
        self.attributes.append("distance")
        self.features.append("distance_haversine")

    def select_man_ability_train(self, intgp=None):
        self.features.append("man_ability_inverse_allocation")

        # Generating counts of each interest_level grouped by manager
        man_df = pd.crosstab(self.data.manager_id, self.data.interest_level)

        if intgp == None:
            # Generating the weights as inversely proportional to number of instances
            intgp = self.data.groupby('interest_level').size()
            intgp = intgp.apply(lambda x: self.data.shape[0] / x)

        # Create man_ability as a weighted sum
        man_df['man_ability'] = sum([intgp[i] * man_df[i] for i in ['high', 'low', 'medium']])
        self.man_df = man_df # We store man_df so that it can be used for predictions afterwards

        # Merge the previous data set to add man_ability
        self.data["man_ability"] = self.data.apply(lambda x: man_df.man_ability[x.manager_id], axis=1)
        self.attributes.append("man_ability")

    def select_man_ability_test(self, man_df):
        mn = man_df.man_ability.mean()
        self.data["man_ability"] = self.data.apply(lambda x: man_df.man_ability.get(x.manager_id, mn), axis=1)
        self.attributes.append("man_ability")

    def clean_feature(self, x):
            prx = x.lower()
            prx = re.sub('[^A-Za-z0-9 ]', ' ', prx)
            prx = " ".join(filter(lambda y: y not in Stopwords, prx.split()))
            temp = map(Stemmer.stem, prx.split())
            temp = filter(lambda y: len(y) > 2, temp)
            return " ".join(map(lambda y: y.strip(), temp))

    def select_feat_bigrams_train(self):
        processed_features = self.data.apply(lambda y: map(self.clean_feature, y.features), axis=1)

        self.features.append("""feature_bigram_(analyzer="word", ngram_range=(1,2), max_features=200, stopwords="english")_cvect""")
        self.cvect_feat = CountVectorizer(analyzer="word", ngram_range=(1,2), max_features=150, stop_words="english")
        self.tfidf_feat = TfidfTransformer()

        processed_features = processed_features.map(" ".join)
        full_sparse = self.cvect_feat.fit_transform(processed_features.values)
        # full_sparse = self.tfidf_feat.fit_transform(full_sparse)
        cc = ["FE_"+i for i in self.cvect_feat.get_feature_names()]

        temp_df = pd.DataFrame(full_sparse.toarray(), columns=cc)
        temp_df.index = self.data.index
        self.data = self.data.join(temp_df)

        self.attributes.extend(cc)

    def select_feature_simple(self):
        self.features.append("feature_simple_(stop_words='english'_max_features=150)_cvect")

        self.cvect_feat_simple = CountVectorizer(stop_words='english', max_features=150)
        self.data['feature_simple'] = self.data["features"].apply(
            lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x])
        )

        full_sparse = self.cvect_feat_simple.fit_transform(self.data.feature_simple)
        cc = ["FE_"+i for i in self.cvect_feat_simple.get_feature_names()]

        temp = pd.DataFrame(full_sparse.toarray(), columns=cc)
        temp.index = self.data.index
        self.data = self.data.join(temp)

        self.attributes.extend(cc)

    def select_feature_simple_test(self, cvect_feat):
        processed_features = self.data.features.apply(
            lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x])
        )
        full_sparse = cvect_feat.transform(processed_features.values)
        cc = ["FE_"+i for i in cvect_feat.get_feature_names()]

        temp_df = pd.DataFrame(full_sparse.toarray(), columns=cc)
        temp_df.index = self.data.index
        self.data = self.data.join(temp_df)

        self.attributes.extend(cc)

    def select_feat_bigrams_test(self, cvect_feat, tfidf_feat):
        processed_features = self.data.apply(lambda y: map(self.clean_feature, y.features), axis=1)
        processed_features = processed_features.map(" ".join)
        full_sparse = cvect_feat.transform(processed_features.values)
        # full_sparse = tfidf_feat.transform(full_sparse)
        cc =  ["FE_"+i for i in cvect_feat.get_feature_names()]
        temp_df = pd.DataFrame(full_sparse.toarray(), columns=cc)
        temp_df.index = self.data.index
        self.data = self.data.join(temp_df)

        self.attributes.extend(cc)

    def select_simple_cat_label_encode(self, test, address=True):
        cat = [
            "manager_id",
            "building_id"
        ]

        if address:
            cat.extend(["display_address", "street_address"])

        self.simple_le_cat = {}

        for f in cat:
            le = preprocessing.LabelEncoder()
            self.simple_le_cat[f] = le
            le.fit(list(self.data[f].values) + list(test[f].values))
            self.data[f + "_simple_le"] = le.transform(self.data[f])
            self.attributes.append(f + "_simple_le")
            self.features.append(f + "_simple_le")

    def select_simple_cat_label_encode_test(self, simple_le_cat):
        for f in simple_le_cat.keys():
            le = simple_le_cat[f]
            self.data[f + "_simple_le"] = le.transform(self.data[f])
            self.attributes.append(f + "_simple_le")
            self.features.append(f + "_simple_le")

    def clean_address(self, t):
        transform = {
            "street": "st",
            "st.": "st",
            "west": "w",
            "east": "e",
            "south": "s",
            "north": "n"
        }
        x = t.lower()
        x = re.sub('[^A-Za-z0-9 ]', ' ', x)
        z = x.split()
        z = map(lambda y: y.strip(), z)
        z = map(lambda y: transform.get(y, y), z)
        z = " ".join(z)
        return z

    def select_display_address(self, test):
        self.features.append("clean_display_address")

        temp1 = self.data.display_address.apply(self.clean_address)
        temp2 = test.display_address.apply(self.clean_address)

        le = preprocessing.LabelEncoder()
        self.le_disp_add = le
        le.fit(list(temp1.values) + list(temp2.values))
        self.data["clean_display_address"] = le.transform(temp1)
        self.attributes.append("clean_display_address")

    def select_display_address_test(self, le):
        self.features.append("clean_display_address")

        self.data["clean_display_address"] = le.transform(self.data.display_address.apply(self.clean_address))
        self.attributes.append("clean_display_address")

    def select_street_address(self, test):
        self.features.append("clean_street_address")

        temp1 = self.data.street_address.apply(self.clean_address)
        temp2 = test.street_address.apply(self.clean_address)
        temp1 = temp1.apply(lambda y: y.lstrip('0123456789 '))
        temp2 = temp2.apply(lambda y: y.lstrip('0123456789 '))

        le = preprocessing.LabelEncoder()
        self.le_street_add = le
        le.fit(list(temp1.values) + list(temp2.values))
        self.data["clean_street_address"] = le.transform(temp1)
        self.attributes.append("clean_street_address")

    def select_street_address_test(self, le):
        self.features.append("clean_street_address")

        temp2 = self.data.street_address.apply(self.clean_address)
        temp2 = temp2.apply(lambda y: y.lstrip('0123456789 '))

        self.data["clean_street_address"] = le.transform(temp2)
        self.attributes.append("clean_street_address")

    def select_description_stats(self, num_caps=False, ratio_up_low=False, num_sp=False):
        def up_low_ratio(x):
            up = 0
            low = 0
            for i in x:
                if i.isupper():
                    up += 1
                elif i.islower():
                    low += 1
            return (up+1)/(low+1)

        if num_caps:
            self.features.append("num_caps_desc")
            self.data["num_caps_desc"] = self.data.description.apply(lambda x: sum(1 for c in x.split() if c.isupper()))
            self.attributes.append("num_caps_desc")

        if ratio_up_low:
            self.features.append("ratio_up_low_desc")
            self.data["ratio_up_low_desc"] = self.data.description.apply(up_low_ratio)
            self.attributes.append("ratio_up_low_desc")

        if num_sp:
            self.features.append("num_sp_desc")
            self.data["num_sp_desc"] = self.data.description.apply(lambda x: len(re.findall('[!#$^&*<>=]', x)))
            self.attributes.append("num_sp_desc")

    def clean_description(self, x):
        y = x.lower()
        y = re.sub('<.*?>', ' ', y)
        y = re.sub('[^A-Za-z0-9 ]', ' ', y)
        z = y.split()
        z = map(Stemmer.stem, z)
        z = filter(lambda t: len(t) > 2, z)
        z = map(lambda t: t.strip(), z)
        return " ".join(z)

    def select_description(self):
        processed_features = self.data.description.apply(self.clean_description)

        self.features.append("""description_(analyzer="word", max_features=200, stopwords="english")_cvect""")
        self.cvect_desc = CountVectorizer(analyzer="word", max_features=100, stop_words="english")
        self.tfidf_desc = TfidfTransformer()

        full_sparse = self.cvect_desc.fit_transform(processed_features.values)
        # full_sparse = self.tfidf_desc.fit_transform(full_sparse)
        cc =  ["DE_"+i for i in self.cvect_desc.get_feature_names()]
        temp_df = pd.DataFrame(full_sparse.toarray(), columns=cc)
        temp_df.index = self.data.index
        self.data = self.data.join(temp_df)

        self.attributes.extend(cc)

    def select_description_test(self, cvect, tfidf):
        processed_features = self.data.description.apply(self.clean_description)
        full_sparse = cvect.transform(processed_features.values)
        # full_sparse = tfidf.transform(full_sparse)
        cc =  ["DE_"+i for i in cvect.get_feature_names()]
        temp_df = pd.DataFrame(full_sparse.toarray(), columns=cc)
        temp_df.index = self.data.index
        self.data = self.data.join(temp_df)

        self.attributes.extend(cc)

    def select_manager_id_level(self):
        self.features.extend(["manager_level_low", "manager_level_medium", "manager_level_high"])

        train = self.data

        train['manager_level_low'] = np.nan
        train['manager_level_medium'] = np.nan
        train['manager_level_high'] = np.nan

        for i in range(5):
            train_index, test_index = train_test_split(list(range(train.shape[0])), test_size=(1.0/5), random_state=(i+1)*100, shuffle=True)

            cv_train = train.iloc[train_index]
            cv_test = train.iloc[test_index]

            for m in cv_train.groupby('manager_id'):
                test_subset = cv_test[cv_test.manager_id == m[0]].index

                train.loc[test_subset, 'manager_level_low'] = (m[1].interest_level == 'low').mean()
                train.loc[test_subset, 'manager_level_medium'] = (m[1].interest_level == 'medium').mean()
                train.loc[test_subset, 'manager_level_high'] = (m[1].interest_level == 'high').mean()

        self.attributes.extend(["manager_level_low", "manager_level_medium", "manager_level_high"])

    def select_manager_id_level_test(self, train):
        self.features.extend(["manager_level_low", "manager_level_medium", "manager_level_high"])

        test = self.data

        test['manager_level_low'] = np.nan
        test['manager_level_medium'] = np.nan
        test['manager_level_high'] = np.nan

        for m in train.groupby('manager_id'):
            test_subset = test[test.manager_id == m[0]].index

            test.loc[test_subset, 'manager_level_low'] = (m[1].interest_level == 'low').mean()
            test.loc[test_subset, 'manager_level_medium'] = (m[1].interest_level == 'medium').mean()
            test.loc[test_subset, 'manager_level_high'] = (m[1].interest_level == 'high').mean()

        self.attributes.extend(["manager_level_low", "manager_level_medium", "manager_level_high"])

    def select_manager_id_building_id_level(self):
        self.features.extend(["manager_building_level"])

        train = self.data

        train['manager_building_level'] = np.nan

        for i in range(5):
            train_index, test_index = train_test_split(list(range(train.shape[0])), test_size=(1.0/5), random_state=(i+1)*100, shuffle=True)

            cv_train = train.iloc[train_index]
            cv_test = train.iloc[test_index]

            for m in cv_train.groupby(['manager_id', 'building_id']):
                test_subset = cv_test[(cv_test.manager_id == m[0][0]) & (cv_test.building_id == m[0][1])].index
                train.loc[test_subset, 'manager_building_level'] = 2 * (m[1].interest_level == 'high').mean() + (m[1].interest_level == 'medium').mean()

        self.attributes.extend(["manager_building_level"])

    def select_manager_id_building_id_level_test(self, train):
        self.features.extend(["manager_building_level"])

        test = self.data

        test['manager_building_level'] = np.nan

        for m in train.groupby(['manager_id', 'building_id']):
            test_subset = test[(test.manager_id == m[0][0]) & (test.building_id == m[0][1])].index

            test.loc[test_subset, 'manager_building_level'] = 2 * (m[1].interest_level == 'high').mean() + (m[1].interest_level == 'medium').mean()

        self.attributes.extend(["manager_building_level"])

    def select_price_building_id_level(self):
        self.features.extend(["price_building_level"])

        train = self.data

        train['price_building_level'] = np.nan
        train['price_bin'] = pd.qcut(train['price'], 50) ;

        for i in range(5):
            train_index, test_index = train_test_split(list(range(train.shape[0])), test_size=(1.0/5), random_state=(i+1)*100, shuffle=True)

            cv_train = train.iloc[train_index]
            cv_test = train.iloc[test_index]

            for m in cv_train.groupby(['price_bin', 'building_id']):
                test_subset = cv_test[(cv_test.price_bin == m[0][0]) & (cv_test.building_id == m[0][1])].index
                train.loc[test_subset, 'manager_building_level'] = 2 * (m[1].interest_level == 'high').mean() + (m[1].interest_level == 'medium').mean()

        self.attributes.extend(["price_building_level"])


    def select_price_building_id_level_test(self, train):
        self.features.extend(["price_building_level"])

        test = self.data

        test['price_building_level'] = np.nan
        train['price_bin'] = pd.qcut(train['price'], 50) ;
        test['price_bin'] = pd.qcut(test['price'], 50) ;

        for m in train.groupby(['price_bin', 'building_id']):
            test_subset = test[(test.price_bin == m[0][0]) & (test.building_id == m[0][1])].index

            test.loc[test_subset, 'price_building_level'] = 2 * (m[1].interest_level == 'high').mean() + (m[1].interest_level == 'medium').mean()

        self.attributes.extend(["price_building_level"])

    def select_building_id_level(self):
        self.features.extend(["building_level_low", "building_level_medium", "building_level_high"])

        train = self.data

        train['building_level_low'] = np.nan
        train['building_level_medium'] = np.nan
        train['building_level_high'] = np.nan

        for i in range(5):
            train_index, test_index = train_test_split(list(range(train.shape[0])), test_size=(1.0/5), random_state=(i+1)*100, shuffle=True)

            cv_train = train.iloc[train_index]
            cv_test = train.iloc[test_index]

            for m in cv_train.groupby('building_id'):
                test_subset = cv_test[cv_test.building_id == m[0]].index

                train.loc[test_subset, 'building_level_low'] = (m[1].interest_level == 'low').mean()
                train.loc[test_subset, 'building_level_medium'] = (m[1].interest_level == 'medium').mean()
                train.loc[test_subset, 'building_level_high'] = (m[1].interest_level == 'high').mean()

        self.attributes.extend(["building_level_low", "building_level_medium", "building_level_high"])

    def select_building_id_level_test(self, train):
        self.features.extend(["building_level_low", "building_level_medium", "building_level_high"])

        test = self.data

        test['building_level_low'] = np.nan
        test['building_level_medium'] = np.nan
        test['building_level_high'] = np.nan

        for m in train.groupby('manager_id'):
            test_subset = test[test.manager_id == m[0]].index

            test.loc[test_subset, 'building_level_low'] = (m[1].interest_level == 'low').mean()
            test.loc[test_subset, 'building_level_medium'] = (m[1].interest_level == 'medium').mean()
            test.loc[test_subset, 'building_level_high'] = (m[1].interest_level == 'high').mean()

        self.attributes.extend(["building_level_low", "building_level_medium", "building_level_high"])


class Model:
    def __init__(self, data, test):
        self.feature_selector = FeatureSelector(data.copy())
        print("Before Selection:", self.feature_selector.data.shape[0])
        num1 = self.data.shape[0]
        self.select_training_features(test)
        num2 = self.data.shape[0]
        print("Selection Done. Before sanitization:", self.feature_selector.data.shape[0])
        if num1 != num2:
            print("WARNING! SELECTION LEADING TO CHANGE IN ROWS!!!")

        self.sanitization()

    @property
    def data(self):
        return self.feature_selector.data

    @property
    def attributes(self):
        return self.feature_selector.attributes

    @property
    def features(self):
        return self.feature_selector.features

    def select_training_features(self, test):
        fs = self.feature_selector
        fs.select_numeric()
        fs.select_total_rooms()
        fs.select_num_features()
        fs.select_num_description()
        fs.select_num_photos()
        fs.select_price_per_bathrooms()
        fs.select_price_per_bedrooms()
        fs.select_price_per_total_rooms()
        fs.select_created(False, True, True)
        fs.select_feature_simple()
        fs.select_simple_cat_label_encode(test, address=True)
        fs.select_manager_id_level()
        # fs.select_distance()
        # fs.select_man_ability_train()
        # fs.select_feat_bigrams_train()
        # fs.select_display_address(test)
        # fs.select_street_address(test)
        # fs.select_description_stats(True, True, True)
        # fs.select_description() # always give after description_stats
        # fs.select_building_id_level()
        # fs.select_price_building_id_level()
        # fs.select_manager_id_building_id_level()

    def select_test_features(self, df):
        fs = FeatureSelector(df.copy())
        fs.select_numeric()
        fs.select_total_rooms()
        fs.select_num_features()
        fs.select_num_description()
        fs.select_num_photos()
        fs.select_price_per_bathrooms()
        fs.select_price_per_bedrooms()
        fs.select_price_per_total_rooms()
        fs.select_created(False, True, True)
        fs.select_feature_simple_test(self.feature_selector.cvect_feat_simple)
        fs.select_simple_cat_label_encode_test(self.feature_selector.simple_le_cat)
        fs.select_manager_id_level_test(self.data)
        # fs.select_distance()
        # fs.select_man_ability_test(self.feature_selector.man_df)
        # fs.select_feat_bigrams_test(self.feature_selector.cvect_feat, self.feature_selector.tfidf_feat)
        # fs.select_display_address_test(self.feature_selector.le_disp_add)
        # fs.select_street_address_test(self.feature_selector.le_street_add)
        # fs.select_description_stats(True, True, True)
        # fs.select_description_test(self.feature_selector.cvect_desc, self.feature_selector.tfidf_desc) # always give after description_stats
        # fs.select_building_id_level_test(self.data)
        # fs.select_price_building_id_level_test(self.data)
        # fs.select_manager_id_building_id_level_test(self.data)

        return fs.data

    def sanitization(self):
        tr = Transform(self.feature_selector.data)

        if 'price' in self.data:
            print("Sanitizing price...")
            tr.bound_drop('price', 15046)
            print("After sanitizing price:", self.data.shape[0])

        if 'distance' in self.data:
            print("Sanitizing distance...")
            tr.bound_drop('distance', 23)
            tr.bound_drop('distance', 0.14, False)
            print("After santitizing distance:", self.data.shape[0])

    def predict(self, data):
        test = self.select_test_features(data)
        prob = self.model.predict_proba(test[self.attributes])

        df = pd.DataFrame(prob, columns=self.model.classes_)
        return df.assign(listing_id=test['listing_id'].values)

class XGB_Model(Model):

    def fit(self, params_provided, validation=False):
        param = {}
        param['objective'] = 'multi:softprob'
        param['eta'] = params_provided['eta']
        param['max_depth'] = params_provided['max_depth']
        param['silent'] = 1
        param['num_class'] = 3
        param['eval_metric'] = "mlogloss"
        param['min_child_weight'] = params_provided['min_child_weight']
        param['subsample'] = params_provided['subsample']
        param['colsample_bytree'] = params_provided['colsample_bytree']
        param['seed'] = params_provided['seed']
        num_rounds = params_provided['num_rounds']

        self.interest_map = {'high':0, 'medium':1, 'low':2}

        train_X = self.data[self.attributes]
        train_y = np.array(self.data['interest_level'].apply(lambda x: self.interest_map[x]))

        param_lst = list(param.items())
        xgtrain = xgb.DMatrix(train_X, label=train_y)

        if validation:
            test = params_provided['test']
            test_interest_level = params_provided['test_interest_level']
            test_y = np.array(test_interest_level.apply(lambda x: self.interest_map[x]))
            xgtest = xgb.DMatrix(test[self.attributes], test_y)
            watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
            self.model = xgb.train(param_lst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
            return self.model.predict(xgtest)

        else:
            watchlist = [ (xgtrain,'train')]
            self.model = xgb.train(param_lst, xgtrain, num_rounds, watchlist)

        return self.model

    def predict(self, data):
        test_features = self.select_test_features(data)
        xgtest = xgb.DMatrix(test_features[self.attributes])
        probability = self.model.predict(xgtest)

        result = pd.DataFrame(data = {'listing_id': test_features['listing_id'].ravel()})
        result['low'] = probability[:, 2]
        result['medium'] = probability[:, 1]
        result['high'] = probability[:, 0]

        return result

class LGBM_Model(Model):

    def fit(self, params):
        self.model = LGB(
            n_estimators=params['n_estimators'],
            random_state=params['random_state'],
            metric="multi_logloss",
            categorical="name:street_address_simple_le,display_address_simple_le,manager_id_simple_le,building_id_simple_le"
        )
        self.model.fit(self.data[self.attributes], self.data['interest_level'])

        try:
            print("Feature Importance:", pd.DataFrame({'attribute': self.feature_selector.attributes, 'importance': self.model.feature_importances_}).sort_values('importance'))
        except AttributeError:
            pass

        try:
            print("Out of Bag estimate:", self.model.oob_score_)
        except AttributeError:
            pass

class LR_Model(Model):

    def fit(self, params):
        self.model = LR(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=params['max_iter'],
                random_state=params['random_state']
            ).fit(self.data[self.attributes], self.data['interest_level'])

        try:
            print("Feature Importance:", pd.DataFrame({'attribute': self.feature_selector.attributes, 'importance': self.model.feature_importances_}).sort_values('importance'))
        except AttributeError:
            pass

class RF_Model(Model):

    def fit(self, params):
        self.model = RF(
            n_estimators=params['n_estimators'],
            random_state=params['random_state'],
            metric="multi_logloss",
            oob_score=True
        )
        self.model.fit(self.data[self.attributes], self.data['interest_level'])

        try:
            print("Feature Importance:", pd.DataFrame({'attribute': self.feature_selector.attributes, 'importance': self.model.feature_importances_}).sort_values('importance'))
        except AttributeError:
            pass

        try:
            print("Out of Bag estimate:", self.model.oob_score_)
        except AttributeError:
            pass


class GBC_Model(Model):

    def fit(self, params):
        self.model = GBC(
            learning_rate=params['learning_rate'],
            subsample = params['subsample'],
            n_estimators= params['n_estimators'],
            max_depth= params['max_depth'],
            random_state= params['random_state']
        )
        self.model.fit(self.data[self.attributes], self.data['interest_level'])

        try:
            print("Feature Importance:", pd.DataFrame({'attribute': self.feature_selector.attributes, 'importance': self.model.feature_importances_}).sort_values('importance'))
        except AttributeError:
            pass


class Transform:
    """Used to transform DataFrame to sanitize data. All operations transform
    DataFrame.
    """
    def __init__(self, data):
        self.data = data

    def bound_correction(self, attr, bound, upper_bound=True):
        """Any row having 'attr' value greater than 'bound' is reassigned to
        'bound'.
        """
        data = self.data
        if upper_bound:
            index = data[data[attr] > bound].index
        else:
            index = data[data[attr] < bound].index

        data.loc[index, attr] = bound

        return data

    def bound_drop(self, attr, bound, upper_bound=True):
        """Any row having 'attr' value greater than 'bound' is dropped."""
        data = self.data
        if upper_bound:
            index = data[data[attr] >= bound].index
        else:
            index = data[data[attr] <= bound].index

        data.drop(index, inplace=True)

        return data

def validation(model_type):
    data = pd.read_json("./input/train.json")

    cv = StratifiedKFold(n_splits=4, random_state=200)
    scores = []

    for train_ind, validate_ind in cv.split(data, data.interest_level):
        print("\n")
        train = data.iloc[train_ind]
        validate = data.iloc[validate_ind]

        md = None
        pred = None
        ll = None

        if model_type == "XGB":
            md = XGB_Model(train, validate)
            params = dict([('eta', 0.02), ('max_depth', 5), ('min_child_weight', 1), ('seed', 10),
                            ('subsample', 0.7), ('colsample_bytree', 0.7), ('num_rounds', 1700)])
            params['test'] = md.select_test_features(validate)[md.attributes]
            params['test_interest_level'] = validate['interest_level']
            pred = md.fit(params, validation=True)
            test_y = np.array(validate['interest_level'].apply(lambda x: md.interest_map[x]))
            ll = log_loss(test_y, pred)
        elif model_type == "LGBM":
            md = LGBM_Model(train, validate)
            proc_validate = md.select_test_features(validate)[md.attributes]
            params = dict([('n_estimators', 300), ('random_state', 200)])
            md.fit(params)
            pred = md.model.predict_proba(proc_validate)
            ll = log_loss(validate.interest_level, output, labels=md.model.classes_)
        elif model_type == "RF":
            md = RF_Model(train, validate)
            proc_validate = md.select_test_features(validate)[md.attributes]
            params = dict([('n_estimators', 100), ('random_state', 100), ('learning_rate', 0.08), ('subsample', 0.8), ('max_depth', 8)])
            md.fit(params)
            pred = md.model.predict_proba(proc_validate)
            ll = log_loss(validate.interest_level, output, labels=md.model.classes_)
        elif model_type == "LR":
            md = LR_Model(train, validate)
            proc_validate = md.select_test_features(validate)[md.attributes]
            params = dict([('max_iter', 100), ('random_state', 100)])
            md.fit(params)
            pred = md.model.predict_proba(proc_validate)
            ll = log_loss(validate.interest_level, output, labels=md.model.classes_)
        else:
            print("Give the model type !!!")
            return

        ll = log_loss(test_y, pred)
        scores.append(ll)
        print("Log-Loss: ", ll)

    for i in range(len(scores)):
        print("Loss", i, ":", scores[i])

    avgloss = (sum(scores)) / len(scores)
    print("Average Loss:", avgloss)

    print("Features Used:")
    print(md.features)

def create_submission(filename, to_be_pickled):
    test = pd.read_json("./input/test.json")
    train = pd.read_json("./input/train.json")

    md = XGB_Model(train, test)
    params = dict([('eta', 0.02), ('max_depth', 5), ('min_child_weight', 1), ('seed', 10), ('subsample', 0.7),
                    ('colsample_bytree', 0.7), ('num_rounds', 1700)])

    md.fit(params)
    output = md.predict(test)

    print("File:", filename)
    print(output.shape)
    print(output.head())

    output.to_csv(filename, index=False)

    #--------------------------pickling the model---------------------------------

    if to_be_pickled:
        pickle_write = open("model.pickle","wb")
        pickle.dump(md, pickle_write)
        pickle_write.close()


def create_submission_from_pickled_model(filename):
    test = pd.read_json("./input/test.json")

    pickle_read = open('./model.pickle', 'rb')
    md = pickle.load(pickle_read)
    output = md.predict(test)

    print("File:", filename)
    print(output.shape)
    print(output.head())

    output.to_csv(filename, index=False)
    pickle_read.close()

#----------------------------- validation --------------------------------
# validation("XGB")

#--------------------------create submission without using pickeled model-----------------
# create_submission("hyperparameter_training.csv", to_be_pickled=False)

#--------------------------create submission using pickeled model-----------------
create_submission_from_pickled_model("xgb_training.csv")
