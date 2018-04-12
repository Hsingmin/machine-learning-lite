# -*- coding: utf-8 -*-
#
# -*- version:
#		python 3.5.2
#		numpy 1.13.1
#		pandas 0.20.3
#		sklearn 0.19.0
#
# -*- author: Hsingmin Lee
#
# ijcai_data_visualizer.py -- Convert given dataset *.txt to *.csv format 
# and get data visualization .

import os
import sys
import codecs
import numpy as np
import pandas as pd
import sklearn as sk
import csv
import matplotlib.pyplot as plt

# Train slices store path
DATASET_DIR = 'd:/engineering-data/Ali-IJCAI-18-data'

TRAIN_DATASET_RAW = 'round1_ijcai_18_train_20180301.txt'

TEST_DATASET_RAW = 'round1_ijcai_18_test_a_20180301.txt'

TRAIN_DATASET_CSV = 'ijcai_18_train_dataset.csv'
TEST_DATASET_CSV = 'ijcai_18_test_dataset.csv'

# Get columns header keywords and write into file.
# args:
#	None
# returns:
#	saved as ./columns.txt
def get_column_header():
	# pd.read_csv(dataset_raw, sep=' ', usecols=[0,6]).to_csv(dataset_csv)
	with codecs.open(dataset_raw, 'r', 'utf-8') as infile:
		line = infile.readline()
		columns = line.split(' ')
		with codecs.open("./columns.txt", 'w', 'utf-8') as outfile:
			for i in range(len(columns)):
				column = columns[i]
				outfile.write(str(i) + ':' + column + '\r\n')

# Split raw dataset into different parts and convert into .csv format.
# args:
#	None
# returns:
#	saved as DATASET/train_trade.csv
#		 DATASET/train_item.csv
#		 DATASET/train_user.csv
#		 DATASET/train_context.csv
#		 DATASET/train_shop.csv
def split_dataset():
	# Get is_trade distribution 
	pd.read_csv(dataset_raw, sep=' ', usecols=[26]).to_csv(
			os.path.join(DATASET_DIR, "train_trade.csv"))
	
	# Get item distribution
	cols = [x for x in range(1, 10)]
	cols.append(26)
	pd.read_csv(dataset_raw, sep=' ', usecols=cols).to_csv(
			os.path.join(DATASET_DIR, "train_item.csv"))
	
	# Get user distribution
	cols = [x for x in range(10, 15)]
	cols.append(26)
	pd.read_csv(dataset_raw, sep=' ', usecols=cols).to_csv(
			os.path.join(DATASET_DIR, "train_user.csv"))

	# Get context distribution
	cols = [x for x in range(15, 19)]
	cols.append(26)
	pd.read_csv(dataset_raw, sep=' ', usecols=cols).to_csv(
			os.path.join(DATASET_DIR, "train_context.csv"))
	
	# Get shop distribution
	cols = [x for x in range(19, 27)]
	pd.read_csv(dataset_raw, sep=' ', usecols=cols).to_csv(
			os.path.join(DATASET_DIR, "train_shop.csv"))

# Get is_trade field data distribution with pie gram.
# args:
#	None
# returns:
#	saved as is_trade distribution.png 
def visualize_trade_distribution():
	df = pd.read_csv(os.path.join(DATASET_DIR, "train_trade.csv"))
	
	df.loc['trade_sums'] = df.apply(lambda x: x.sum())
	drade_count = df.loc['trade_sums']['is_trade']
	slices = [trade_count, len(df)-trade_count]
	activities = ['is_trade=1', 'is_trade=0']
	colors = ['g', 'r']
	plt.pie(slices, labels=activities,
		colors=colors, startangle=90,
		shadow=True, explode=(0.1, 0),
		autopct='%1.1f%%')
	plt.title('is_trade distribution')
	plt.show()

# Get item field data distribution with histogram.
# args:
#	None
# returns:
#	saved as item distribution.png 
#		 top20 items.png
def visualize_item_distribution():
	df = pd.read_csv(os.path.join(DATASET_DIR, "train_item.csv"))
	item_dict = {}
	
	for item in df['item_id']:
		if item in item_dict:
			item_dict[item] += 1
		else:
			item_dict.update({item: 1})
	
	item_count_dict = {}
	for item in item_dict:
		if item_dict[item] in item_count_dict:
			item_count_dict[item_dict[item]] += 1
		else:
			item_count_dict.update({item_dict[item]: 1})
	
	plt.bar([c for c in range(1, 51)], [item_count_dict[c] for c in range(1, 51)],
			label="Item Count", color='g')
	plt.xlabel('item occur times')
	plt.ylabel('occur times counts')
	plt.title('Item Distribution')
	plt.legend()
	plt.show()
	
	item_list = []
	item_times_list = []
	st = sorted(item_dict.items(), key=lambda e: e[1], reverse=True)
	for item in st:
		if item[1] > 1300:
			item_list.append(item[0])
			item_times_list.append(item[1])
	with codecs.open('./item_axis.txt', 'w', 'utf-8') as outfile:
		for i in range(len(item_list)):
			outfile.write(str(i) + ':' + str(int(item_list[i])) + '\r\n')
	plt.bar([id for id in range(len(item_list))], item_times_list, 
			label="Item Count", color='g')
	plt.xlabel('item id')
	plt.ylabel('occur times')
	plt.title('Top20 Items')
	plt.legend()
	plt.show()

# Get shop field data distribution with histogram.
# args:
#	None
# returns:
#	saved as shop distribution.png 
#		 top20 shops.png
def visualize_shop_distribution():
	df = pd.read_csv(os.path.join(DATASET_DIR, "train_shop.csv"))
	shop_dict = {}
	
	for shop in df['shop_id']:
		if shop in shop_dict:
			shop_dict[shop] += 1
		else:
			shop_dict.update({shop: 1})
	
	shop_count_dict = {}
	for shop in shop_dict:
		if shop_dict[shop] in shop_count_dict:
			shop_count_dict[shop_dict[shop]] += 1
		else:
			shop_count_dict.update({shop_dict[shop]: 1})
	
	plt.bar([c for c in range(1, 51)], [shop_count_dict[c] for c in range(1, 51)],
			label="Shop Count", color='g')
	plt.xlabel('shop occur times')
	plt.ylabel('occur times counts')
	plt.title('Shop Distribution')
	plt.legend()
	plt.show()
	
	shop_list = []
	shop_times_list = []
	st = sorted(shop_dict.items(), key=lambda e: e[1], reverse=True)
	for shop in st:
		if shop[1] > 2000:
			shop_list.append(shop[0])
			shop_times_list.append(shop[1])
	with codecs.open('./shop_axis.txt', 'w', 'utf-8') as outfile:
		for i in range(len(shop_list)):
			outfile.write(str(i) + ':' + str(shop_list[i]) + '\r\n')
	plt.bar([id for id in range(len(shop_list))], shop_times_list, 
			label="Shop Count", color='g')
	plt.xlabel('shop id')
	plt.ylabel('occur times')
	plt.title('Top20 Shops')
	plt.legend()
	plt.show()

# Get user field data distribution with pie gram for user_gender_id 
# and user_star_level, hitogram for user_age_level and user_occupation_id.
# args:
#	None
# returns:
#	saved as user_gender distribution.png
#		 user_star_level distribution.png
#		 user_age_level distribution.png
#		 user_occupation_id distribution.png
def visualize_user_distribution():
	df = pd.read_csv(os.path.join(DATASET_DIR, "train_user.csv"))
	
	# Get user gender distribution .
	unknown_count = 0
	female_count = 0
	male_count = 0
	family_count = 0
	for id in df['user_gender_id']:
		if id == 0:
			female_count += 1
		elif id == 1:
			male_count += 1
		elif id == 2:
			family_count += 1
		else:
			unknown_count += 1
	slices = [unknown_count, female_count, male_count, family_count]
	activities = ['unknown', 'female', 'male', 'family']
	colors = ['g', 'r', 'b', 'y']
	plt.pie(slices, labels=activities,
		colors=colors, startangle=90,
		shadow=True, explode=(0.1, 0, 0, 0),
		autopct='%1.1f%%')
	plt.title('User Gender distribution')
	plt.show()

	# Get user star level distribution .
	user_star_dict = {}
	for level in df['user_star_level']:
		if level in user_star_dict:
			user_star_dict[level] += 1
		else:
			user_star_dict.update({level: 1})
	slices = []
	activities = []
	for star_level in user_star_dict:
		slices.append(user_star_dict[star_level])
		activities.append(str(star_level))
	# print('len(slices) = %d, len(activities) = %d' %(len(slices), len(activities)))
	colors = ['g', 'r', 'b', 'y', 'k', 'm', 'c', 'b', 'g', 'r', 'c', 'y']
	plt.pie(slices, labels=activities,
		colors=colors, startangle=90,
		shadow=True, explode=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
		autopct='%1.1f%%')
	plt.title('User star level distribution')
	plt.show()
	
	# Get user age level distribution .
	user_age_dict = {}
	for age in df['user_age_level']:
		if age in user_age_dict:
			user_age_dict[age] += 1
		else:
			user_age_dict.update({age: 1})
	age_list = []
	count_list = []
	user_age_dict = sorted(user_age_dict.items(), key=lambda e: e[0])
	for age_level in user_age_dict:
		if age_level[0] == -1:
			age_list.append(999)
		else:
			age_list.append(age_level[0])
		count_list.append(age_level[1])
	plt.bar(age_list, count_list, 
			label="age level count", color='g')
	plt.xlabel('age level')
	plt.ylabel('counts')
	plt.title('User Age Distribution')
	plt.legend()
	plt.show()
	
	# Get user age level distribution .
	user_occupation_dict = {}
	for occupation in df['user_occupation_id']:
		if occupation in user_occupation_dict:
			user_occupation_dict[occupation] += 1
		else:
			user_occupation_dict.update({occupation: 1})
	occupation_list = []
	count_list = []
	user_occupation_dict = sorted(user_occupation_dict.items(), key=lambda e: e[0])
	for occupation in user_occupation_dict:
		if occupation[0] == -1:
			occupation_list.append(2001)
		else:
			occupation_list.append(occupation[0])
		count_list.append(occupation[1])
	plt.bar(occupation_list, count_list, 
			label="occupation count", color='g')
	plt.xlabel('occupation id')
	plt.ylabel('counts')
	plt.title('User Occupation Distribution')
	plt.legend()
	plt.show()

# Get shop field data distribution with hitogram for shop_review_num_level and
# shop_star_level, fill-line graph for shop_review_positive_rate, shop_score_service,
# shop_score_delivery and shop_score_description.
# args:
#	None
# returns:
#	saved as shop_review_num_level distribution.png
#		 shop_star_level distribution.png
#		 shop_review_positive_rate distribution.png
#		 shop_score_service distribution.png
# 		 shop_score_delivery distribution.png
#		 shop_score_description distribution.png.
def visualize_shop_score_distribution():
	df = pd.read_csv(os.path.join(DATASET_DIR, "train_shop.csv"))
	
	# Get shop star level distribution .
	shop_star_dict = {}
	for star in df['shop_star_level']:
		if star in shop_star_dict:
			shop_star_dict[star] += 1
		else:
			shop_star_dict.update({star: 1})
	star_list = []
	count_list = []
	shop_star_dict = sorted(shop_star_dict.items(), key=lambda e: e[0])
	for shop in shop_star_dict:
		star_list.append(shop[0])
		count_list.append(shop[1])
	plt.bar(star_list, count_list, 
			label="shop star level count", color='g')
	plt.xlabel('shop star level')
	plt.ylabel('counts')
	plt.title('Shop Star Level Distribution')
	plt.legend()
	plt.show()

	# Get shop review num level distribution .
	shop_review_num_dict = {}
	for review in df['shop_review_num_level']:
		if review in shop_review_num_dict:
			shop_review_num_dict[review] += 1
		else:
			shop_review_num_dict.update({review: 1})
	review_list = []
	count_list = []
	shop_review_num_dict = sorted(shop_review_num_dict.items(), key=lambda e: e[0])
	for shop in shop_review_num_dict:
		review_list.append(shop[0])
		count_list.append(shop[1])
	plt.bar(review_list, count_list, 
			label="shop review num level count", color='g')
	plt.xlabel('shop review num level')
	plt.ylabel('counts')
	plt.title('Shop Review Num Level Distribution')
	plt.legend()
	plt.show()
	
	# Get shop review positive rate distribution .
	shop_review_rate_dict = {75: 0, 80: 0, 85: 0, 90: 0, 95: 0, 100: 0}
	outliers = []
	for rate in df['shop_review_positive_rate']:
		if rate >= 0.980 and rate < 0.985:
			shop_review_rate_dict[80] += 1
		elif rate >= 0.985 and rate < 0.990:
			shop_review_rate_dict[85] += 1
		elif rate >= 0.990 and rate < 0.995:
			shop_review_rate_dict[90] += 1
		elif rate >= 0.995 and rate < 1.000:
			shop_review_rate_dict[95] += 1
		elif rate >= 1.000:
			shop_review_rate_dict[100] += 1
		else:
			shop_review_rate_dict[75] += 1
			outliers.append(rate)
	interval_list = [75, 80, 85, 90, 95, 100]
	count_list = [shop_review_rate_dict[key] for key in interval_list]
	#print(outliers)
	print(shop_review_rate_dict)
	plt.bar(interval_list, count_list, 
			label="shop review positive rate count", color='g')
	plt.xlabel('shop review positive rate')
	plt.ylabel('counts')
	plt.title('Shop Review Positive Rate Distribution')
	plt.legend()
	plt.show()
	
	# Get shop score service distribution .
	shop_score_service_dict = {-2: 0, 0: 0, 2: 0, 4: 0, 6: 0, 8: 0, 10: 0}
	outliers = []
	for score in df['shop_score_service']:
		if score >= 0.90 and score < 0.92:
			shop_score_service_dict[0] += 1
		elif score >= 0.92 and score < 0.94:
			shop_score_service_dict[2] += 1
		elif score >= 0.94 and score < 0.96:
			shop_score_service_dict[4] += 1
		elif score >= 0.96 and score < 0.98:
			shop_score_service_dict[6] += 1
		elif score >= 0.98 and score < 1.00:
			shop_score_service_dict[8] += 1
		elif score >= 1.00:
			shop_score_service_dict[10] += 1
		else:
			shop_score_service_dict[-2] += 1
			outliers.append(score)
	interval_list = [-2, 0, 2, 4, 6, 8, 10]
	count_list = [shop_score_service_dict[key] for key in interval_list]
	print(shop_score_service_dict)
	plt.bar(interval_list, count_list, 
			label="shop score service count", color='g')
	plt.xlabel('shop score service')
	plt.ylabel('counts')
	plt.title('Shop Score Service Distribution')
	plt.legend()
	plt.show()
	
	# Get shop score delivery distribution .
	shop_score_delivery_dict = {-2: 0, 0: 0, 2: 0, 4: 0, 6: 0, 8: 0, 10: 0}
	outliers = []
	for score in df['shop_score_delivery']:
		if score >= 0.90 and score < 0.92:
			shop_score_delivery_dict[0] += 1
		elif score >= 0.92 and score < 0.94:
			shop_score_delivery_dict[2] += 1
		elif score >= 0.94 and score < 0.96:
			shop_score_delivery_dict[4] += 1
		elif score >= 0.96 and score < 0.98:
			shop_score_delivery_dict[6] += 1
		elif score >= 0.98 and score < 1.00:
			shop_score_delivery_dict[8] += 1
		elif score >= 1.00:
			shop_score_delivery_dict[10] += 1
		else:
			shop_score_delivery_dict[-2] += 1
			outliers.append(score)
	interval_list = [-2, 0, 2, 4, 6, 8, 10]
	count_list = [shop_score_delivery_dict[key] for key in interval_list]
	print(shop_score_delivery_dict)
	plt.bar(interval_list, count_list, 
			label="shop score delivery count", color='g')
	plt.xlabel('shop score delivery')
	plt.ylabel('counts')
	plt.title('Shop Score Delivery Distribution')
	plt.legend()
	plt.show()
	
	# Get shop score description distribution .
	shop_score_description_dict = {-2: 0, 0: 0, 2: 0, 4: 0, 6: 0, 8: 0, 10: 0}
	outliers = []
	for score in df['shop_score_description']:
		if score >= 0.90 and score < 0.92:
			shop_score_description_dict[0] += 1
		elif score >= 0.92 and score < 0.94:
			shop_score_description_dict[2] += 1
		elif score >= 0.94 and score < 0.96:
			shop_score_description_dict[4] += 1
		elif score >= 0.96 and score < 0.98:
			shop_score_description_dict[6] += 1
		elif score >= 0.98 and score < 1.00:
			shop_score_description_dict[8] += 1
		elif score >= 1.00:
			shop_score_description_dict[10] += 1
		else:
			shop_score_description_dict[-2] += 1
			outliers.append(score)
	interval_list = [-2, 0, 2, 4, 6, 8, 10]
	count_list = [shop_score_description_dict[key] for key in interval_list]
	print(shop_score_description_dict)
	plt.bar(interval_list, count_list, 
			label="shop score description count", color='g')
	plt.xlabel('shop score description')
	plt.ylabel('counts')
	plt.title('Shop Score Description Distribution')
	plt.legend()
	plt.show()	

# Get item field data distribution with pie for item_brand_id, item_city_id 
# and item_collected_level, histogram for item_sales_level.
# args:
#	None
# returns:
#	saved as item_brand_id distribution.png 
#		 item_city_id distribution.png
#		 item_collected_id distribution.png
#

def get_item_field_pie(field):
	df = pd.read_csv(os.path.join(DATASET_DIR, "train_item.csv"))
	
	# Get item brand distribution .
	item_field_dict = {}
	for f in df[field]:
		if f in item_field_dict:
			item_field_dict[f] += 1
		else:
			item_field_dict.update({f: 1})
	item_field_dict = sorted(item_field_dict.items(), key=lambda e: e[1], reverse=True)

	item_field_list = []
	item_field_times = []

	for f in item_field_dict:
		item_field_list.append(f[0])
		item_field_times.append(f[1])
	slices = [item_field_times[i] for i in range(10)]
	slices.append(sum(item_field_times[10:]))
	activities = [item_field_list[i] for i in range(10)]
	activities.append('other')

	colors = ['g', 'r', 'b', 'y', 'k', 'm', 'c', 'b', 'g', 'r', 'c']
	plt.pie(slices, labels=activities,
		colors=colors, startangle=90,
		shadow=True, explode=(0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
		autopct='%1.1f%%')
	plt.title('Item ' + field.split('_')[1].upper() + ' Distribution')
	plt.show()

def get_item_field_histogram(field):
	df = pd.read_csv(os.path.join(DATASET_DIR, "train_item.csv"))
	
	# Get item brand distribution .
	item_field_dict = {}
	for f in df[field]:
		if f in item_field_dict:
			item_field_dict[f] += 1
		else:
			item_field_dict.update({f: 1})
	item_field_dict = sorted(item_field_dict.items(), key=lambda e: e[0])

	item_field_list = []
	item_field_times = []

	for f in item_field_dict:
		item_field_list.append(f[0])
		item_field_times.append(f[1])
	
	#interval_list = [-2, 0, 2, 4, 6, 8, 10]
	#count_list = [shop_score_description_dict[key] for key in interval_list]
	field = field.replace('_', ' ')
	plt.bar(item_field_list, item_field_times, 
			label="%s count" %field, color='g')
	plt.xlabel(field)
	plt.ylabel('counts')
	plt.title('%s Distribution' %(field.upper()))
	plt.legend()
	plt.show()

# Convert dataset raw txt format into csv.
# args:
#	None
# returns:
#	None
def raw_convert_csv(dataset):
	if dataset == 'training':
		dataset_csv = os.path.join(DATASET_DIR, TRAIN_DATASET_CSV)
		dataset_raw = os.path.join(DATASET_DIR, TRAIN_DATASET_RAW)
	elif dataset == 'test':
		dataset_csv = os.path.join(DATASET_DIR, TEST_DATASET_CSV)
		dataset_raw = os.path.join(DATASET_DIR, TEST_DATASET_RAW)
	else:
		print('no dataset intended.')
		return

# Execute every function one by one to visualize the column data distribution.
if __name__ == '__main__':
	# visualize_item_distribution()	
	# visualize_user_distribution()
	# visualize_shop_distribution()
	# visualize_shop_score_distribution()
	# visualize_item_detail_distribution()
	
	# get_item_field_pie('item_city_id')
	# get_item_field_pie('item_brand_id')
	# get_item_field_histogram('item_price_level')
	# get_item_field_histogram('item_sales_level')
	# get_item_field_histogram('item_pv_level')
	# get_item_field_histogram('item_collected_level')











