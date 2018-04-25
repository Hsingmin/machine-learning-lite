# Level Order Traverse a Directory
import os
import codecs
import string


TRAIN_IMAGE_DIR = 'd:/engineering-data/Ali-ICPR-data/train_image_9000'
TRAIN_BBOX_DIR = 'd:/engineering-data/Ali-ICPR-data/train_txt_9000'

def traverse_directory(path):
	sub_path_list = []
	file_path_list = []
	for rootdir, subdirs, filenames in os.walk(path):
		for subdir in subdirs:
			sub_path_list.append(os.path.join(rootdir, subdir))

		for filename in filenames:
			file_path_list.append(os.path.join(rootdir, filename))
	return sub_path_list, file_path_list

def test_traverse_directory(arg=None):
	path = TRAIN_IMAGE_DIR
	sub_directories, files = traverse_directory(path)
	print('sub directory list is : ', sub_directories)
	print('file list is : ', files)

	print("%d images in current directory" %(len(files)))

def get_single_image_bboxes(bbox_dir):

	bbox_list = []
	labels = []
	with codecs.open(bbox_dir, 'r', 'utf-8') as bf:
		for line in bf:
			box = line.strip().split(',')
			labels.append(box[-1])
			bbox_list.append(box[:-1])

	'''
	sized_image = np.asarray(image_data.eval(session=sess), dtype='uint8')
	height = len(sized_image)
	width = len(sized_image[0])
	'''
	bboxes = []
	for bbox in bbox_list:
		try:
			bboxes.append([int(float(bbox[1])), int(float(bbox[0])), int(float(bbox[5])), int(float(bbox[4]))])
		except ValueError as e:
			print("error", e, "on bbox : ", bbox)
			continue
	return bboxes, labels

if __name__ == '__main__':
	bbox_dir = os.path.join(TRAIN_BBOX_DIR, "T10o9DFaRXXXXXXXXX_!!0-item_pic.jpg.txt")
	get_single_image_bboxes(bbox_dir)










