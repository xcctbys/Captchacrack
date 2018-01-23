# -*- coding:UTF-8 -*-
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals import joblib
import os
import copy
import datetime

class ShanghaiCaptchaSolver(object):
	"""用于破解企业征信上海验证码

	参数
	－－－－
	train: bool
		所建对象是否用于训练模型，用于训练模型则为True；用于破解则为False
	training_images_path: str
		用于训练模型的验证码图片所在文件夹路径
	checking_path: str
		用于检查自动分割效果的文件夹所在路径，可为空。为空时，检查路径自动设置
		在图片路径下的checking文件夹中

	属性
	－－－－
	width: int
		验证码图片宽度
	height: int
		验证码图片高度
	char_width: int
		验证码图片中单个字符的标准宽度
	char_height: int
		验证码图片中单个字符的标准高度
	n_chars: int
		验证码图片中的字符个数
	"""
	def __init__(self, train=False, training_images_path=None, checking_path=None):
		parent = os.path.dirname(__file__)

		self.width = 160
		self.height = 53
		self.char_width = 25
		self.char_height = 30
		self.n_chars = 5
		self.train = train
		self.model_path = os.path.join(parent, "model")
		self.num_model_path = os.path.join(self.model_path, 'num', 'model.m')
		self.sym_model_path = os.path.join(self.model_path, 'sym', 'model.m')
		self.num_model = None
		self.sym_model = None
		if train:
			if training_images_path:
				self.training_images_path = training_images_path
			else:
				print u"当前对象无法实现预期作用，进行模型训练"
				print u"请提供训练模型验证码图片所在路径"
				self.training_images_path = None
			self.checking_path = checking_path

	def duplicate_check(self):
		"""检查训练图片集中的重复个数

		返回值
		－－－－
		all_images: DataFrame [n_samples, 2]
			返回所有图片的文件名和像素信息
		"""
		if not self.train:
			print u"非训练对象，该方法可能不可用"
			return None

		images = self.find_images()
		ims_data = []
		for image in images:
			ims_data.append(str(list(Image.open(image).getdata())))
		all_images = pd.DataFrame({'image': images, 'pixels': ims_data})
		n_dup = sum(all_images.pixels.duplicated(keep=False))
		print u"重复图片共有%s个" % n_dup
		return all_images.ix[all_images.pixels.argsort(),:]

	def find_images(self):
		"""找到训练图片集中所有图片的路径

		返回值
		－－－－
		images: list
			所有验证码图片的绝对路径
		"""
		if not self.train:
			print u"非训练对象，该方法可能不可用"
			return None

		files = os.listdir(self.training_images_path)
		images = [os.path.join(self.training_images_path, f) for f in files if f.endswith('.jpg')]

		return images

	def image_combine(self):
		"""垂直组合所有验证码图片

		返回值
		－－－－
		True: bool
			当程序正常进行时可以看到组合图片，并返回True
		"""
		image_files = self.find_images()
		ims = map(Image.open, image_files)

		widths, heights = zip(*(i.size for i in ims))
		max_width = max(widths)
		total_height = sum(heights)

		new_im = Image.new('RGB', (max_width, total_height))

		x_offset = 0
		for im in ims:
		  new_im.paste(im, (0, x_offset))
		  x_offset += im.size[1]
		new_im.show()
		return True

	def _detect_line(self, pix_vec, threshold=2):
		"""检测像素矩阵中的线

		参数
		－－－－
		pix_vec: {array-like} [self.width*self.height]
			整个验证码图片的像素矩阵，需经过2值化
		threshold: int
			确定为噪声线的周围不为零像素点个数，默认设置为2

		返回值
		－－－－
		isbkg_noise: {array-like} [self.width*self.height]
			boolean array，当isbkg_noise[i]为True，代表像素矩阵中i位置
			像素为噪声线或背景
		"""
		upper = np.append(np.zeros(self.width), pix_vec[:len(pix_vec)-self.width])
		upper_left = np.append(0, upper[1:])
		upper_right = np.append(upper[:-1], 0)
		left = np.append(0, pix_vec[1:])
		right = np.append(pix_vec[:-1], 0)
		bottom = np.append(pix_vec[self.width:], np.zeros(self.width))
		bottom_left = np.append(0, bottom[1:])
		bottom_right = np.append(bottom[:-1], 0)

		around = np.array([upper_left, upper, upper_right, 
			left, right, bottom_left, bottom, bottom_right]).T
		rsum_around = np.sum(around, axis=1)
		isbkg_noise = (rsum_around <= threshold)
		return isbkg_noise

	def de_noise(self, image_path):
		"""检测原图像中的噪点和噪声线

		参数
		－－－－
		image_path: str
			单个验证码图像的路径

		返回值
		－－－－
		new_im: Image
			去噪后的验证码图片
		"""
		image = Image.open(image_path)
		pixs = np.array(image.convert('L').getdata())
		pixs[pixs < 220] = 1
		pixs[pixs >= 220] = 0
		isbkg_noise = self._detect_line(pixs)

		image_rgb = list(image.getdata())
		i = 0
		for bkg_noise in isbkg_noise:
			if bkg_noise:
				image_rgb[i] = (255, 255, 255)
			i += 1
		new_im = Image.new('RGB', (self.width, self.height))
		new_im.putdata(image_rgb)

		return new_im

	def de_line(self, pix_vec):
		"""对聚类后形成的新图像的像素矩阵进行去线操作，以提高分割准确性

		参数
		－－－－
		pix_vec: {array-like} [self.width*self.height]
			2值化后的图像像素矩阵

		返回值
		－－－－
		pix_vec: {array-like} [self.width*self.height]
			去线后的图像像素矩阵
		"""
		isnoise = self._detect_line(pix_vec, 2)
		pix_vec[isnoise] = 0
		return pix_vec

	def clustering_captcha(self, image_path, check=False):
		"""对验证码图像进行聚类操作以分离出验证码图片中的各个字符

		参数
		－－－－
		image_path: str
			单个验证码图片的绝对路径
		check: bool
			是否对聚类后的验证码图片检查聚类效果及基于列的像素点分布图

		返回值
		－－－－
		(image_vectors, col_npixs): tuple [2]
			长度为2的tuple，其中tuple的第一个对象为根据聚类得到的除背景以外的
			所有类的像素矩阵，tuple的第二个对象为第一个对象所形成图像的每一列
			的非背景像素个数
			image_vectors: {array-like} [self.width * self.height, self.n_chars + 1]
			col_npixs: {array-like} [self.width, self.n_chars + 1]
		"""
		image = self.de_noise(image_path)
		image_pixs = np.array(image.getdata())
		image_pixs = image_pixs.astype(np.float)

		sc = StandardScaler()
		km = KMeans(n_clusters=(self.n_chars + 2))
		clu = Pipeline(steps=[('sc', sc), ('km', km)])
		clusters = clu.fit_predict(image_pixs)

		image_vectors = np.zeros((self.n_chars+2, self.width*self.height))
		col_npixs = np.zeros((self.n_chars+2, self.width))

		for i in np.unique(clusters):
			image_vectors[i, clusters == i] = 1
			image_vectors[i, :] = self.de_line(image_vectors[i, :])
			col_npixs[i, :] = image_vectors[i, :].reshape((
				self.height, self.width)).sum(axis=0)
		cluster_bkg = np.argmax(col_npixs.sum(axis=1))
		image_vectors = np.delete(image_vectors, (cluster_bkg), axis=0)
		col_npixs = np.delete(col_npixs, (cluster_bkg), axis=0)

		if check:
			if not self.checking_path:
				self.checking_path = os.path.join(self.training_images_path, 'checking')

			if not os.path.isdir(self.checking_path):
				os.mkdir(self.checking_path)

			clusters_path = os.path.join(self.checking_path, 'clusters')
			if not os.path.isdir(clusters_path):
				os.mkdir(clusters_path)

			n_clusters = col_npixs.shape[0]
			img_name = os.path.split(image_path)[1].split('.')[0]
			for i in range(n_clusters):
				new_img_name = os.path.join(clusters_path, 
					img_name + '_cluster' + str(i) + '_img' + '.jpg')
				new_fig_name = os.path.join(clusters_path, 
					img_name + '_cluster' + str(i) + '_fig' + '.jpg')

				im_new = Image.new('1', (self.width, self.height))
				im_new.putdata(image_vectors[i, :])
				im_new.save(new_img_name)
				plt.plot(col_npixs[i, :])
				plt.savefig(new_fig_name)
				plt.close('all')

		return (image_vectors, col_npixs)

	def detect_chars(self, npix_col, lefts, rights):
		"""探测字符存在的可能区间

		作用于函数adjusted_positions当中
		"""
		char_lefts = []
		char_rights = []
		for left, right in zip(lefts, rights):
			area_max = max(npix_col[left:(right+1)])
			if area_max >= 8:
				char_lefts.append(left)
				char_rights.append(right)
		return (char_lefts, char_rights)

	def combine_dup(self, adjusted_lefts, adjusted_rights):
		"""合并可能为同一个字符的位置组合
		作用于函数adjusted_positions当中
		"""
		combined_lefts = []
		combined_rights = []

		adjusted_chunks = zip(adjusted_lefts, adjusted_rights)
		adjusted_chunks = sorted(list(set(adjusted_chunks)))

		if len(adjusted_chunks) == 0:
			return (np.array([]), np.array([]))

		if len(adjusted_chunks) == 1:
			return (np.array([adjusted_chunks[0][0]]), np.array([adjusted_chunks[0][1]]))

		left, right = adjusted_chunks[0]
		for i in range(1, len(adjusted_chunks)):
			left_tmp, right_tmp = adjusted_chunks[i]
			if right <= left_tmp:
				combined_lefts.append(left)
				combined_rights.append(right)
				left = left_tmp
				right = right_tmp
			else:
				right = right_tmp
		combined_lefts.append(left)
		combined_rights.append(right)
		return (np.array(combined_lefts), np.array(combined_rights))

	def adjust_positions(self, npix_col, lefts, rights):
		"""对初步切割后的切割点位置进行调整

		参数
		－－－－
		npix_col: {array-like} [self.width]
			聚类后形成图片每列不为背景的像素个数
		lefts: {array-like}
			get_positions初步分割后得到的分割点左边界
		rights: {array-like}
			get_positions初步分割后得到的分割点右边界

		返回值
		－－－－
		(adjusted_lefts, adjusted_rights): tuple [2]
			经过调整的分割点左边界和右边界
			adjusted_lefts: {array-like}
			adjusted_rights: {array-like}
		"""
		char_lefts, char_rights = self.detect_chars(npix_col, lefts, rights)

		adjusted_lefts = []
		adjusted_rights = []
		for left, right in zip(char_lefts, char_rights):
			dis = right - left + 1
			if dis >= 18:
				adjusted_lefts.append(left)
				adjusted_rights.append(right)
			else:
				search_left = left - (self.char_width - dis)
				search_right = right + (self.char_width - dis)
				lefts_tmp = lefts[(lefts >= search_left) & (rights <= search_right)]
				rights_tmp = rights[(lefts >= search_left) & (rights <= search_right)]
				chunks_tmp = zip(lefts_tmp, rights_tmp)
				start_chunk_tmp = chunks_tmp.index((left, right))
				left_shift = 1
				right_shift = 1
				while((len(chunks_tmp) - 1 - start_chunk_tmp) >= right_shift) or (start_chunk_tmp >= left_shift):
					left_dis = None
					right_dis = None

					if left_shift <= start_chunk_tmp:
						left_chunk = chunks_tmp[start_chunk_tmp - left_shift]
						left_dis = left - left_chunk[1]

					if start_chunk_tmp + right_shift <= len(chunks_tmp) - 1:
						right_chunk = chunks_tmp[start_chunk_tmp + right_shift]
						right_dis = right_chunk[0] - right

					if (right_dis is None) or ((left_dis is not None) and (left_dis < right_dis)):
						left = left_chunk[0]
						left_shift += 1
					elif (left_dis is None) or ((right_dis is not None) and (left_dis > right_dis)):
						right = right_chunk[1]
						right_shift += 1
					else:
						left = left_chunk[0]
						right = right_chunk[1]
						left_shift += 1
						right_shift += 1

					if (right - left) >= 18:
						adjusted_lefts.append(left)
						adjusted_rights.append(right)
						break
		adjusted_lefts = np.array(adjusted_lefts)
		adjusted_rights = np.array(adjusted_rights)

		return self.combine_dup(np.array(adjusted_lefts), np.array(adjusted_rights))

	def get_positions(self, npix_col):
		"""确定聚类后图片中字符的位置

		参数
		－－－－
		npix_col: {array-like} [self.width]
			聚类后形成图片每列不为背景的像素个数

		返回值
		－－－－
		(lefts, rights): tuple [2]
			验证码图片中各个字符的左右边界位置信息。
			lefts为左边界，rights为右边界
		"""
		lefts = []
		rights = []
		if npix_col[0] != 0:
			lefts.append(0)
		for i in range(0, self.width - 1):
			if npix_col[i] == 0 and npix_col[i + 1] > 0:
			    lefts.append(i)
			if npix_col[i] > 0 and npix_col[i + 1] == 0:
			    rights.append(i)
		if npix_col[self.width-1] != 0:
			rights.append(self.width-1)
		lefts_tmp = set(copy.copy(lefts))
		rights_tmp = set(copy.copy(rights))
		lefts = np.array(sorted(list(lefts_tmp - rights_tmp)))
		rights = np.array(sorted(list(rights_tmp - lefts_tmp)))

		return self.adjust_positions(npix_col, lefts, rights)

	def seg_standardize(self, img_matrix):
		"""对切割后的图像进行标准化使其具有相同的大小
		"""
		n_rows, n_cols = img_matrix.shape

		if n_cols < self.char_width:
			ncol_add = self.char_width - n_cols
			ncol_add_left = int(ncol_add / 2)
			ncol_add_right = int(ncol_add - ncol_add_left)
			img_matrix = np.concatenate((np.zeros((n_rows, ncol_add_left)), img_matrix, 
				np.zeros((n_rows, ncol_add_right))), axis=1)
		if n_cols > self.char_width:
			ncol_red = n_cols - self.char_width
			ncol_red_left = int(ncol_red / 2)
			ncol_red_right = int(ncol_red - ncol_red_left)
			img_matrix = img_matrix[:, ncol_red_left:(n_cols-ncol_red_right)]
		if n_rows < self.char_height:
			nrow_add = self.char_height - n_rows
			nrow_add_top = int(nrow_add / 2)
			nrow_add_bottom = int(nrow_add - nrow_add_top)
			img_matrix = np.concatenate((np.zeros((nrow_add_top, self.char_width)), img_matrix,
				np.zeros((nrow_add_bottom, self.char_width))), axis=0)
		if n_rows > self.char_height:
			nrow_red = n_rows - self.char_height
			nrow_red_top = int(nrow_red / 2)
			nrow_red_bottom = int(nrow_red - nrow_red_top)
			img_matrix = img_matrix[nrow_red_top:(n_rows-nrow_red_bottom), :]
		return img_matrix.flatten()

	def get_segment_data(self, positions, image_matrix):
		"""根据提供的切割位置获得所切割的图像
		"""
		image_data = []

		left, right = positions
		left = int(left)
		right = int(right)
		image_data = image_matrix[:, left:(right+1)]
		rsum_g2 = np.where(image_data.sum(axis=1) >= 2)[0]
		if len(rsum_g2) < 5:
			return None
		top = rsum_g2[0]
		bottom = rsum_g2[-1]
		image_data = image_data[top:(bottom+1), :]

		return self.seg_standardize(image_data)

	def get_segments(self, position_matrix, image_vectors):
		"""得到破解需要的切割后字符像素矩阵
		"""
		positions = position_matrix[np.argsort(position_matrix[:,0]), :]
		image_matrices = []
		for i in range(image_vectors.shape[0]):
			image_matrices.append(image_vectors[i, :].reshape(self.height, self.width))

		i = 0
		nr = 0
		seg_vecs = []
		nchars = positions.shape[0]
		while nr < nchars:
			seg_vec = self.get_segment_data(positions[nr, 0:2], image_matrices[int(positions[nr, 2])])
			if seg_vec is not None:
				seg_vecs.append(seg_vec)
			nr += 1
			i += 1
			if i > 2:
				break
		return np.array(seg_vecs)

	def segment(self, image_path, check=False):
		"""对聚类后的图片分割出其中的字符

		参数
		－－－－
		image_path: str
			验证码图片的绝对路径
		check: bool
			是否检查分割后的验证码图片

		返回值
		－－－－
		captcha_chars
			验证码图片中所有字符的像素矩阵
		"""
		captcha_chars = []
		image_vectors, col_npixs = self.clustering_captcha(image_path, check)

		n_img = image_vectors.shape[0]
		positions = np.zeros((1, 3))
		for i in range(n_img):
			image_col = col_npixs[i, :]
			lefts_tmp, rights_tmp = self.get_positions(image_col)
			n_lefts_tmp = len(lefts_tmp)
			if n_lefts_tmp != 0:
				rec_tmp = np.concatenate((lefts_tmp.reshape(-1, 1), rights_tmp.reshape(-1, 1), 
					np.repeat(i, n_lefts_tmp).reshape(-1, 1)), axis=1)
				positions = np.vstack((positions, rec_tmp))
		positions = np.delete(positions, (0), axis=0)
		seg_vectors = self.get_segments(positions, image_vectors)

		if seg_vectors.shape[0] != 3:

			return None

		if check:
			if not self.checking_path:
				self.checking_path = os.path.join(self.training_images_path, 'checking')
			if not os.path.isdir(self.checking_path):
				os.mkdir(self.checking_path)

			segments_path = os.path.join(self.checking_path, 'segments')
			segments_pos1_path = os.path.join(segments_path, 'pos_1')
			segments_pos2_path = os.path.join(segments_path, 'pos_2')
			segments_pos3_path = os.path.join(segments_path, 'pos_3')
			if not os.path.isdir(segments_path):
				os.mkdir(segments_path)
			if not os.path.isdir(segments_pos1_path):
				os.mkdir(segments_pos1_path)
			if not os.path.isdir(segments_pos2_path):
				os.mkdir(segments_pos2_path)
			if not os.path.isdir(segments_pos3_path):
				os.mkdir(segments_pos3_path)

			img_name = os.path.split(image_path)[1].split('.')[0]
			for i in range(3):
				new_img_name = os.path.join(segments_path, 'pos_' + str(i+1), 
					img_name + '.jpg')
				im_new = Image.new('1', (self.char_width, self.char_height))
				im_new.putdata(seg_vectors[i, :])
				im_new.save(new_img_name)

		return seg_vectors

	def form_training_set(self, labels_path):
		"""形成建模所需数据集合

		参数
		－－－－
		labels_path: str
			标签所在文件

		返回值
		－－－－
		x_train_num: {array-like}
			数字模型的特征矩阵
		y_train_num: {array-like}
			数字模型的目标矩阵
		x_train_sym: {array-like}
			符号模型的特征矩阵
		y_train_sym: {array-like}
			符号模型的目标矩阵
		"""
		data = pd.read_csv(labels_path, sep=';', header=0)

		x_train_num = []
		y_train_num = []
		x_train_sym = []
		y_train_sym = []

		for idx in data.index:
			file = data.ix[idx, 0]
			file_path = os.path.join(self.training_images_path, file)
			seg_vectors = self.segment(file_path)

			if seg_vectors is not None:
				pos_notnull = list(data.ix[idx, 1:].notnull())
				labels = list(data.ix[idx, 1:])
				for i in range(3):
					if i != 1:
						if pos_notnull[i]:
							x_train_num.append(seg_vectors[i,:])
							y_train_num.append(labels[i])
					else:
						if pos_notnull[i]:
							x_train_sym.append(seg_vectors[i, :])
							y_train_sym.append(labels[i])
		return (np.array(x_train_num, dtype=float), np.array(y_train_num, dtype=float), 
			np.array(x_train_sym, dtype=float), np.array(y_train_sym, dtype=float))

	def choose_model(self, x_num, y_num, x_sym, y_sym):
		"""用cross validation进行模型筛选
		"""
		kfold_num = StratifiedKFold(y=y_num, n_folds=5, shuffle=True)
		kfold_sym = StratifiedKFold(y=y_sym, n_folds=5, shuffle=True)
		accs_num = []
		for k, (train, test) in enumerate(kfold_num):
			x_train_num = x_num[train, :]
			y_train_num = y_num[train]
			x_test_num = x_num[test, :]
			y_test_num = y_num[test]

			pca = PCA(n_components=0.85)
			svm = SVC()
			clf = Pipeline(steps=[('pca', pca), ('svm', svm)])
			clf.fit(x_train_num, y_train_num)
			acc = clf.score(x_test_num, y_test_num)
			accs_num.append(acc)
			print u"数字模型第%s组的精确度为%.6f" % ((k+1), acc)
		print u"数字模型总精确度为%.6f" % np.mean(np.array(accs_num))

		accs_sym = []
		for k, (train, test) in enumerate(kfold_sym):
			x_train_sym = x_sym[train, :]
			y_train_sym = y_sym[train]
			x_test_sym = x_sym[test, :]
			y_test_sym = y_sym[test]

			pca = PCA(n_components=0.85)
			svm = SVC()
			knn = KNeighborsClassifier()
			clf = Pipeline(steps=[('pca', pca), ('svm', svm)])
			clf.fit(x_train_sym, y_train_sym)
			acc = clf.score(x_test_sym, y_test_sym)
			accs_sym.append(acc)
			print u"符号模型第%s组的精确度为%.6f" %((k+1), acc)
		print u"符号模型总精度为%.6f" % np.mean(np.array(accs_sym))
		return True

	def _calculate(self, num_left, operator, num_right):
		"""对验证码的预测结果返回计算值
		"""
		if operator == 1:
			return (num_left + num_right)
		elif operator == 2:
			return (num_left - num_right)
		elif operator == 3:
			return (num_left * num_right)
		else:
			if num_right == 0:
				return None
			else:
				return (num_left / num_right)

	def build_model(self, labels_path):
		"""训练并输出数字和符号模型

		参数
		－－－－
		labels_path: str
			标签路径

		返回值
		－－－－
		True: 
		"""
		if not os.path.isdir(self.model_path):
			os.mkdir(self.model_path)

		if not os.path.isdir(self.model_path + '/num'):
			os.mkdir(self.model_path + '/num')

		if not os.path.isdir(self.model_path + '/sym'):
			os.mkdir(self.model_path + '/sym')

		if not os.path.isdir(self.training_images_path):
			exit(1)

		date = datetime.datetime.now()
		x_num, y_num, x_sym, y_sym = self.form_training_set(labels_path)
		print u"形成训练数据集用时：%s" % (datetime.datetime.now() - date)
		print u"开始训练模型..."

		pca = PCA(n_components=0.85)
		svm = SVC()
		clf = Pipeline(steps=[('pca', pca), ('svm', svm)])
		
		num_model = clf.fit(x_num, y_num)
		joblib.dump(num_model, self.model_path + "/num/model.m")
		sym_model = clf.fit(x_sym, y_sym)
		joblib.dump(sym_model, self.model_path + "/sym/model.m")

		return True

	def predict_result(self, image_path):
		"""利用模型对验证码图片进行破解

		参数
		－－－－
		image_path: str
			验证码图片的绝对路径

		返回值
		－－－－
		(left_num, operator, right_num): tuple(int, int, int)
			对验证码图片中的字符进行识别的结果
		cal_result: int
			验证破解后的计算值
		"""
		if self.num_model is None:
			if os.path.isfile(self.num_model_path):
				self.num_model = joblib.load(self.num_model_path)
			else:
				raise IOError
		if self.sym_model is None:
			if os.path.isfile(self.sym_model_path):
				self.sym_model = joblib.load(self.sym_model_path)
			else:
				raise IOError

		try:
			seg_vectors = self.segment(image_path)
		except:
			return None

		if seg_vectors is None:
			return None

		nums = self.num_model.predict(seg_vectors[[0, 2], :])
		operator = int(self.sym_model.predict(seg_vectors[1, :].reshape(1, -1))[0])
		left_num = int(nums[0])
		right_num = int(nums[1])

		return (left_num, operator, right_num), self._calculate(left_num, operator, right_num)
