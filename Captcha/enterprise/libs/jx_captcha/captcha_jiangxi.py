# coding:UTF-8

from PIL import Image, ImageFilter, ImageEnhance
# import matplotlib.pyplot as plt
import numpy as np
import os
import copy
from sklearn.svm import SVC
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')


class CAPTCHA_JX(object):

    number_customized_width = 30
    symbol_customized_width = 40
    customized_width = 30
    img_top = 14
    img_bottom = 44
    mask = 115
    symbol_mask = 120
    number_mask = 115
    number_symbol_mask = 145
    image_label_count = 5
    pixes_data = list()
    number_label_list = [u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"?", u"=", u"+", u"-", u"*"]
    symbol_label_list = [u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"?", u"等", u'于', u"加", u"减", u"乘"]
    pixes_lable = list()
    number_model_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'number/number_model_jx')
    symbol_model_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'symbol/symbol_model_jx')
    suffix_dict = {'symbol': '.jpeg'}
    model = None

    def __init__(self, img_dir=None, label_path=None):
        if img_dir is None:
            self.img_dirs = ['E:/raw/images/Jiangxi']
        elif type(img_dir) == list:
            self.img_dirs = img_dir
        elif type(img_dir) == str:
            self.img_dirs = [img_dir]
        else:
            print(u'img_dir必须是字符串或者list')

        if label_path is None:
            self.label_paths = ['E:/raw/jiangxi_dataset.csv']
        elif type(label_path) == list:
            self.label_paths = label_path
        elif type(label_path) == str:
            self.label_paths = [label_path]
        else:
            print(u'label_path必须是字符串或者list')

    # 判断验证码中是否有汉字
    def check_captcha_type(self, pix_sum):
        index = len(pix_sum) - 1
        for i in range(len(pix_sum)-1, -1, -1):
            if pix_sum[i] > 0:
                index = i
                break
        captcha_type = 'number' if index < self.number_symbol_mask else 'symbol'
        return captcha_type

    # 读取验证码图片并切分
    def read_img(self, img_path, captcha_type=None):
        img = Image.open(img_path)

        img = img.crop((0, self.img_top, img.size[0], self.img_bottom))

        positions, pix_data, captcha_type = self.cut_image_jiangxi(img, captcha_type)

        return img, positions, pix_data, captcha_type

    def full_img_name(self, img_name, captcha_type):
        # 补全图像名（加后缀）
        return img_name if '.' in img_name else (img_name + self.suffix_dict[captcha_type])

    # 从文件夹中加载所有训练样本
    def load_img(self, captcha_type=None):
        # print self.label_paths, self.img_dirs
        for label_path, img_dir in zip(self.label_paths, self.img_dirs):

            file = np.loadtxt(label_path, dtype=str, delimiter=',')[1:, :]
            for image_name, code in file:
                # image_name = 'captcha228.jpeg'

                image_name = self.full_img_name(image_name, captcha_type)
                img_path = img_dir + '/' + image_name
                img, positions, pix_data, captcha_type = self.read_img(img_path, captcha_type)
                if positions is None:
                    continue

                self.pixes_data.extend(self.get_pix_list(pix_data, positions, captcha_type))
                self.get_lable(code, captcha_type)

                # print np.array(self.pixes_lable).shape, np.array(self.pixes_data).shape
                # break
        return captcha_type

    # 根据像素直方图识别图片中字符位置
    def get_position_number(self, width, pix_sum):
        positions = []
        for i in range(1, width - 1):
            if pix_sum[i] == 0 and pix_sum[i + 1] > 0:
                positions.append(i)
            if pix_sum[i] > 0 and pix_sum[i + 1] == 0:
                positions.append(i+1)

        # 对position去重
        if len(positions) > 10 and positions[-1] < 110:
            positions_temp = set(copy.copy(positions))
            positions = list((positions_temp - set([i for i in positions_temp if positions.count(i) > 1])))
            positions = sorted(positions)

        if len(positions) == 10 and positions[-1] < 110:
            self.image_label_count = 5
            return positions
        elif len(positions) == 12 and positions[-1] > 110:
            self.image_label_count = 6
            return positions
        else:
            return None

    def get_position_symbol(self, width, pix_sum):
        positions = []
        for i in range(1, width - 1):
            if pix_sum[i] == 0 and pix_sum[i + 1] > 0:
                positions.append(i)
            if pix_sum[i] > 0 and pix_sum[i + 1] == 0:
                positions.append(i + 1)
        if pix_sum[-1] > 0:
            positions.append(width - 1)

        positions_temp = set(copy.copy(positions))
        positions = list((positions_temp - set([i for i in positions_temp if positions.count(i) > 1])))
        positions = sorted(positions)

        positions = self.adjust_posotions(positions)
        # 验证码中出现的字符数
        self.image_label_count = 6 if positions is None else (len(positions) / 2)
        return positions

    # 对汉字切分出的位置进行调整
    def adjust_posotions(self, positions):
        # print positions
        customized_len_min = [7, 26, 7, 24, 23, 7]
        customized_len_max = [20, 34, 20, 34, 34, 20]

        # 从长度检测切分位置是否合理,默认切分不合理
        is_reasonable = False
        if len(positions) == 12 or len(positions) == 14:
            for i in range(len(positions) / 2):
                diff = positions[2 * i + 1] - positions[2 * i]
                if diff < customized_len_min[i] or diff > customized_len_max[i]:
                    break
                if i == 5:
                    is_reasonable = True
                    break
        # 如果合理则直接退出返回正确的切分位置
        if is_reasonable:
            return positions

        # 切分不正确则进行重新切分和判断
        positions_temp = list()
        start = 0
        while start <= len(positions) - 1:
            # 处理6个字符的验证码
            if len(positions_temp) == 12 and (len(positions) - start) == 2:
                positions_temp.extend(positions[-2:])
                break

            # 确定起步位置
            if len(positions_temp) % 2 == 0:
                positions_temp.append(positions[start])
                start += 1
                continue

            index = len(positions_temp) / 2
            diff = positions[start] - positions_temp[-1]

            if len(positions) > 12:
                if index > len(customized_len_min) - 1:
                    break
                if diff < customized_len_min[index]:
                    start += 1
                    continue
                elif customized_len_max[index] >= diff >= customized_len_min[index]:
                    # if index < 5 and (positions[start + 2] - positions[start + 1]) >= customized_len_min[index + 1]
                    if index < 6:
                        positions_temp.append(positions[start])
                    start += 1
                    continue
                elif diff > customized_len_max[index]:
                    start += 1
                    continue

            else:

                # 如何切分位置数小于10，则自定义切分位置,待升级。。。
                break

        # print positions_temp
        # 最后判断切分是否正确
        if len(positions_temp) == 12 or len(positions_temp) == 14:
            return positions_temp
        else:
            return None

    # 根据字符位置获得相应字符
    def get_lable(self, code, captcha_type):
        if captcha_type == 'number':
            self.pixes_lable.extend([self.number_label_list.index(c) for c in code.strip()])
        else:
            # 对标签中的中文？进行检测并修改为英文?
            code_list = list(code.strip().decode('utf-8'))

            if u'？' in code_list:
                code_list[code_list.index(u'？')] = u'?'
            code = u''.join(code_list)

            self.pixes_lable.extend([self.symbol_label_list.index(c) for c in code])

    def get_pix_list(self, pix_data, positions, captcha_type):
        """
        对对象大小归一
        :param pix_data
        :param positions:
        :param code:
        :return:
        """
        pixes_data = []
        self.customized_width = self.number_customized_width if captcha_type == 'number' else self.symbol_customized_width
        print self.customized_width
        if self.customized_width is not None:
            # print self.image_label_count
            for i in range(self.image_label_count):
                img_left = positions[2 * i]
                img_right = positions[2 * i + 1]

                temp = pix_data[:, img_left: img_right].flatten().tolist()

                if self.customized_width > abs(img_right - img_left):
                    difference = self.customized_width - abs(img_right - img_left)
                    half = difference / 2
                    pixes_data.append([0] * half * (self.img_bottom - self.img_top) + temp +
                                           [0] * half * (self.img_bottom - self.img_top) +
                                           [0] * (difference % 2) * (self.img_bottom - self.img_top)
                                           )

                elif self.customized_width == abs(img_right - img_left):
                    pixes_data.append(temp)
                else:
                    # 自定义归一大小小于某个切分宽度
                    print(u'自定义归一大小小于某个切分宽度!!!')
                    continue
            return pixes_data

    def cut_image_jiangxi(self, img, captcha_type=None):
        # for p in range(self.img_start, self.width, self.customized_width):
        #     if p >= self.img_start + self.customized_width * self.image_label_count:
        #         break
        #     sub_img = self.img.crop((p, 0, p + self.customized_width, self.heght))
        #     sub_img.show()
        # posotion_left = [10, 27, 47, 60, 80, 95]
        # position_right = [26, 45, 62, 77, 92, 111]
        # for l, r in zip(posotion_left, position_right):
        #     sub_img = self.img.crop((l, 0, r, self.heght))
        #     sub_img.show()

        # self.img = self.img.filter(ImageFilter.MedianFilter)

        self.mask = self.symbol_mask if captcha_type == 'symbol' else self.number_mask

        width, height = img.size
        img_b = img.convert("L").load()
        # self.img_b.show()

        pix_sum = list()
        pix_data = list()
        for w in range(width):
            temp = 0
            col_list = []
            for h in range(height):
                temp += 10 if img_b[w, h] < self.mask else 0
                col_list.append(1 if img_b[w, h] < self.mask else 0)
            pix_data.append(col_list)
            pix_sum.append(temp)

        captcha_type = self.check_captcha_type(pix_sum) if captcha_type is None else captcha_type
        positions = self.get_position_number(width, pix_sum) if captcha_type == 'number' else self.get_position_symbol(width, pix_sum)
        # plt.plot(range(self.width), pix_sum, '-r')
        # plt.show()
        return positions, np.array(pix_data).transpose(), captcha_type

    def __same_cal_syb(self, result):
        try:

            if u'等' in result:
                result[result.index(u'等')] = u'='
            if u'于' in result:
                result.remove(u'于')
            if u'加' in result:
                result[result.index(u'加')] = u'+'
            if u'减' in result:
                result[result.index(u'减')] = u'-'
        except Exception as e:
            raise Exception
        return result

    def __caculate(self, result, captcha_type):
        try:
            result = result if captcha_type == 'number' else self.__same_cal_syb(result)
            equal_syb_index = result.index(u"=")
            cal_syb = u""
            if u"+" in result:
                cal_syb_index = result.index(u"+")
                cal_syb = u"+"
            if u"-" in result:
                cal_syb_index = result.index(u"-")
                cal_syb = u"-"
            question_syb_index = result.index(u"?")
            if question_syb_index == 0:
                first_num = int(u"".join(result[cal_syb_index+1: equal_syb_index]))
                second_num = int(u"".join(result[equal_syb_index+1:]))
                if cal_syb == u"+":
                    return second_num - first_num
                if cal_syb == u"-":
                    return second_num + first_num

            elif question_syb_index == len(result)-1:
                first_num = int(u"".join(result[0: cal_syb_index]))
                second_num = int(u"".join(result[cal_syb_index + 1: equal_syb_index]))
                if cal_syb == u"+":
                    return first_num + second_num
                if cal_syb == u"-":
                    return first_num - second_num
            else:
                first_num = int(u"".join(result[0: cal_syb_index]))
                second_num = int(u"".join(result[equal_syb_index + 1:]))
                if cal_syb == u"+":
                    return second_num - first_num
                if cal_syb == u"-":
                    return first_num - second_num
        except Exception:
            return ""
        return ""

    def build_model(self, captcha_type=None):
        captcha_type = self.load_img(captcha_type)
        x = self.pixes_data
        y = self.pixes_lable
        print u'开始建模。。。'
        rbm = BernoulliRBM(
            random_state=0,
            verbose=True, learning_rate=0.02,
            n_iter=400, n_components=650,
            batch_size=12)
        svm = SVC(kernel="linear", tol=5e-14, class_weight="balanced")
        classifier = Pipeline(steps=[("rbm", rbm), ("svm", svm)])
        self.model = classifier.fit(x, y)
        if captcha_type == 'number':
            joblib.dump(self.model, self.number_model_file)
        else:
            joblib.dump(self.model, self.symbol_model_file)

    # 预测
    def predict(self, img_path):
        img, positions, pix_data, captcha_type = self.read_img(img_path)
        print positions, captcha_type
        if positions is None:
            print('图像切分错误！')
            return None
        x = np.array(self.get_pix_list(pix_data, positions, captcha_type))
        if captcha_type == 'number':
            if self.model is None or os.path.isfile(self.number_model_file):
                self.model = joblib.load(self.number_model_file)
            else:
                raise IOError
        elif self.model is None or os.path.isfile(self.symbol_model_file):
                self.model = joblib.load(self.symbol_model_file)
        else:
            raise IOError
        predict_label = list()
        for i in range(x.shape[0]):
            input = x[i, :]
            predict_y = self.model.predict(input)[0]
            if int(predict_y) >= len(self.number_label_list) or int(predict_y) < 0:
                return "", ""
            if captcha_type == 'number':
                predict_label.append(self.number_label_list[predict_y])
            else:
                predict_label.append(self.symbol_label_list[predict_y])

        return u"".join(predict_label), self.__caculate(predict_label, captcha_type)

# number_img_dir = 'E:/raw/images/Jiangxi'
# number_label_path = 'E:/raw/jiangxi_dataset.csv'
# symbol_img_dir = ['E:/plkj-work/jiangxi_catchpa/symbol/left', 'E:/plkj-work/jiangxi_catchpa/symbol/right']
# symbol_label_dir = [d + '/label.csv' for d in symbol_img_dir]

# captcha_jx = CAPTCHA_JX(img_dir=number_img_dir, label_path=number_label_path)
# captcha_jx.build_model(captcha_type='number')
#
# captcha_jx = CAPTCHA_JX(img_dir=symbol_img_dir, label_path=symbol_label_dir)
# captcha_jx.build_model(captcha_type='symbol')

# predit_y = []
# true_y = []
#
# captcha_jx = CAPTCHA_JX()
# file = np.loadtxt("E:/raw/jiangxi_dataset.csv", dtype=str, delimiter=',')[1:, :]
# corr_num = 0
# total_num = 0
# for image_name, code in file:
#     code = code.strip()
#     img_path = 'E:/raw/images/Jiangxi/' + image_name
#     result = captcha_jx.predict(img_path)
#     if result is None:
#         continue
#     if result[0] == code:
#         corr_num += 1
#     print result[0], code, result[0] == code
#     total_num += 1
# print corr_num, total_num







