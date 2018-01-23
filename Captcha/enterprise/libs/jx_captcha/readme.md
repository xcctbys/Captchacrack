## 功能
主要对江西验证码识别并自动计算出结果

当前文件夹包括
| captcha_jiangxi.py
| __init__.py
| number
	| number_model_jx
	| number_model_jx_{01-15}.py
| symbol
	| symbol_model_jx_{01-15}.py

其中目录number和symbol是带有数字验证码识别和带有汉字验证码识别的模型

captcha_jiangxi.py 中自定义类CAPTCHA_JX，用于对江西验证码进行识别

## 使用方法

### 如果number目录为空，默认不为空，已经训练好模型
	如果为空，则需要对数字验证码进行训练建模
	```python
		captcha_jx = CAPTCHA_JX(img_dir=number_img_dir, label_path=number_label_path)
		captcha_jx.build_model(captcha_type='number')
	```
	其中number_img_dir为数字验证码图片目录，可以为list和字符串目录
	number_label_path为数字验证码标签文件路径，为csv格式
	build_model()需要指定验证码类型为number，否则自动识别的准确率相对较低

### 如果symbol目录为空，默认不为空，已经训练好模型
	如果为空，则需要对带有汉字验证码进行训练建模
	```python
		captcha_jx = CAPTCHA_JX(img_dir=symbol_img_dir, label_path=symbol_label_dir)
		captcha_jx.build_model(captcha_type='symbol')
	```
	其中symbol_img_dir为带有汉字验证码图片目录，可以为list和字符串目录
	symbol_label_dir为带有汉字验证码标签文件路径，为csv格式
	build_model()需要指定验证码类型为symbol，否则自动识别的准确率相对较低
	
### 如果number目录和symbol目录都不为空
	```python
	captcha_jx = CAPTCHA_JX()
	result = captcha_jx.predict(img_path)
	```
	其中img_path为江西验证码一个图片，其中会自动识别验证码类型
	result为验证结果，list类型：图片字符串，计算结果
	

	