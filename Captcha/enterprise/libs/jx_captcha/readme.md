## ����
��Ҫ�Խ�����֤��ʶ���Զ���������

��ǰ�ļ��а���
| captcha_jiangxi.py
| __init__.py
| number
	| number_model_jx
	| number_model_jx_{01-15}.py
| symbol
	| symbol_model_jx_{01-15}.py

����Ŀ¼number��symbol�Ǵ���������֤��ʶ��ʹ��к�����֤��ʶ���ģ��

captcha_jiangxi.py ���Զ�����CAPTCHA_JX�����ڶԽ�����֤�����ʶ��

## ʹ�÷���

### ���numberĿ¼Ϊ�գ�Ĭ�ϲ�Ϊ�գ��Ѿ�ѵ����ģ��
	���Ϊ�գ�����Ҫ��������֤�����ѵ����ģ
	```python
		captcha_jx = CAPTCHA_JX(img_dir=number_img_dir, label_path=number_label_path)
		captcha_jx.build_model(captcha_type='number')
	```
	����number_img_dirΪ������֤��ͼƬĿ¼������Ϊlist���ַ���Ŀ¼
	number_label_pathΪ������֤���ǩ�ļ�·����Ϊcsv��ʽ
	build_model()��Ҫָ����֤������Ϊnumber�������Զ�ʶ���׼ȷ����Խϵ�

### ���symbolĿ¼Ϊ�գ�Ĭ�ϲ�Ϊ�գ��Ѿ�ѵ����ģ��
	���Ϊ�գ�����Ҫ�Դ��к�����֤�����ѵ����ģ
	```python
		captcha_jx = CAPTCHA_JX(img_dir=symbol_img_dir, label_path=symbol_label_dir)
		captcha_jx.build_model(captcha_type='symbol')
	```
	����symbol_img_dirΪ���к�����֤��ͼƬĿ¼������Ϊlist���ַ���Ŀ¼
	symbol_label_dirΪ���к�����֤���ǩ�ļ�·����Ϊcsv��ʽ
	build_model()��Ҫָ����֤������Ϊsymbol�������Զ�ʶ���׼ȷ����Խϵ�
	
### ���numberĿ¼��symbolĿ¼����Ϊ��
	```python
	captcha_jx = CAPTCHA_JX()
	result = captcha_jx.predict(img_path)
	```
	����img_pathΪ������֤��һ��ͼƬ�����л��Զ�ʶ����֤������
	resultΪ��֤�����list���ͣ�ͼƬ�ַ�����������
	

	