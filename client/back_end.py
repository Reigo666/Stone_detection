import os
import time
import PIL
import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtCore
from config import my_config
import client.ezlog as ezlog
from client.segment.seg_model import SegModel
from client.web import DataDictDeque
from client.web import app
from client.data_lists import DatabaseDictDeque
import pymysql
from pymodbus.client import ModbusTcpClient
from pymodbus.payload import BinaryPayloadBuilder


class Worker(QtCore.QThread):
    # 日志信号
    log_signal = QtCore.pyqtSignal(str)
    # 分割信号
    segmentation_signal = QtCore.pyqtSignal(Image.Image, dict)

    def __init__(self):
        super(Worker, self).__init__()
        self.start_work = False
        # 背景图像，与背景图像相似度达到一定程度则进行后续实例分割
        # self.background_image = cv2.imread(my_config.config['background_image'], cv2.IMREAD_GRAYSCALE)
        self.init_model()

    def init_model(self):
        root = os.getcwd()
        # model_config = f'D:\jpf\mine_terminal\client\segment\model_config\model\mask_rcnn_r50_fpn_v3_c.py'
        # model_checkpoint = f'D:\jpf\mine_terminal\client\segment\model_config\model\mask_rcnn_r50_fpn_v3_c.pth'
        model_config = f'D:\\jpf\\mine_terminal\\client\\segment\\model_config\\model\\cascade_mask_rcnn_v2c_bt.py'
        model_checkpoint = f'D:\\jpf\\mine_terminal\\client\\segment\\model_config\\model\\cascade_mask_rcnn_v2c_bt.pth'
        self.model = SegModel(model_config, model_checkpoint)

    def read_data(self):
        data = None
        try:
            data = DataDictDeque.data_list.popleft()
            return data
        except IndexError:
            self.log_signal.emit(ezlog.new_info('数据缓冲队列为空'))
        return None

    def segment(self, save=True):
        try:
            data = self.read_data()
            if data is None:
                time.sleep(1)
                return

            image_raw = data['image']
            image = np.array(image_raw).reshape((1000, 1000, 3))
            self.log_signal.emit(ezlog.new_info('读取到一幅图像'))

            result = self.model.inference_detector(image)
            pred_result = self.model.deal_result(image, result, score_thr=0.55)
            predict_total = sum(pred_result['predict_nums'])
            if predict_total < 100:
                print('[Worker] [segment] error: 未检测到足够的目标')
                return
            predict_percentage = pred_result['predict_nums'] / predict_total
            predict_roundness_percentage = pred_result['roundness_sum'] / predict_total
            image_c = pred_result['img']
            result_dict = dict(
                predict_nums=pred_result['predict_nums'],
                predict_percentage=predict_percentage,
                predict_total=predict_total,
                predict_roundness_percentage=predict_roundness_percentage,
                total_length=pred_result['total_length'],
                max_length=pred_result['max_length']
            )
            self.segmentation_signal.emit(Image.fromarray(image_c).resize((800, 800)), result_dict)
            image_info = {'image_raw': image,
                          'image_color': image_c,
                          'image_sid': data['sharding_id'],
                          'image_time': data['time'],
                          'predict_nums': pred_result['predict_nums'],
                          'roundness_sum': pred_result['roundness_sum']}
            # 是否保存结果
            if save:
                DatabaseDictDeque.db_list.append(image_info)
        except Exception as err:
            print('[Worker] [segment] error:{}'.format(err))

    # 鉴别图像是否有效
    def validate_image(self, image):
        if image is None:
            return False
        return True
        # image1 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        # sim = metrics.structural_similarity(self.background_image, image1)
        # if sim > 0.5:
        #     return True
        # return False

    # 线程主流程
    def run(self):
        # 正式执行
        while True:
            if not self.start_work:
                time.sleep(1)
            else:
                self.segment()


# 本地测试工作线程
class LocalWorker(QtCore.QThread):
    segmentation_signal = QtCore.pyqtSignal(Image.Image, dict)

    def __init__(self):
        super(LocalWorker, self).__init__()
        self.start_work = False
        self.image_list = list()
        self.idx = 0
        self.get_img_list_flag = False
        self.image_path = None
        self.init_model()

    def init_model(self):
        model_config = f'D:\\jpf\\mine_terminal\\client\\segment\\model_config\\model\\cascade_mask_rcnn_v2c_bt.py'
        model_checkpoint = f'D:\\jpf\\mine_terminal\\client\\segment\\model_config\\model\\cascade_mask_rcnn_v2c_bt.pth'
        self.model = SegModel(model_config, model_checkpoint)

    def update_image_list(self, path):
        self.image_path = path
        self.get_img_list_flag = True

    def _get_img_list(self, path):
        self.image_list.clear()
        self.idx = 0
        try:
            dir_or_file = os.listdir(path)
            for file in dir_or_file:
                if file.endswith(('.jpg', '.JPG')):
                    file_path = os.path.join(self.image_path, file)
                    self.image_list.append(file_path)
        except Exception as err:
            print('[LocalWorker] [_get_img_list] error:{}'.format(err))

    def run(self):
        while True:
            area = np.zeros(4)
            r_area = 0
            if self.start_work and self.idx < len(self.image_list):
                try:
                    result = self.model.inference_detector(self.image_list[self.idx])
                    pred_result = self.model.deal_result(self.image_list[self.idx], result, score_thr=0.55)
                    predict_total = sum(pred_result['predict_nums'])
                    predict_percentage = pred_result['predict_nums'] / predict_total
                    area += pred_result['predict_area']
                    r_area += pred_result['roundness_area']
                    result_dict = dict(
                        predict_nums=pred_result['predict_nums'],
                        predict_percentage=predict_percentage,
                        predict_total=predict_total,
                    )
                    self.segmentation_signal.emit(Image.fromarray(pred_result['img']).resize((800, 800)), result_dict)
                except Exception as err:
                    print('[LocalWorker] [run] error:{}'.format(err))
                self.idx += 1
                if self.idx % 300 == 0:
                    total_area = sum(area)
                    area_percentage = area / total_area
                    print('粒径分布：{}，针片状：{}'.format(area_percentage, r_area / total_area))
                    area = np.zeros(4)
                    r_area = 0
                if self.idx >= len(self.image_list):
                    self.idx = 0

            if self.get_img_list_flag:
                self._get_img_list(self.image_path)
                self.get_img_list_flag = False

            time.sleep(1)


class WebServer(QtCore.QThread):
    app = app

    def run(self) -> None:
        app.run(host='0.0.0.0', port=5000)


class SaveWorker(QtCore.QThread):
    # 保存照片和结果
    def save_result(self, image_info):
        time_str = time.strptime(image_info['image_time'], '%Y-%m-%d %H:%M:%S')
        today = time.strftime('%Y%m%d', time_str)
        now = time.strftime('%Y%m%d%H%M%S', time_str)

        save_path = my_config.config['product_info']['save_path']
        path_data = r'{}/{}'.format(save_path, today)
        if not os.path.exists(path_data):
            os.mkdir(path_data)
        if not os.path.exists(path_data + '/raw'):
            os.mkdir(path_data + '/raw')
        if not os.path.exists(path_data + '/done'):
            os.mkdir(path_data + '/done')

        image_raw_path = r'{}/raw//aggregate_{}_{}_{}.jpg'.format(path_data, now, image_info['image_sid'], my_config.config['product_info']['id'])
        image_done_path = r'{}/done/aggregate_{}_{}_{}.jpg'.format(path_data, now, image_info['image_sid'], my_config.config['product_info']['id'])
        image_raw = PIL.Image.fromarray(np.uint8(image_info['image_raw']))
        image_raw.save(image_raw_path)
        image_done = PIL.Image.fromarray(np.uint8(image_info['image_color']))
        image_done.save(image_done_path)
        db, cursor = None, None
        try:
            db = pymysql.connect(host=my_config.config['database']['host'],
                                 port=my_config.config['database']['port'],
                                 user=my_config.config['database']['user'],
                                 password=my_config.config['database']['password'],
                                 database='aggregate')
            cursor = db.cursor()
            param = (image_raw_path,
                     int(my_config.config['product_info']['id']),
                     int(image_info['image_sid']),
                     int(image_info['predict_nums'][0]),
                     int(image_info['predict_nums'][1]),
                     int(image_info['predict_nums'][2]),
                     int(image_info['predict_nums'][3]),
                     image_info['roundness_sum'],
                     image_info['image_time'])
            cursor.execute('insert into running_info values(0, %s, %s, %s, %s, %s, %s, %s, %s, %s)', param)
            db.commit()
        except Exception as err:
            print('[SaveWorker] [save_result] error:{}'.format(err))
        finally:
            if cursor is not None:
                cursor.close()
            if db is not None:
                db.close()

    def send2dcs(self, image_info):
        rank1 = image_info['predict_nums'][0]
        rank2 = image_info['predict_nums'][0]
        rank3 = image_info['predict_nums'][0]
        rank4 = image_info['predict_nums'][0]
        nums = rank1 + rank2 + rank3 + rank4
        rank1p = round(100 * rank1 / nums, 2) * 100
        rank2p = round(100 * rank2 / nums, 2) * 100
        rank3p = round(100 * rank3 / nums, 2) * 100
        rank4p = round(100 * rank4 / nums, 2) * 100
        max_length = image_info['max_length']
        avg_length = round(image_info['total_length'] / nums)
        roundness_per = round(100 * image_info['roundness_sum'] / nums, 2) * 100
        try:
            client = ModbusTcpClient(my_config.config['dcs_ip'], port=502)
            if client.connect():
                builder = BinaryPayloadBuilder()
                builder.add_16bit_int(rank1p)
                builder.add_16bit_int(rank2p)
                builder.add_16bit_int(rank3p)
                builder.add_16bit_int(rank4p)
                builder.add_16bit_int(nums)
                builder.add_16bit_int(roundness_per)
                builder.add_16bit_int(max_length)
                builder.add_16bit_int(avg_length)
                payload = builder.build()
                res = client.write_registers(450, payload, skip_encode=True)
        except Exception as err:
            print('[SaveWorker] [send2dcs] error:{}'.format(err))


    def run(self) -> None:
        while True:
            try:
                data = DatabaseDictDeque.db_list.popleft()
                if data is None:
                    time.sleep(2)
                else:
                    self.save_result(data)
            except Exception as err:
                # print('[SaveWorker] [run] error:{}'.format(err))
                time.sleep(2)


class DbOpt:
    @staticmethod
    def query_by_time(start, end, offset, row):
        db, cursor = None, None
        try:
            db = pymysql.connect(host=my_config.config['database']['host'],
                                 port=my_config.config['database']['port'],
                                 user=my_config.config['database']['user'],
                                 password=my_config.config['database']['password'],
                                 database='aggregate')
            cursor = db.cursor()
            sql = 'select * from running_info where create_time between %s and %s limit %s, %s'
            cursor.execute(sql, (start, end, offset, row))
            results = cursor.fetchall()
            return results
        except Exception as err:
            print('[QueryDB] [query_by_time] error:{}'.format(err))
            return None
        finally:
            if cursor is not None:
                cursor.close()
            if db is not None:
                db.close()


if __name__ == '__main__':
    try:
        client = ModbusTcpClient('192.168.16.133', port=502)
        if client.connect():
            builder = BinaryPayloadBuilder()
            builder.add_16bit_int(2500)
            builder.add_16bit_int(2500)
            builder.add_16bit_int(2500)
            builder.add_16bit_int(2500)
            builder.add_16bit_int(100)
            builder.add_16bit_int(1000)
            builder.add_16bit_int(25)
            builder.add_16bit_int(20)
            payload = builder.build()
            res = client.write_registers(450, payload, skip_encode=True)
    except Exception as err:
        print('[SaveWorker] [send2dcs] error:{}'.format(err))

