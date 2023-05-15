import time
from PIL import ImageQt
from PyQt5.QtChart import QChart, QLineSeries, QValueAxis, QChartView, QPieSeries
from PyQt5.QtWidgets import QMainWindow, QGraphicsPixmapItem, QGraphicsScene, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QDateTime
from PyQt5.QtGui import QPixmap, QPainter, QColor
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QRegExpValidator
from client.back_end import Worker, LocalWorker, WebServer, SaveWorker
from config import my_config
from datetime import datetime
from client.back_end import DbOpt


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # 加载UI
        self.ui = loadUi('D:\\jpf\\mine_terminal\\client\\ui.ui', self)
        # 用于显示图像的场景
        self.image_scene = QGraphicsScene()
        self.image_from_camera = None
        self.image_scene_local = QGraphicsScene()
        self.image_from_local = None
        # 分布图
        self.line_chart1 = ChartsLine()
        self.line_chart2 = ChartsLineSecond()
        self.line_chart3 = ChartsLine()
        self.pie_chart1 = ChartsPie()
        self.chart_init()
        # 绑定信号，初始化后端
        self.worker = Worker()
        self.worker.log_signal.connect(self.log)
        self.worker.segmentation_signal.connect(self.visual_info)
        self.worker.start()
        self.local_worker = LocalWorker()
        self.local_worker.segmentation_signal.connect(self.visual_local)
        self.local_worker.start()
        self.save_worker = SaveWorker()
        self.save_worker.start()
        # web服务器
        self.web_worker = WebServer()
        self.web_worker.start()
        # 绑定按钮槽函数
        self.main_btn_connect()
        # 加载配置
        self.init_my_config()
        self.init_validator()
        # 初始化运行时周期统计量
        self.init_period_data()

    def init_validator(self):
        doubleValidator = QDoubleValidator(self)
        doubleValidator.setRange(0.01, 360)
        doubleValidator.setNotation(QDoubleValidator.StandardNotation)
        # 设置精度，小数点2位
        doubleValidator.setDecimals(2)
        self.ui.tab3_lineEdit_1.setValidator(doubleValidator)

    def init_my_config(self):
        self.ui.tab1_text_info_1.setText(str(my_config.config['product_info']['id']))
        self.ui.tab1_text_info_2.setText(my_config.config['product_info']['location'])
        self.ui.tab1_text_info_3.setText(str(my_config.config['product_info']['belt_width']))
        self.ui.tab1_text_info_4.setText(str(my_config.config['product_info']['belt_speed']))
        self.tab3_dateTimeEdit_1.setDateTime(QDateTime.currentDateTime().addDays(-1))
        self.tab3_dateTimeEdit_2.setDateTime(QDateTime.currentDateTime())
        # 设置文本最大显示行数
        self.ui.tab1_text_log.setMaximumBlockCount(100)
        self.ui.tab3_text_log.setMaximumBlockCount(100)

    def init_period_data(self):
        self.update_time = datetime.now()
        self.max_length = 0
        self.total_length = 0
        self.total = 0

    def main_btn_connect(self):
        self.ui.tab1_btn_start.clicked.connect(self.start_work)
        self.ui.tab1_btn_end.clicked.connect(self.end_work)
        # self.ui.tab1_btn_init.clicked.connect(self.camera_init)
        self.ui.tab1_btn_clear_log.clicked.connect(self.clear_log)
        self.ui.tab2_btn_start.clicked.connect(self.start_work_local)
        self.ui.tab2_btn_end.clicked.connect(self.end_work_local)
        self.ui.tab2_btn_path.clicked.connect(self.open_dir)
        self.ui.tab3_btn_query.clicked.connect(self.query_db)
        self.ui.tab3_btn_clear.clicked.connect(self.clear_log2)

    def log(self, info):
        self.ui.tab1_text_log.appendPlainText(info)

    def clear_log(self):
        self.ui.tab1_text_log.clear()

    def clear_log2(self):
        self.ui.tab3_text_log.clear()
        self.line_chart3.clear_chart()

    def start_work(self):
        self.worker.start_work = True

    def end_work(self):
        self.worker.start_work = False

    def start_work_local(self):
        self.local_worker.start_work = True

    def end_work_local(self):
        self.local_worker.start_work = False

    def chart_init(self):
        self.ui.tab1_view_line_chart.setChart(self.line_chart1)
        self.ui.tab1_view_line_chart.setRenderHint(QPainter.Antialiasing)
        self.ui.tab1_view_line_chart.setRubberBand(QChartView.RectangleRubberBand)

        self.ui.tab2_view_line_chart.setChart(self.line_chart2)
        self.ui.tab2_view_line_chart.setRenderHint(QPainter.Antialiasing)
        self.ui.tab2_view_line_chart.setRubberBand(QChartView.RectangleRubberBand)

        self.ui.tab3_view_line_chart.setChart(self.line_chart3)
        self.ui.tab3_view_line_chart.setRenderHint(QPainter.Antialiasing)
        self.ui.tab3_view_line_chart.setRubberBand(QChartView.RectangleRubberBand)

        self.ui.tab1_view_pie_chart.setChart(self.pie_chart1)
        self.ui.tab1_view_pie_chart.setRenderHint(QPainter.Antialiasing)
        self.ui.tab1_view_pie_chart.setRubberBand(QChartView.RectangleRubberBand)

    def visual_info(self, image, result_dict):
        self.image_from_camera = ImageQt.ImageQt(image)
        self.visual_image()
        predict_nums = result_dict['predict_nums']
        predict_percentage = result_dict['predict_percentage']
        predict_total = result_dict['predict_total']
        predict_roundness_percentage = result_dict['predict_roundness_percentage']
        self.visual_result(predict_total, predict_percentage, predict_nums, predict_roundness_percentage)
        self.visual_period(result_dict['max_length'], result_dict['total_length'], predict_total)
        self.line_chart1.update_chart(predict_percentage)
        self.pie_chart1.update_chart(predict_percentage)

    def visual_image(self):
        pix = QPixmap.fromImage(self.image_from_camera)
        item = QGraphicsPixmapItem(pix)
        self.image_scene.clear()
        self.image_scene.addItem(item)

        self.ui.tab1_view_img.setScene(self.image_scene)
        self.ui.tab1_view_img.show()

    def visual_result(self, total, percent, nums, predict_roundness_percentage):
        self.ui.tab1_lineEdit_1.setText(str(int(nums[0])))
        self.ui.tab1_lineEdit_2.setText(str(int(nums[1])))
        self.ui.tab1_lineEdit_3.setText(str(int(nums[2])))
        self.ui.tab1_lineEdit_4.setText(str(int(nums[3])))
        self.ui.tab1_lineEdit_5.setText(str(format(round(percent[0], 3), '.1%')))
        self.ui.tab1_lineEdit_6.setText(str(format(round(percent[1], 3), '.1%')))
        self.ui.tab1_lineEdit_7.setText(str(format(round(percent[2], 3), '.1%')))
        self.ui.tab1_lineEdit_8.setText(str(format(round(percent[3], 3), '.1%')))
        self.ui.tab1_lineEdit_9.setText(str(int(total)))
        self.ui.tab1_lineEdit_10.setText(str(format(round(predict_roundness_percentage, 3), '.1%')))

    def visual_period(self, max_length, total_length, total):
        self.max_length = max(self.max_length, max_length)
        self.total_length += total_length
        self.total += total
        now = datetime.now()
        if (now - self.update_time).total_seconds() > 3600 * my_config.config['running_info']['statistical_interval']:
            self.ui.tab1_lineEdit_11.setText(now.strftime('%Y-%m-%d %H:%M:%S'))
            self.ui.tab1_lineEdit_12.setText(str(round(self.max_length, 1)))
            self.ui.tab1_lineEdit_13.setText(str(round(self.total_length / self.total, 1)))
            self.init_period_data()

    def visual_local(self, image, result_dict):
        self.image_from_local = ImageQt.ImageQt(image)
        pix = QPixmap.fromImage(self.image_from_local)
        item = QGraphicsPixmapItem(pix)

        self.image_scene_local.clear()
        self.image_scene_local.addItem(item)
        self.ui.tab2_view_img.setScene(self.image_scene_local)
        self.ui.tab2_view_img.show()

        predict_nums = result_dict['predict_nums']

        self.ui.tab2_lineEdit_1.setText(str(int(predict_nums[0])))
        self.ui.tab2_lineEdit_2.setText(str(int(predict_nums[1])))
        self.ui.tab2_lineEdit_3.setText(str(int(predict_nums[2])))
        self.ui.tab2_lineEdit_4.setText(str(int(predict_nums[3])))
        self.ui.tab2_lineEdit_5.setText(str(int(result_dict['predict_total'])))

        self.line_chart2.update_chart(result_dict['predict_percentage'])

    def open_dir(self):
        path = QFileDialog.getExistingDirectory(self, "请选择文件夹路径", "\\")
        self.ui.tab2_lineEdit_path.setText(path)
        self.local_worker.update_image_list(path)

    def query_db(self):
        self.line_chart3.clear_chart()
        start = self.ui.tab3_dateTimeEdit_1.dateTime().toPyDateTime()
        end = self.ui.tab3_dateTimeEdit_2.dateTime().toPyDateTime()
        interval = float('0.5' if self.ui.tab3_lineEdit_1.text() == '' else self.ui.tab3_lineEdit_1.text())
        self.ui.tab3_text_log.appendPlainText('开始时间：{}，结束时间：{}，间隔{}小时平均统计结果'.format(
            start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S'), interval))
        offset, row = 0, 2
        nums, rank1, rank2, rank3, rank4 = 0, 0, 0, 0, 0
        last_time = start
        do = True
        while do:
            results = DbOpt.query_by_time(start, end, offset, row)
            for result in results:
                nums += 1
                rank1 += result[4]
                rank2 += result[5]
                rank3 += result[6]
                rank4 += result[7]
                if (result[9] - last_time).total_seconds() > 3600 * interval:
                    ranks = [rank1 / nums, rank2 / nums, rank3 / nums, rank4 / nums]
                    rank_sum = sum(ranks)
                    rank_perc = [item / rank_sum for item in ranks]
                    self.line_chart3.update_chart(rank_perc)
                    nums, rank1, rank2, rank3, rank4 = 0, 0, 0, 0, 0
                    self.ui.tab3_text_log.appendPlainText('[{}]-->[{}] 平均粒径分布：[{:.2%}, {:.2%}, {:.2%}, {:.2%}]'.format(
                        last_time.strftime('%Y-%m-%d %H:%M:%S'), result[9].strftime('%Y-%m-%d %H:%M:%S'),
                        rank_perc[0], rank_perc[1], rank_perc[2], rank_perc[3]))
                    last_time = result[9]
            if results is None or len(results) < row:
                do = False
            offset += row


class ChartsLine(QChart):
    def __init__(self, parent=None):
        super(ChartsLine, self).__init__(parent)
        self.window = parent
        self.legend().show()
        self.x_range = 60
        self.counter = 0
        self.series_1 = QLineSeries()
        self.series_1.setName("0-10")
        self.series_1.setBrush(QColor(51, 153, 204))
        self.series_2 = QLineSeries()
        self.series_2.setName("10-21")
        self.series_2.setBrush(QColor(51, 204, 51))
        self.series_3 = QLineSeries()
        self.series_3.setName("21-26.5")
        self.series_3.setBrush(QColor(204, 102, 51))
        self.series_4 = QLineSeries()
        self.series_4.setName("26.5+")
        self.series_4.setBrush(QColor(102, 51, 204))

        # self.val1 = [0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0]
        # self.val2 = [0.30, 0.25, 0, 0, 0, 0, 0, 0, 0, 0]
        # self.val3 = [0.15, 0.25, 0, 0, 0, 0, 0, 0, 0, 0]
        # self.val4 = [0.40, 0.25, 0, 0, 0, 0, 0, 0, 0, 0]
        # for idx in range(len(self.val1)):
        #     self.series_1.append(self.counter, self.val1[idx])
        #     self.series_2.append(self.counter, self.val2[idx])
        #     self.series_3.append(self.counter, self.val3[idx])
        #     self.series_4.append(self.counter, self.val4[idx])
        #     self.counter += 1
        self.series_1.append(self.counter, 0)
        self.series_2.append(self.counter, 0)
        self.series_3.append(self.counter, 0)
        self.series_4.append(self.counter, 0)
        self.counter += 1
        # 坐标轴
        self.x_Aix = QValueAxis()
        self.x_Aix.setRange(0, 60)
        self.x_Aix.setLabelFormat("%d")
        self.x_Aix.setTickCount(7)
        self.x_Aix.setTitleText("T")
        self.y_Aix = QValueAxis()
        self.y_Aix.setRange(0.00, 1.00)
        self.y_Aix.setLabelFormat("%.2f")
        self.y_Aix.setTickCount(11)
        self.y_Aix.setMinorTickCount(1)
        self.y_Aix.setTitleText("RATIO")
        # 画坐标轴
        self.addAxis(self.x_Aix, Qt.AlignBottom)
        self.addAxis(self.y_Aix, Qt.AlignLeft)
        # 画线
        self.addSeries(self.series_1)
        self.addSeries(self.series_2)
        self.addSeries(self.series_3)
        self.addSeries(self.series_4)
        # 把曲线关联到坐标轴
        self.series_1.attachAxis(self.x_Aix)
        self.series_1.attachAxis(self.y_Aix)
        self.series_2.attachAxis(self.x_Aix)
        self.series_2.attachAxis(self.y_Aix)
        self.series_3.attachAxis(self.x_Aix)
        self.series_3.attachAxis(self.y_Aix)
        self.series_4.attachAxis(self.x_Aix)
        self.series_4.attachAxis(self.y_Aix)

    def update_chart(self, result):
        if self.counter <= self.x_range:
            self.series_1.append(self.counter, result[0])
            self.series_2.append(self.counter, result[1])
            self.series_3.append(self.counter, result[2])
            self.series_4.append(self.counter, result[3])
            self.counter += 1
        else:
            points1 = self.series_1.pointsVector()
            points2 = self.series_2.pointsVector()
            points3 = self.series_3.pointsVector()
            points4 = self.series_4.pointsVector()
            for i in range(self.x_range - 1):
                points1[i].setY(points1[i + 1].y())
                points2[i].setY(points2[i + 1].y())
                points3[i].setY(points3[i + 1].y())
                points4[i].setY(points4[i + 1].y())
            points1[-1].setY(result[0])
            points2[-1].setY(result[1])
            points3[-1].setY(result[2])
            points4[-1].setY(result[3])
            self.series_1.replace(points1)
            self.series_2.replace(points2)
            self.series_3.replace(points3)
            self.series_4.replace(points4)

    def clear_chart(self):
        self.counter = 0
        self.series_1.replace([])
        self.series_2.replace([])
        self.series_3.replace([])
        self.series_4.replace([])
        self.series_1.append(self.counter, 0)
        self.series_2.append(self.counter, 0)
        self.series_3.append(self.counter, 0)
        self.series_4.append(self.counter, 0)
        self.counter += 1

class ChartsLineSecond(QChart):
    def __init__(self, parent=None):
        super(ChartsLineSecond, self).__init__(parent)
        self.window = parent
        self.legend().show()
        self.series_1 = QLineSeries()
        # self.series_1.setName("预测值")
        self.series_1.setBrush(QColor(51, 153, 204))
        # self.series_2 = QLineSeries()
        # self.series_2.setName("真实值")
        # self.series_2.setBrush(QColor(51, 204, 51))

        # 坐标轴
        self.x_Aix = QValueAxis()
        self.x_Aix.setRange(1, 4)
        self.x_Aix.setTickCount(4)
        self.x_Aix.setLabelFormat("%d")
        self.x_Aix.setTitleText("粒径等级")
        self.y_Aix = QValueAxis()
        self.y_Aix.setRange(0.00, 1.00)
        self.y_Aix.setLabelFormat("%.2f")
        self.y_Aix.setTickCount(11)
        self.y_Aix.setMinorTickCount(1)
        self.y_Aix.setTitleText("RATIO")
        # 画坐标轴
        self.addAxis(self.x_Aix, Qt.AlignBottom)
        self.addAxis(self.y_Aix, Qt.AlignLeft)
        # 画线
        self.addSeries(self.series_1)
        # self.addSeries(self.series_2)
        # 把曲线关联到坐标轴
        self.series_1.attachAxis(self.x_Aix)
        self.series_1.attachAxis(self.y_Aix)
        # self.series_2.attachAxis(self.x_Aix)
        # self.series_2.attachAxis(self.y_Aix)

    def update_chart(self, result):
        self.series_1.clear()
        self.series_1.append(1, result[0])
        self.series_1.append(2, result[1])
        self.series_1.append(3, result[2])
        self.series_1.append(4, result[3])


class ChartsPie(QChart):
    def __init__(self, parent=None):
        super(ChartsPie, self).__init__(parent)
        self.window = parent
        self.legend().show()

        self.series = QPieSeries()
        self.series.append('0-10', 0.25)
        self.series.append('10-21', 0.25)
        self.series.append('21-26.5', 0.25)
        self.series.append('26.5+', 0.25)

        self.series.setLabelsVisible(True)
        slice_1 = self.series.slices()[0]
        slice_1.setBrush(QColor(51, 153, 204))
        slice_2 = self.series.slices()[1]
        slice_2.setBrush(QColor(51, 204, 51))
        slice_3 = self.series.slices()[2]
        slice_3.setBrush(QColor(204, 102, 51))
        slice_4 = self.series.slices()[3]
        slice_4.setBrush(QColor(102, 51, 204))

        self.addSeries(self.series)

    def update_chart(self, result):
        self.series.clear()
        self.series.append('0-10', result[0])
        self.series.append('10-21', result[1])
        self.series.append('21-26.5', result[2])
        self.series.append('26.5+', result[3])

        self.series.setLabelsVisible(True)
        slice_1 = self.series.slices()[0]
        slice_1.setBrush(QColor(51, 153, 204))
        slice_2 = self.series.slices()[1]
        slice_2.setBrush(QColor(51, 204, 51))
        slice_3 = self.series.slices()[2]
        slice_3.setBrush(QColor(204, 102, 51))
        slice_4 = self.series.slices()[3]
        slice_4.setBrush(QColor(102, 51, 204))
