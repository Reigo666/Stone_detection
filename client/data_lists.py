from collections import deque
import threading

# 传图
class DataDictDeque(threading.Thread):
    data_list = deque(maxlen=60)

# 保存结果
class DatabaseDictDeque(threading.Thread):
    db_list = deque(maxlen=60)

