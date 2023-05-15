import threading
from flask import Flask, request
from flask import jsonify
from collections import deque
import numpy as np
import json
import time
from client.data_lists import DataDictDeque

app = Flask(__name__)


@app.route('/stone', methods=['POST'])
def post_data():
    print('success')
    raw_data = request.get_data()
    data_dict = json.loads(raw_data)
    data_dict['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    DataDictDeque.data_list.append(data_dict)
    # print(data_dict['time'])
    return jsonify(dict(status='ok'))


if __name__ == '__main__':
    app.run()
