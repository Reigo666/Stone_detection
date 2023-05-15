## 4070
password:hitwu
conda env:mypy38

## client
客户端代码
> ### segment
> 分割代码
> > ### model_config/model
> > > pth
> > > 模型配置
> > ### seg_model.py
> > 模型推理 计算粒径圆度
> > ### test_img.py
> > 测试运行时间
> ### back_end.py
> 后端程序 接收图像，调用segment代码，可视化显示结果
>  1. 读取图像线程
>  - 从缓冲队列接收图像 并对图像进行预测分析的线程（Worker）
>  - 或从本地文件夹中读取图像进行处理（LocalWorker）
>  2. 结果保存线程,保存图像到指定路径，保存数据到数据库(SaveWorker)
>  3. 启动web服务器（开放一个web端口接受图像保存到缓冲队列）（WebServer）
>  4. 数据库查询（DbOpt）
> ### data_lists.py
> 两个缓冲队列
> 1. 传图
> 2. 保存结果
> ### ezlog.py
> 写日志
> ### ui.ui
> 画界面
> ### web.py
> 开放一个web端口接受图像保存到缓冲队列
> ### window.py
> 界面逻辑

## mmdet
框架的源文件（不用管） 需要配合mmcd
官网: https://mmdetection.readthedocs.io/zh_CN/latest/model_zoo.html

## config.py
读写配置文件

## config.yml
配置文件

## main.py
入口文件

## main.spec
打包的配置文件
python打包生成exe
用pyinstaller进行打包

## test_web.py
往client发图片,用于测试

