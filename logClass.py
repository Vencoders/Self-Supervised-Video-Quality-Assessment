import logging
import os.path
import sys
import time
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class loggerClass:
    def __init__(self, name='Mylogger', logLevel='INFO', save=True, save_level=logging.WARNING,
                 log_path='./logs', is_debug=False):
        """
        log消息等级从低到高依次为 DEBUG | INFO | WARNING | ERROR | CRITICAL
        """
        self.log_level = logging.INFO       # stream流 默认log等级为INFO
        self.name = name    # log name
        self.logLevel = logLevel    # stream流 默认log等级为INFO
        self.save = save    # 是否启用 file流
        self.save_level = save_level    # 默认save流 log等级
        self.log_path = log_path    # log路径
        self.is_debug = is_debug    # 是否开启stream流

    def init_logger(self, fh_type=2):

        """
        fh_type: 1 for fileHandler; 2 for RotatingFileHandler; 3 for TimedRotatingFileHandler
        """

        if self.logLevel == 'DEBUG':
            self.log_level = logging.DEBUG
        elif self.logLevel == 'INFO':
            self.log_level = logging.INFO
        elif self.logLevel == 'WARNING':
            self.log_level = logging.WARNING
        elif self.logLevel == 'ERROR':
            self.log_level = logging.ERROR
        elif self.logLevel == 'CRITICAL':
            self.log_level = logging.CRITICAL

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(thread)s - %(message)s')  # log输出的format
        # 当前输出格式为 时间 - log名称 - PID - 内容

        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)

        if self.is_debug:
            logger.addHandler(self.init_streamHd(formatter))  # 启用stream流 handler

        if self.save:

            log_file = os.path.join(self.log_path, self.name + '.log')

            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)


            # 启用file handler
            if fh_type == 1:
                logger.addHandler(self.init_fileHd(formatter, log_file))
            elif fh_type == 2:
                logger.addHandler(self.init_rotatingfileHd(formatter, log_file))
            elif fh_type == 3:
                logger.addHandler(self.init_timerotatingfileHd(formatter, log_file))

        return logger

    def init_streamHd(self, formatter):
        streamHandler = logging.StreamHandler(sys.stdout)
        streamHandler.setLevel(self.log_level)
        streamHandler.setFormatter(formatter)

        return streamHandler

    def init_fileHd(self, formatter, log_file_name='mylog.log'):
        fileHandler = logging.FileHandler(log_file_name)
        fileHandler.setLevel(self.save_level)
        fileHandler.setFormatter(formatter)

        return fileHandler

    def init_rotatingfileHd(self, formatter, log_file_name='mylog.log'):
        fh = RotatingFileHandler(
            filename=log_file_name,
            mode='a',
            maxBytes=10*1024,
            backupCount=7
        )
        fh.setLevel(self.save_level)
        fh.setFormatter(formatter)

        return fh

    def init_timerotatingfileHd(self, formatter, log_file_name='mylog.log', frequency='D', interval=1):
        fh = TimedRotatingFileHandler(
            filename=log_file_name,
            when=frequency,
            interval=interval,
            backupCount=7
        )
        fh.setLevel(self.save_level)
        fh.setFormatter(formatter)

        return fh

# if __name__ == '__main__':
#
#     logger = loggerClass(logLevel='DEBUG', save_level=logging.DEBUG, log_path='./logs').init_logger(2)
#
#     while True:
#         logger.debug('start loop')
#         time.sleep(1)
