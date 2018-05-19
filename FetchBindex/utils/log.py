#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

# 定义handler的输出格式formatter
formatter = logging.Formatter(
    '[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s'
)

logger = logging.getLogger(__name__)
channel = logging.StreamHandler()
logger.setLevel(logging.DEBUG)
logger.addHandler(channel)

channel.setFormatter(formatter)
channel.setFormatter(formatter)
