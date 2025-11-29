import os
import sys
from logging_config import logger
import requests
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def send_custom_robot_group_message(access_token, msg, at_user_ids=None, at_mobiles=None, is_at_all=False,title="【通知】"):
    """
    发送钉钉自定义机器人群消息
    :param title:
    :param access_token: 机器人webhook的access_token
    :param secret: 机器人安全设置的加签secret
    :param msg: 消息内容
    :param at_user_ids: @的用户ID列表
    :param at_mobiles: @的手机号列表
    :param is_at_all: 是否@所有人
    :return: 钉钉API响应
    """

    url = f'https://oapi.dingtalk.com/robot/send?access_token={access_token}'
    formatted_utc_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = {
        "at": {
            "isAtAll": str(is_at_all).lower(),
            "atUserIds": at_user_ids or [],
            "atMobiles": at_mobiles or []
        },
        "markdown": {
            "title": formatted_utc_time + title,
            "text": msg
        },
        "msgtype": "markdown"
    }
    headers = {'Content-Type': 'application/json'}
    resp = requests.post(url, json=body, headers=headers)
    logger.info("钉钉自定义机器人群消息响应：%s", resp.text)
    return resp.json()


def send_to_dingtalk(msg: str,err:False):
    formatted_utc_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    at_user_ids = []
    at_mobiles = []
    send_custom_robot_group_message(
        'f38e4a0b83763311cff9aed9bfc1bb789dafa10c51ed8356649dcba8786feea2',
        formatted_utc_time+"通知\n" + msg,
        at_user_ids=at_user_ids,
        at_mobiles=at_mobiles,
        is_at_all=False,
        title="【错误】" if err else '【通知】'
    )

