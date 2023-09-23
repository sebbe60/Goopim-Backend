import os
import secrets

import flask_security

from app import config




def generate_coupon_code() -> str:
    return '-'.join(''.join(secrets.choice("0123456jhytouyFGHIJKL") for _ in range(4)) for _ in range(4))


def nvl(param1, param2):
    if param1 is not None:
        return param1
    else:
        return param2


def get_cdn_path():
    return config.Config.CDN_DOMAIN if os.environ['FLASK_ENV'] == 'production' else '/'


def is_admin_or_employee():
    return [x for x in flask_security.current_user.roles if x in ['admin', 'employee']]


def is_customer():
    return not [x for x in flask_security.current_user.roles if x in ['admin', 'employee']]



