import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    SECRET_KEY = '123456789'
    SQLALCHEMY_DATABASE_URI = 'postgresql://icagntjypbchsy:b556dc64b17235d552e42a94193940618b7c608836785cd7cd9d941dd8f841af@database-1.cdmhtriqsond.eu-north-1.rds.amazonaws.com:5432/postgres'
# os.getenv("DATABASE_URI",'postgresql://')
    #SQLALCHEMY_DATABASE_URI =    'postgresql://postgres:123456789@localhost:5432/goopim'
    #os.getenv("URI", "sqlite://")
    #os.getenv("DATABASE_URL", "sqlite://")


    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAIL_SENDGRID_API_KEY = '123456789'
    MAIL_DEFAULT_SENDER = 'mrjohnugbor@gmail.com'

    # Flask Security settings
    SECURITY_PASSWORD_SALT = SECRET_KEY
    SECURITY_REGISTERABLE = True
    SECURITY_RECOVERABLE = True
    SECURITY_CHANGEABLE = True
    SECURITY_REGISTER_URL = '/register'
    SECURITY_LOGIN_URL = '/login'
    SECURITY_POST_LOGIN_VIEW = '/'
    SECURITY_LOGOUT_URL = '/logout'
    SECURITY_POST_LOGOUT_VIEW = '/'
    SECURITY_RESET_URL = '/reset'
    SECURITY_CHANGE_URL = '/change'
    SECURITY_USER_IDENTITY_ATTRIBUTES = ['email']

    SECURITY_EMAIL_SUBJECT_REGISTER = 'Registration Confirmation'
    SECURITY_EMAIL_SUBJECT_PASSWORD_NOTICE = 'Your Password has been Reset'
    SECURITY_EMAIL_SUBJECT_PASSWORD_RESET = 'Reset Your Password'
    SECURITY_EMAIL_SUBJECT_PASSWORD_CHANGE_NOTICE = 'Your Password was Changed'
    SECURITY_EMAIL_SUBJECT_CONFIRM = 'Please Confirm Your Email Address'
    SECURITY_EMAIL_PLAINTEXT = False
    SECURITY_EMAIL_HTML = True

