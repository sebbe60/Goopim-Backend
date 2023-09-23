# using SendGrid's Python Library
# https://github.com/sendgrid/sendgrid-python
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

message = Mail(from_email='mrjohnugbor@gmail.com',
               to_emails='mrjohnugbor@gmail.com',
               subject='Test Email',
               html_content='<strong>Test message.</strong>')
try:
    sg = SendGridAPIClient('')
    response = sg.send(message)
    print(response.status_code)
    print(response.body)
    print(response.headers)
except Exception as e:
    print(e)