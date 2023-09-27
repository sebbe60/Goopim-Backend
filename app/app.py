# app.py
import json

from flask import Flask, render_template, request, session, url_for, redirect, flash, abort, \
    Response, jsonify, make_response
from collections import Counter
from datetime import datetime, timedelta, timezone
import uuid
import os, pathlib
#from flask_mail import Mail, Message
from flask_security import login_user, SQLAlchemyUserDatastore, Security, roles_required, user_registered, roles_accepted
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_, func, or_, orm, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Query
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from functools import wraps
from wtforms import Form, StringField
from wtforms.validators import Email
from dotenv import load_dotenv
from flask_jwt_extended import create_access_token
from flask_jwt_extended import get_jwt
from flask_jwt_extended import get_jwt_identity
from flask_jwt_extended import jwt_required
from flask_jwt_extended import JWTManager
from flask_jwt_extended import set_access_cookies
from flask_jwt_extended import unset_jwt_cookies
#from flask_bcrypt import generate_password_hash
from flask_cors import CORS
#from app.db_util import get_current_cart, get_title_from_model_col, get_session_cart, get_current_cart_items,get_current_resale_cart,\
#get_current_resale_cart_items
import openai
import stripe
import nltk
import re
import string
import random
from langchain.embeddings import OpenAIEmbeddings




from pydantic import BaseModel, HttpUrl, validator
from typing import List
import pinecone
import time

from flask_socketio import SocketIO, emit,join_room,leave_room

from flask_login import (
    LoginManager,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from oauthlib.oauth2 import WebApplicationClient
import requests
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow


#from google.auth.transport import requests

import google

#from app import
#from app.db_util import get_current_cart, get_title_from_model_col, get_session_cart, get_current_cart_items,get_current_resale_cart,\
#get_current_resale_cart_items
from config import Config
#from app.config import Config
# from app import config, models
from models.model import *
#from models import models
# import models.model

from enum import Enum
from helpers import add_or_update_user_function, find_recommended_providers_function


# custom Flask class to run some commands before app execution
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):
        # if not self.debug or os.getenv('WERKZEUG_RUN_MAIN') == 'true':
        #     with self.app_context():
        #         get_all_products_cached()
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, load_dotenv=load_dotenv, **options)


app = MyFlaskApp(__name__)
load_dotenv()  # Load environment variables from .env file
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading", manage_session=True,logger=True, engineio_logger=True, flask_app=app)

socketio.init_app(app, cors_allowed_origins="*")

app.config.from_object(Config)

# db = SQLAlchemy()
db.init_app(app)

CORS(app)
#mail= Mail(app)

openai.apikey = os.getenv("OPENAI_API_KEY")
stripe.api_key =os.getenv("STRIPE_KEY")
endpoint_secret = os.getenv("STRIPE_WEBSOCKET_KEY")

nltk.download('punkt')
nltk.download('stopwords')

# class CustomJSONEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, Enum):
#             return obj.name
#         if isinstance(obj, datetime):
#             return obj.isoformat()
#         return super().default(obj)

# app.json = CustomJSONEncoder

# #new config
# app.config["Access-Control-Allow-Headers"]="Content-Type"
#
# # bypass http
# os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
# app.secret_key = os.getenv("SECRET_KEY")
# GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
# client_secrets_file = os.path.join(pathlib.Path(__file__).parent, "client-secret.json")
#algorithm = os.getenv("ALGORITHM")
BACKEND_URL= os.getenv("BACKEND_URL") #"http://127.0.0.1:5000"
FRONTEND_URL= os.getenv("FRONTEND_URL")#"http://127.0.0.1:3000"
#end new config
# Initilize Open AI embedder
embedder = OpenAIEmbeddings(
    model="text-embedding-ada-002", show_progress_bar=True)  # type: ignore

# ------------------ PINECONE SETUP ------------------------
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENVIRONMENT"))

# Set pinecone index name
pinecone_index_name = os.getenv("PINECONE_INDEX")
# Configuration
GOOGLE_CLIENT_ID =os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_DISCOVERY_URL = (
    "https://accounts.google.com/.well-known/openid-configuration"
)
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = os.getenv("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config['SENDGRID_API_KEY'] = os.getenv("SENDGRID_API_KEY")
# Configuration for file uploads
UPLOAD_FOLDER = '/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_URL_PATH'] = '/static'
# Set the Flask-Security configuration options
# app.config['SECURITY_PASSWORD_HASH'] = 'bcrypt'
# app.config['SECURITY_PASSWORD_SALT'] = 'my-app-salt'
app.config['SECURITY_CONFIRMABLE'] = True
app.config['SECURITY_REGISTERABLE'] = True
app.config['SECURITY_SEND_REGISTER_EMAIL'] = True
app.config['SECURITY_EMAIL_SENDER'] = 'noreply@example.com'
app.config["JWT_COOKIE_SECURE"] = False
app.config["JWT_TOKEN_LOCATION"] = ["headers"]
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")  # Change this in your code!
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)
app.config['JSON_ENCODER'] = CustomJSONEncoder
# Configure Cloudinary with your account credentialsrr


jwt = JWTManager(app)
#mail = Mail(app)


logd = app.logger.debug
# log = app.logger.info
logw = app.logger.warning
loge = app.logger.error
logc = app.logger.critical
# Check if index does not exist, then create if required.
if pinecone_index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=pinecone_index_name,
        dimension=1536,
        metric="euclidean"
    )

# Wait for index to be initialized
while not pinecone.describe_index(pinecone_index_name).status['ready']:
    time.sleep(1)

# Connect to index
index = pinecone.Index(pinecone_index_name)

# OAuth 2 client setup
# client = WebApplicationClient(GOOGLE_CLIENT_ID)
def log(msg):
    print(f"{datetime.now().strftime('%d/%m/%y %H:%M:%S.%f')}: {msg}")


def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        token = None
        if 'x-access-tokens' in request.headers:
            token = request.headers['x-access-tokens']

        if not token:
            return jsonify({'message': 'a valid token is missing'})
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = Users.query.filter_by(public_id=data['public_id']).first()
        except:
            return jsonify({'message': 'token is invalid'})

        return f(current_user, *args, **kwargs)

    return decorator
def check_email_exists_already(email):


    # Check if the original goopim_username already exists in the Users table
    existing_user = Users.query.filter_by(email=email).first() or None

    if existing_user:
        return True
    return False
def make_unique_username(goopim_username):
    original_username = goopim_username
    counter = 1

    # Check if the original goopim_username already exists in the Users table
    existing_user = Users.query.filter_by(goopim_username=goopim_username).first()

    # If the username already exists, append a number at the end until it becomes unique
    while existing_user:
        goopim_username = f"{original_username}{counter}"
        counter += 1
        existing_user = Users.query.filter_by(goopim_username=goopim_username).first()

    return goopim_username
def register_user_on_cometchat(uuid,name,email,link):
    url = "https://2399887cdeccbaac.api-eu.cometchat.io/v3/users"

    payload = {
        "metadata": {"@private": {
            "email": email,
            "contactNumber": ""
        }},
        "uid": uuid,
        "name": name,
        "link": link,
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "apikey": "698f477cf5dfc1c264d65ebd9605fecdf458f6da"
    }

    response = requests.post(url, json=payload, headers=headers)

    #return (response.text)
def update_user_profile_image_cometchat(uuid, profile_url):
    url = f'https://2399887cdeccbaac.api-eu.cometchat.io/v3/users/{uuid}'

    payload = {
        "avatar": profile_url,

    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "apikey": "698f477cf5dfc1c264d65ebd9605fecdf458f6da"
    }

    response = requests.put(url, json=payload, headers=headers)


@app.route('/register', methods=['POST'])
def signup_user():
    data = request.get_json()
    hashed_password = generate_password_hash(data['password'], method='sha256')
    goopim_username = make_unique_username(data['username'])
    user_email_exists_already = check_email_exists_already(data['email'])

    if user_email_exists_already:
        return jsonify({'message': 'Email is already registered!'})

    new_user = Users(public_id=str(uuid.uuid4()), first_name=data['first_name'], last_name=data['last_name'], password=hashed_password, email=data['email'],
                     is_goopim_admin=False, is_user=data['is_user'], is_provider=data['is_provider'], goopim_username=goopim_username)
    db.session.add(new_user)


    db.session.commit()
    profile_url_link = f'{FRONTEND_URL}/u/{new_user.goopim_username}'
    print(f'{new_user}')
    register_user_on_cometchat(new_user.public_id,new_user.first_name,new_user.email,profile_url_link)
    send_verification_request(new_user.email)
    payment_account = MyPaymentAccount(user=new_user, balance=0.0)
    db.session.add(payment_account)
    db.session.commit()
    return jsonify({'message': 'registered successfully'})
@app.route('/adminregister', methods=['POST'])
@jwt_required()
def signup_adminuser():
    is_admin = get_jwt_identity()['id']
    if not  is_admin.is_goopim_admin:
        return jsonify({'message': 'Not authorized'}), 401
    data = request.get_json()
    hashed_password = generate_password_hash(data['password'], method='sha256')

    new_user = Users(public_id=str(uuid.uuid4()), first_name=data['first_name'], last_name=data['last_name'], password=hashed_password, email=data['email'],
                     is_goopim_admin=True, is_user=False, is_provider=False)
    send_verification_request(new_user.email)
    db.session.add(new_user)
    db.session.commit()
    payment_account = MyPaymentAccount(user=new_user, balance=0.0)
    db.session.add(payment_account)
    db.session.commit()
    return jsonify({'message': 'registered successfully'})
#data['is_user'], is_provider=data['is_provider']
@app.route('/login', methods=['POST'])

def login_user():
    email = request.json.get("email", None)
    password = request.json.get("password", None)
    user = Users.query.filter_by(email=email).first()
    if not user:

        return jsonify({ "message":"Wrong email or password"}),401


    user = Users.query.filter_by(email=email).first()
    if check_password_hash(user.password, password):
        # token = jwt.encode({'public_id': user.public_id, 'exp': datetime.utcnow() + timedelta(hours=700)},
        #                    app.config['SECRET_KEY'], "HS256")
        #print(user)

        user_dict = user.to_dict()

        token  = create_access_token(identity=user_dict)

        return jsonify({"message":"logged succesfull","token": token,"userId":user.id,"freelancer":user.is_provider,"employer":user.is_user,"mincon":user.is_goopim_admin,"token2":user.public_id}),200

    return jsonify({"message":"login required"}),401


@app.route('/adminlogin', methods=['POST'])

def login_admin():
    email = request.json.get("email", None)
    password = request.json.get("password", None)
    user = Users.query.filter_by(email=email).first()
    if not user:

        return jsonify({ "message":"Wrong email or password"}),401


    user = Users.query.filter_by(email=email).first()
    if user.is_goopim_admin:

        if check_password_hash(user.password, password):
            # token = jwt.encode({'public_id': user.public_id, 'exp': datetime.utcnow() + timedelta(hours=700)},
            #                    app.config['SECRET_KEY'], "HS256")
            #print(user)

            user_dict = user.to_dict()

            token  = create_access_token(identity=user_dict)

            return jsonify({"message":"logged succesfull","token": token,"userId":user.id,"freelancer":user.is_provider,"employer":user.is_user,"mincon":user.is_goopim_admin}),200

    return jsonify({"message":"login required"}),401

#update user profile


CLIENT_ID = 'your-google-client-id'

@app.route('/google-login', methods=['POST'])
def google_login():
    token = request.json['token']
    try:
        idinfo = id_token.verify_oauth2_token(token, requests.Request(), CLIENT_ID)

        user = Users.query.filter_by(email=idinfo).first()
        if not user:
            return jsonify({"message": "Wrong email or password"}), 401


        if user:
            # token = jwt.encode({'public_id': user.public_id, 'exp': datetime.utcnow() + timedelta(hours=700)},
            #                    app.config['SECRET_KEY'], "HS256")
            # print(user)

            user_dict = user.to_dict()

            token = create_access_token(identity=user_dict)

            return jsonify(
                {"message": "logged succesfull", "token": token, "userId": user.id, "freelancer": user.is_provider,
                 "employer": user.is_user, "mincon": user.is_goopim_admin}), 200

        return jsonify({"message": "login required"}), 401

        # Check if the user's email is in your database
        # If not, create a new user record in your database
        # Generate a JWT token for the user and return it to the frontend
    except ValueError:
        # Invalid token
        return jsonify({'error': 'Invalid token'})

@socketio.on('connect')
#def handle_connect():
   #print(f'Client connected: {request.sid}')

# def upload_image(file_path):
#     # Upload the image to Cloudinary
#     response = cloudinary.uploader.upload(file_path)
#
#     # Retrieve the URL of the uploaded image
#     image_url = response['secure_url']
#
#     return image_url
# def allowed_file(filename):
#     # Get the allowed extensions from the configuration
#     allowed_extensions =['png', 'jpg', 'jpeg', 'gif']
#     # Check if the file extension is in the allowed extensions
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
#
#
# @app.route('/upload-profile-picture', methods=['POST'])
# def upload_profile_picture():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part in the request'})
# #add suploads in static folder
#     file = request.files['file']
#
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
#
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file)
#         #uploaded_image_url = upload_image(file_path)
#         #file.save(os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
#         profile_img_url = f"{request.host_url}{app.config['UPLOAD_FOLDER']}/{filename}"
#         return jsonify({'profile_img_url': profile_img_url})
#
#     return jsonify({'error': 'Invalid file type'})
@app.route('/check_username', methods=['POST'])
def check_username():
    goopim_username = request.json.get('goopim_username')

    if not goopim_username:
        return jsonify({'error': 'goopim_username not provided'}), 400

    # Check if the goopim_username exists in the Users table
    existing_user = Users.query.filter_by(goopim_username=goopim_username).first()

    if existing_user:
        return jsonify({'exists': True}), 200
    else:
        return jsonify({'exists': False}), 200
def update_unique_username(goopim_username, user_id):
    original_username = goopim_username
    counter = 1

    # Check if the original goopim_username already exists in the Users table
    existing_user = Users.query.filter_by(goopim_username=goopim_username).first()

    # If the username already exists and belongs to a different user, append a number at the end until it becomes unique
    while existing_user and existing_user.id != user_id:
        goopim_username = f"{original_username}{counter}"
        counter += 1


    return goopim_username
@app.route('/api/users/update', methods=['PUT'])
@jwt_required()
def update_user():
    current_user_id = get_jwt_identity()['id']
    user = Users.query.filter_by(id=current_user_id).first()

    if not user:
        return jsonify({'message': 'User not found'}), 404

    # Get request data
    data = request.json

    # Update user information
    if 'first_name' in data and data['first_name']:
        user.first_name = data['first_name']
    if 'last_name' in data and data['last_name']:
        user.last_name = data['last_name']
    if 'is_verified' in data and isinstance(data['is_verified'], bool):
        user.is_verified = data['is_verified']
    if 'active' in data and isinstance(data['active'], bool):
        user.active = data['active']
    if 'is_goopim_admin' in data and isinstance(data['is_goopim_admin'], bool):
        user.is_goopim_admin = data['is_goopim_admin']
    if 'is_google_user' in data and isinstance(data['is_google_user'], bool):
        user.is_google_user = data['is_google_user']
    if 'is_user' in data and isinstance(data['is_user'], bool):
        user.is_user = data['is_user']
    if 'is_provider' in data and isinstance(data['is_provider'], bool):
        user.is_provider = data['is_provider']
    if 'profile_img_url' in data and data['profile_img_url']:
        user.profile_img_url = data['profile_img_url']
        update_user_profile_image_cometchat(user.public_id,data['profile_img_url'])
    if 'description' in data and data['description']:
        user.description = data['description']
    if 'keyword' in data and data['keyword']:
        user.keyword = data['keyword']
    if 'hourly_rate' in data and data['hourly_rate']:
        user.hourly_rate = data['hourly_rate']
    if 'portfolio' in data and data['portfolio']:
        user.portfolio = data['portfolio']
    if 'profile_cover_url' in data and data['profile_cover_url']:
        user.profile_cover_url = data['profile_cover_url']
    if 'goopim_username' in data and data['goopim_username']:
        goopim_username =data['goopim_username']

        user.goopim_username =data['goopim_username']
        if goopim_username != user.goopim_username:
             user.goopim_username  = update_unique_username(goopim_username, user.id)
        user.goopim_username =goopim_username


    if 'rating' in data and data['rating']:
        user.rating = data['rating']
    if 'company1' in data and data['company1']:
        company1_id =data['company1']
        company1 = Companies.query.get(company1_id)
        user.past_companies.append(company1)
    if 'company2' in data and data['company2']:
        company2_id =data['company2']
        company2 = Companies.query.get(company2_id)
        user.past_companies.append(company2)
    if 'company3' in data and data['company3']:
        company3_id =data['company3']
        company3 = Companies.query.get(company3_id)
        user.past_companies.append(company3)



    try:
        db.session.commit()
        get_updated_user_details = get_user()
        user_info = get_updated_user_details.json.get("myprofile") if get_updated_user_details.status_code == 200 else None

        print(f"{user_info}")
        if user_info is None:
            return jsonify({"message": "Error fetching user details"}), 500



        modified_user_info = {
            "name": user_info["first_name"],
            "keywords": user_info["keyword"],
            "provider_id": user_info["public_id"],
            "profile_picture": user_info["profile_img_url"],
            "profile_url": user_info["profile_cover_url"],
            "rating":user_info["rating"],
            "description":user_info["description"],
            "portfolio":user_info["portfolio"],
            "hourly_rate":user_info["hourly_rate"],
            "username":user_info["username"]

            # ... Copy other fields as needed
        }
        print(f"{modified_user_info}")
        # Call the function to update the user in Pinecone
        add_or_update_user_function(modified_user_info, embedder, index)


        return jsonify({'message': 'User updated successfully'}), 200
    except IntegrityError:
        db.session.rollback()
        return jsonify({'message': 'An error occurred while updating user information'}), 500

@app.route('/myprofile', methods=['GET'])
@jwt_required()
def get_user():
    current_user_id = get_jwt_identity()["id"]
    user = Users.query.filter_by(id=current_user_id).first()
    past_companies = user.past_companies

    # Iterate over the past companies and access their attributes
    users_past_companies =[]
    for company in past_companies:
        users_past_companies.append((company.id, company.name, company.logo_url))
    return jsonify({"myprofile":user.to_dict(),'pastcompany':users_past_companies})
@app.route('/u/<string:goopim_username>', methods=['GET'])
def get_user_details(goopim_username):
    user = Users.query.filter_by(goopim_username=goopim_username).first()

    if not user:
        return jsonify({'error': 'User not found'}), 404

    user_details = user.to_dict()
    return jsonify({'public_profile':user_details}), 200
@app.route("/api/providers", methods=["POST"])
def get_providers():
    data = request.get_json()

    project_description = data.get("project_description")
    budget = data.get("budget")
    if project_description:
        searchLog = SearchLog(text=project_description)
        db.session.add(searchLog)
        db.session.commit()

    #print(f'project {project_description}{budget}')
    if project_description is None or budget is None:
        return jsonify({"error": "Invalid request"}), 400

    budget = float(budget)

    # Tokenize and remove stopwords from project description
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(project_description.lower())
    keywords = [token for token in tokens if not token in stop_words]

    response = []
    providers = Users.query.filter(
        Users.is_provider == True,
        Users.description != None
    ).all()
    for provider in providers:
        prompt = f"Act as a service comparison service and tell me why should I choose {provider.first_name} for my project {project_description} with no chit-chat and straight to the point based on thier {provider.portfolio}, and their {provider.description}: {project_description}"

        completions = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=300, n=1,
                                               temperature=0.5)
        reason = completions.choices[0].text
        profile_url = f'/u/{provider.goopim_username}'  # get the provider's profile URL from the JSON file
        past_companies = provider.past_companies
        user_address = provider.address
        country = ''
        city = ''
        chatkey= provider.public_id
        username=provider.goopim_username

        if user_address:
            country = user_address.country
            city = user_address.city

        past_company_list = []
        for company in past_companies:
            past_company_list.append({
                'id': company.id,
                'name': company.name,
                'logo_url': company.logo_url
            })
        response.append({"name": provider.first_name, "portfolio": provider.portfolio, "rating": provider.rating,
                         "description": provider.description, "keywords": provider.keyword,
                         "hourly_rate": provider.hourly_rate, "reason": reason,
                         "id": provider.id,'username':username,'profile_url':profile_url,'profile_img_url':provider.profile_img_url,'companies_worked_with':past_company_list, 'country':country,'city':city,'chatkey':chatkey})
   # print(response)
    # Sort providers based on relevance
    def relevance_score(provider):
        if not provider:
            return 0
        # keyword = provider.get("keywords", [])
        # portfolio_score = len(set(keyword).intersection(set(keywords)))
       # print(provider.get("description", ""))
       # print(project_description)
       # print(provider)
        keyword = provider.get("keywords", "")
        if keyword:
            keyword_list = keyword.split(",")
            portfolio_score = len(set(keyword_list).intersection(set(keywords)))
        else:
            portfolio_score = 0
        description_score = 1 if re.search(project_description.lower(), provider.get("description", "").lower()) else 0
        budget_score = 1 if not budget or provider["hourly_rate"] <= budget else 0
        return 3 * portfolio_score + 2 * description_score + budget_score

    response = sorted(response, key=relevance_score, reverse=True)

    # Filter top 3 providers based on relevance and budget
    top_providers = []
    for provider in response:
        if not budget or provider.get("hourly_rate") <= budget:
            provider_copy = provider.copy()
            provider_copy.pop("description")  # remove description field from the response
            top_providers.append(provider_copy)
            if len(top_providers) >= 3:
                break

    # Add button to redirect user to provider's profile on goopim.com
    for provider in top_providers:
        provider["profile_button"] = f'<a href="{provider.get("id")}" target="_blank">View profile</a>'

    return jsonify({"topproviders":top_providers})
@app.route('/searchlogs', methods=['GET'])
def get_search_logs():
    page = request.args.get('page', 1, type=int)
    per_page = 100

    total_count = SearchLog.query.count()
    total_pages = (total_count + per_page - 1) // per_page

    offset = (page - 1) * per_page
    searchlogs = SearchLog.query.order_by(SearchLog.creation_date.desc()).offset(offset).limit(per_page).all()

    if not searchlogs:
        return jsonify({'message': 'No search logs found.'}), 404

    next_page = page + 1 if page < total_pages else None
    prev_page = page - 1 if page > 1 else None

    return jsonify({
        'data': [SearchLog.searchlog_to_dict(self=searchlog) for searchlog in searchlogs],
        'next_page': next_page,
        'prev_page': prev_page,
    })

# @app.route('/searchlogs', methods=['GET'])
# def get_search_logs():
#     page = request.args.get('page', 1, type=int)
#     per_page = 100
#
#     query = SearchLog.query.order_by(SearchLog.creation_date.desc())
#     paginated_searchlogs = query.paginate(page, per_page, error_out=False)
#
#     searchlogs = paginated_searchlogs.items
#     if not searchlogs:
#         return jsonify({'message': 'No search logs found.'}), 404
#
#     next_page = paginated_searchlogs.next_num if paginated_searchlogs.has_next else None
#     prev_page = paginated_searchlogs.prev_num if paginated_searchlogs.has_prev else None
#
#     return jsonify({
#         'data': [searchlog_to_dict(searchlog, next_page, prev_page) for searchlog in searchlogs],
#         'next_page': next_page,
#         'prev_page': prev_page,
#     })

# create a new message
@app.route('/api/messages', methods=['POST'])
@jwt_required()
def create_message():
    data = request.json
    sender_id = data.get('sender_id')
    receiver_id = data.get('receiver_id')
    text = data.get('text')
    message_type = data.get('type', 'normal')

    # retrieve the sender and receiver from the database
    sender = Users.query.filter_by(id=sender_id).first()
    receiver = Users.query.filter_by(id=receiver_id).first()

    if not sender:
        return jsonify({'error': 'Sender not found.'}), 404

    if not receiver:
        return jsonify({'error': 'Receiver not found.'}), 404

    # create or retrieve an existing conversation between the sender and receiver
    conversation = Conversation.query.filter(
        db.or_(
            db.and_(Conversation.user1 == sender, Conversation.user2 == receiver),
            db.and_(Conversation.user1 == receiver, Conversation.user2 == sender)
        )
    ).first()

    if conversation is None:
        if not sender.is_user:
            return jsonify({'error': 'Only employer can start a conversation.'}), 400
        conversation = Conversation(user1=sender, user2=receiver)
        db.session.add(conversation)
        db.session.commit()

        # create a new chat room for the conversation
        room = f'conversation-{conversation.id}'
        #socketio.emit('join_conversation', {conversation.id})
        #socketio.emit('join_room', {'room': room, 'user_id': sender_id}, )
        # socketio.emit('join_conversation', {'conversation_id': conversation.id}, namespace='/chat')
        socketio.emit('new_message',{'message':[{'text':'Hi', 'conversation_id': conversation.id,'sender_id':sender_id,'receiver_id':receiver_id}]},to=room)

        # add the users to the conversation
        # conversation.users.append(sender)
        # conversation.users.append(receiver)
        # db.session.commit()

        # send a message to the client that the conversation has been created
        #socketio.emit('new_conversation', {'conversation': conversation.to_dict()}, room=room)

    # retrieve the chat room for the existing conversation
    else:
        room = f'conversation-{conversation.id}'
        #socketio.emit('join_conversation',{conversation.id})
        print( f'conversation-{conversation.id}')
        #socketio.emit('join_room', {'room': room, 'user_id': sender_id}, namespace='/chat')
    # create a new message and save it to the database
        #room = f'conversation-{conversation.id}'
        message = Message(text=text, type=message_type, sender=sender, receiver=receiver, conversation_id=conversation.id)
        db.session.add(message)
        db.session.commit()
        #my_message_id = message.id

        # broadcast the message to the chat room
        #socketio.emit('new_message', {'message': message.to_dict()}, room=room, include_self=True)
        #join_room(room)

        socketio.emit('new_message', {'message': [{'message_id':message.id,'text':text,'conversation_id': conversation.id, 'sender_id': sender_id, 'receiver_id': receiver_id}]}, to=room)


        return jsonify({'success': True, 'message_id': message.id, 'conversation_id':conversation.id}), 201
# join a conversation chat room
@socketio.on('connect')
def test_connect():
    print('Client connected')
@socketio.on('new_message')
def new_message(message):
    print(f'{message}')
@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

# @socketio.on('new_message')
# def receive_send_message(message):
#     socketio.emit('new_message', {'data': message})

@socketio.on('join_conversation')
def on_join_conversation(data):
    room = f'conversation-{data}'
    join_room(room)
    print(f'joined room {room}')
@socketio.on('join')
def handle_join(conversation_id):
    conversation = Conversation.query.filter_by(id=conversation_id).first()
    if conversation:
        # check that the user is authorized to access the conversation
        user_id = current_user.id
        if user_id != conversation.user1_id and user_id != conversation.user2_id:
            return

        room = f'conversation-{conversation_id}'
        join_room(room)

@app.route('/api/groupmessages/<int:conversation_id>')
@jwt_required()
def get_messages(conversation_id):
    page = request.args.get('page', type=int, default=1)
    load_previous = request.args.get('load_previous', type=bool, default=False)
    messages_per_page = 15
    conversation_id =conversation_id;

    conversation = Conversation.query.get_or_404(conversation_id)

    # Check that the user is authorized to access the conversation
    user_id = get_jwt_identity()['id']
    if user_id != conversation.user1_id and user_id != conversation.user2_id:

        return jsonify({'message':'Not authorized'}), 401

    # Determine the message offset based on the requested page and load_previous parameter
    if load_previous:
        offset = (page - 2) * messages_per_page
    else:
        offset = (page - 1) * messages_per_page

    # Retrieve the messages for the conversation
    messages = (
        Message.query
        .filter_by(conversation_id=conversation_id)
        .order_by(Message.creation_date.desc())
        .limit(messages_per_page)
        .offset(offset)
        .all()
    )
    messages = messages[::-1]
    # Convert the messages to dictionaries and return them
    return jsonify({'messages': [message.to_dict() for message in messages]})

#get users chat rooms - that is   conversation that a user belongs to

@app.route('/api/user_rooms', methods=['GET'])
@jwt_required()
def get_user_rooms():
    user_id = get_jwt_identity()['id']

    if user_id is None:
        return jsonify({'error': 'User not found.'}), 404
    #user = User.query.get(user_id)
    # retrieve all conversations that the user is a part of
    conversations = Conversation.query.filter(
        db.or_(
            Conversation.user1_id == user_id,
            Conversation.user2_id == user_id
        )
    ).all()

    # sort conversations by most recent message timestamp
    conversations = sorted(conversations, key=lambda c: c.last_message().creation_date if c.last_message() else c.creation_date, reverse=True)

    # build JSON response
    myrooms = []
    for conversation in conversations:
        #other_user = conversation.get_other_user(user)
        room = f'conversation-{conversation.id}'
        #join_room(room)
        myrooms.append({
            'id': conversation.id,
            'currentUserId':conversation.user1_id,

            'otherUserId':conversation.user2_id,
            'employer_name': conversation.user1.first_name,
            'freelancer_name':conversation.user2.first_name,
            'freelancer_profile_url':conversation.user2.profile_img_url,
            'employer_profile_url': conversation.user1.profile_img_url,
            'last_message': conversation.last_message().text if conversation.last_message() else '',
            'last_message_timestamp':  conversation.creation_date.strftime('%Y-%m-%d %H:%M:%S')
        })

    return jsonify({'rooms': myrooms})



#join a chat room or create one
def add_user_to_room(user_id, conversation_id):
    # Check if the conversation exists in the database
    conversation = Conversation.query.get(conversation_id)
    if conversation is None:
        # The conversation doesn't exist, return an error response
        return jsonify({'error': f'Conversation {conversation_id} not found.'}), 404

    # Check that the user is authorized to access the conversation
    if user_id not in [conversation.user1_id, conversation.user2_id]:
        # The user is not authorized to access the conversation, return an error response
        return jsonify({'error': 'User is not authorized to access this conversation.'}), 401

    # Join the room for the conversation
    room = f'conversation-{conversation_id}'
    join_room(room)

    return jsonify({'success': f'User joined conversation {conversation_id}.'}), 200
#create a contract
@app.route('/contracts', methods=['POST'])
@jwt_required()
def create_contract():

    contract_title = request.json['contract_title']
    contract_details = request.json['contract_details']
    contract_amount = request.json['contract_amount']
    provider_id = request.json['provider_id']
   # user_id_from_form = request.json['user_id']


    user_id = get_jwt_identity()['id']
    #if user_id == provider_id:
    #    provider_id = user_id_from_form



    # Check if user is valid


    if user_id is None:
         return jsonify({'error': 'Invalid employer'}), 400

    # Check if provider is valid
    provider = Users.query.filter_by(id=provider_id).first()
    get_user_email = Users.query.filter_by(id=user_id).first()



    if not get_user_email.is_user:
        return jsonify({'error': 'Not authorized to create contract'}), 400
    user_email = get_user_email.email
    provider_email = provider.email

    # Create the contract
    contract = Contract(user_id=user_id, provider_id=provider_id,contract_title=contract_title, contract_details=contract_details, contract_amount=contract_amount,contract_status=ContractStatus.PENDING)

    try:
        db.session.add(contract)
        db.session.commit()
        contract_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <style>
            /* Mobile-first responsive styles */
            @media screen and (max-width: 480px) {{
              .container {{
                width: 100%;
                padding: 10px;
              }}

              .message {{
                font-size: 16px;
              }}
            }}
          </style>
        </head>
        <body>
          <div class="container" style="max-width: 600px; margin: 0 auto;">
            <h1>New Contract </h1>
            <p>Hello,</p>
            <p>Contract has been created.</p>



            <p>Thank you!</p>
          </div>
        </body>
        </html>
        """

        message = Mail(
            from_email="sebastian@goopim.com",
            to_emails=[user_email,provider_email],
            subject="Contract",
            html_content=contract_html)

        try:
            # Initialize the SendGrid client with your API key
            sg = SendGridAPIClient(app.config['SENDGRID_API_KEY'])

            # Send the email using the SendGrid client
            response = sg.send(message)

        except Exception as e:
            # Return an error message if there was an issue sending the email
            email_response = 'Error sending email: ' + str(e)

    except:
        db.session.rollback()
        return jsonify({'error': 'Error creating contract'}), 500



    return jsonify({'message': 'Contract created successfully'})

#inputs contact_id, contract_status (REJECTED or ACCEPTED), conversation_id
@app.route('/contracts/<int:contract_id>/edit', methods=['PUT'])
@jwt_required()
def edit_contract(contract_id):
    current_user_id = get_jwt_identity()['id']

    contract = Contract.query.filter_by(id=contract_id).first()
    if not contract:
        return jsonify({'error': 'Contract not found'}), 404

    #email notification setup
    user = Users.query.filter_by(id=contract.user_id).first()
    provider = Users.query.filter_by(id=contract.provider_id).first()
    if contract.user_id != user.id:
        return jsonify({'error': 'user is not authorized'}), 404
    if contract.provider_id != provider.id:
        return jsonify({'error': 'provider is not authorized'}), 404
    user_email = user.email
    provider_email = provider.email

    contract_accepted_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        /* Mobile-first responsive styles */
        @media screen and (max-width: 480px) {{
          .container {{
            width: 100%;
            padding: 10px;
          }}

          .message {{
            font-size: 16px;
          }}
        }}
      </style>
    </head>
    <body>
      <div class="container" style="max-width: 600px; margin: 0 auto;">
        <h1>Contract status</h1>
        <p>Hello,</p>
        <p>Contract has been accepted and project created.</p>
        <p>Go to project tab to view project</p>
        
      
        <p>Thank you!</p>
      </div>
    </body>
    </html>
    """
    contract_rejected_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        /* Mobile-first responsive styles */
        @media screen and (max-width: 480px) {{
          .container {{
            width: 100%;
            padding: 10px;
          }}

          .message {{
            font-size: 16px;
          }}
        }}
      </style>
    </head>
    <body>
      <div class="container" style="max-width: 600px; margin: 0 auto;">
        <h1>Contract status</h1>
        <p>Hello,</p>
        <p>Contract has been rejected.</p>
      


        <p>Thank you!</p>
      </div>
    </body>
    </html>
    """
    contract_updated_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        /* Mobile-first responsive styles */
        @media screen and (max-width: 480px) {{
          .container {{
            width: 100%;
            padding: 10px;
          }}

          .message {{
            font-size: 16px;
          }}
        }}
      </style>
    </head>
    <body>
      <div class="container" style="max-width: 600px; margin: 0 auto;">
        <h1>Contract status</h1>
        <p>Hello,</p>
        <p>Contract has been updated.</p>



        <p>Thank you!</p>
      </div>
    </body>
    </html>
    """



    contract_status = request.json.get('contract_status')
    if not contract_status:
        return jsonify({'error': 'Contract status not provided'}), 400

    # conversation_id = request.json.get('conversation_id')
    # if not conversation_id:
    #     return jsonify({'error': 'conversation_id not provided'}), 400
    if request.json.get('contract_title'):
        contract_title = request.json.get('contract_title')
    if request.json.get('contract_details'):
        contract_details = request.json.get('contract_details')
    if  request.json.get('contract_amount'):
        contract_amount = request.json.get('contract_amount')
    # if not conversation_id:
    #     return jsonify({'error': 'conversation_id not provided'}), 400

    if contract_status == 'CANCELLED' and contract.contract_status ==ContractStatus.PENDING:
        # Update contract status
        contract.contract_status = ContractStatus.CANCELLED
        db.session.commit()
        message = Mail(
            from_email="sebastian@goopim.com",
            to_emails=[user_email,provider_email],
            subject="Contract Rejected",
            html_content=contract_rejected_html)

        try:
            # Initialize the SendGrid client with your API key
            sg = SendGridAPIClient(app.config['SENDGRID_API_KEY'])

            # Send the email using the SendGrid client
            response = sg.send(message)
            # print(response.status_code)
            # print(response.body)
            # print(response.headers)

            # Return a success message if the email was sent successfully
            email_response = 'Reset password link sent'
        except Exception as e:
            # Return an error message if there was an issue sending the email
            email_response = 'Error sending email: ' + str(e)


        return jsonify({'success': True, 'message': 'Contract rejected'}), 201



    elif contract_status == 'ACCEPTED' and contract.contract_status == ContractStatus.PENDING:
        # Check if user has enough funds

        user_id = contract.user_id
        if current_user_id != user_id:
            return jsonify({'error':'Not authorized'}),400
        provider_id = contract.provider_id
        # user_account = MyPaymentAccount.query.filter_by(user_id=contract.user_id).first()
        # goopim_admin_account =MyPaymentAccount.query.filter_by(user_id=7).first()
        # provider_account = MyPaymentAccount.query.filter_by(user_id=contract.provider_id).first()
        # if user_account.balance < contract.contract_amount:
        #     amount_remaining = contract.contract_amount - user_account.balance
        #     return jsonify({'error': 'Insufficient funds','amount_remaining':amount_remaining}), 400

        # Update contract status
        contract.contract_status = ContractStatus.ACCEPTED
        db.session.commit()

        # Create project

        project = Project(user_id=contract.user_id, provider_id=contract.provider_id, contract_amount=contract.contract_amount)
        db.session.add(project)
        db.session.commit()

        message = Mail(
            from_email="sebastian@goopim.com",
            to_emails=[user_email,provider_email],
            subject="Contract Accepted",
            html_content=contract_accepted_html)

        try:
            # Initialize the SendGrid client with your API key
            sg = SendGridAPIClient(app.config['SENDGRID_API_KEY'])

            # Send the email using the SendGrid client
            response = sg.send(message)

        except Exception as e:
            # Return an error message if there was an issue sending the email
            email_response = 'Error sending email: ' + str(e)

        return jsonify({'success': True, 'message': 'Contract accepted and projected created'}), 201

    elif contract.contract_status == ContractStatus.ACCEPTED:

        return jsonify({'message': 'Contract is already accepted'})
    elif contract.contract_status == ContractStatus.CANCELLED:

        return jsonify({'message': 'Contract is already rejected'})

    elif contract.contract_status == ContractStatus.PENDING:

        contract.contract_title = contract_title
        contract.contract_details = contract_details
        contract.contract_amount = contract_amount
        db.session.commit()
        message = Mail(
            from_email="sebastian@goopim.com",
            to_emails=[user_email,provider_email],
            subject="Contract Updated",
            html_content=contract_updated_html)

        try:
            # Initialize the SendGrid client with your API key
            sg = SendGridAPIClient(app.config['SENDGRID_API_KEY'])

            # Send the email using the SendGrid client
            response = sg.send(message)

        except Exception as e:
            # Return an error message if there was an issue sending the email
            email_response = 'Error sending email: ' + str(e)

        return jsonify({'message': 'Contract updated'})
    else:
        return jsonify({'error': 'Invalid contract status'}), 400


@app.route('/contracts/<int:contract_id>', methods=['GET'])
@jwt_required()
def get_single_contract(contract_id):
    current_user_id = get_jwt_identity()['id']

    contract = Contract.query.filter_by(id=contract_id).first()
    if contract.user_id == current_user_id or contract.provider_id == current_user_id:
        return  jsonify({'contract':[contract.serialize()]})
    else:
        return jsonify({'message':'not authorized'})
@app.route('/users_contract', methods=['GET'])
@jwt_required()
def get_all_users_contracts():
  
    current_user = get_jwt_identity()['id']
    user = Users.query.filter_by(id=current_user).first()
    if user.is_provider:

        contracts = Contract.query.filter_by(provider_id=current_user).order_by(Contract.creation_date).all()
    else:

        contracts = Contract.query.filter_by(user_id=current_user).order_by(Contract.creation_date).all()



    serialized_contracts = [contract.serialize() for contract in contracts]
    return jsonify({"contracts": serialized_contracts})

#retrieve project details
@app.route('/project/<int:project_id>', methods=['GET'])
@jwt_required()
def get_project_data(project_id):
    project = Project.query.get(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    milestone_escrow_accounts = project.milestone_escrow_accounts
    quotes = project.quotes
    timelines = project.timelines
    deliverables = project.deliverables

    return jsonify({
        'milestone_escrow_accounts': [mea.serialize() for mea in milestone_escrow_accounts],
        'quotes': [quote.serialize() for quote in quotes],
        'timelines': [timeline.serialize() for timeline in timelines],
        'deliverables': [deliverable.serialize() for deliverable in deliverables]
    })


#retrieve project details
@app.route('/project/<int:project_id>', methods=['PUT'])
@jwt_required()
def edit_project_data(project_id):
    data = request.get_json()
    project_title = data["project_title"]
    project_description = data["project_description"]
    project = Project.query.get(project_id)
    currentuser_id = get_jwt_identity()["id"]

    if not project:
        return jsonify({'error': 'Project not found'}), 404

    if currentuser_id != project.user_id:
        return  jsonify({'error': 'Not authorized to edit'}), 404

    project.title = project_title
    project.description = project_description

    db.session.commit()



    return jsonify({"message":"project updated"

    })
#get all project for a user
@app.route('/projects', methods=['GET'])
@jwt_required()
def get_projects():
    current_user_id = get_jwt_identity()['id']
    user = Users.query.filter_by(id =current_user_id).first()
    #is_current_user_provider =get_jwt_identity()['is_provider']
    if user.is_provider:
        projects = Project.query.filter_by(provider_id=current_user_id).all()
    else:
        projects = Project.query.filter_by(user_id=current_user_id).all()

    # Serialize the projects and return them as a response
    return jsonify([p.serialize() for p in projects])


@app.route('/withdraw', methods=['POST'])
@jwt_required()
def withdraw():
    current_user_id = get_jwt_identity()["id"]
    user_id = request.json.get('user_id')
    amount = request.json.get('amount')
    amount = float(amount)

    # # Check if the authenticated user is the same as the requested user_id
    # if current_user_id != user_id:
    #     return jsonify({'error': 'Not authorized'}), 401

    # Check if the user has enough balance
    account = MyPaymentAccount.query.filter_by(user_id=current_user_id).first()
    if account is None:
        return jsonify({'error': 'User account not found'}), 404
    if account.balance < amount:
        return jsonify({'error': 'Insufficient balance'}), 400

    # Debit the amount from the user's account
    account.balance -= amount
    db.session.add(account)

    # Create a new withdrawal request
    #withdrawal = WithdrawalRequest(WithdrawalStatus.PENDING,user_id=current_user_id, amount=amount)
    new_withdrawal_request = WithdrawalRequest.create_new_withdrawal_request(user_id=current_user_id, amount=amount)
    #db.session.add(withdrawal)

    #db.session.commit()

    return jsonify(new_withdrawal_request), 201

@app.route('/withdrawals/all', methods=['GET'])
@jwt_required()
def get_user_withdrawals():
    current_user_id = get_jwt_identity()["id"]
    withdrawals = WithdrawalRequest.query.filter_by(user_id=current_user_id).order_by(WithdrawalRequest.creation_date.desc()).all()
    serialized_withdrawals = [withdrawal.serialize() for withdrawal in withdrawals]
    return jsonify({"withdrawal":serialized_withdrawals})



def is_admin(id):
    user = Users.query.filter_by(id=id).first()
    if user.is_goopim_admin:
        return True
    else:
        return False
@app.route('/admin/withdrawals/pending', methods=['GET'])
@jwt_required()
def get_pending_withdrawals():
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        withdrawals = WithdrawalRequest.query.filter_by(status=WithdrawalStatus.PENDING).all()
        serialized_withdrawals = [withdrawal.serialize() for withdrawal in withdrawals]
        return jsonify({"withdrawals":serialized_withdrawals}), 200
    else:
        return jsonify({'message': 'Unauthorized access'}), 401


@app.route('/admin/withdrawals/processing', methods=['GET'])
@jwt_required()
def get_processing_withdrawals():
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        withdrawals = WithdrawalRequest.query.filter_by(status=WithdrawalStatus.PROCESSING).all()
        serialized_withdrawals = [withdrawal.serialize() for withdrawal in withdrawals]
        return jsonify({"withdrawals":serialized_withdrawals}), 200
    else:
        return jsonify({'message': 'Unauthorized access'}), 401


@app.route('/admin/withdrawals/completed', methods=['GET'])
@jwt_required()
def get_completed_withdrawals():
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        withdrawals = WithdrawalRequest.query.filter_by(status=WithdrawalStatus.COMPLETED).all()
        serialized_withdrawals = [withdrawal.serialize() for withdrawal in withdrawals]
        return jsonify({"withdrawals":serialized_withdrawals}), 200
    else:
        return jsonify({'message': 'Unauthorized access'}), 401




@app.route('/admin/withdrawals/<int:id>', methods=['PUT'])
@jwt_required()
def update_withdrawal_status(id):
    withdrawal = WithdrawalRequest.query.get_or_404(id)
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400
    status = request.json.get('status')
    status = status.upper()
    if not status:
        return jsonify({"msg": "Missing status parameter"}), 400
    if status not in WithdrawalStatus.__members__:
        return jsonify({"msg": "Invalid status parameter"}), 400
    withdrawal.status = WithdrawalStatus[status.upper()]
    db.session.commit()
    return jsonify({"msg": "success"})





@app.route('/charge', methods=['POST'])
def charge():
    # Get payment details from request body
    amount = request.json.get('amount')
    email = request.json.get('email')
    payment_method_id = request.json.get('payment_method_id')
    print(f'pid{payment_method_id}')

    # Verify payment amount and currency
    if not amount or not email or not payment_method_id:
        return jsonify({'error': 'Invalid payment details.'}), 400
    try:
        amount = float(amount)
        if amount <= 0:
            raise ValueError('Payment amount must be positive.')
        # if request.json.get('currency') != 'usd':
        #     raise ValueError('Only USD currency is currently supported.')
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # Create a new customer object in Stripe API
    try:
        customer = stripe.Customer.create(email=email)
    except stripe.error.StripeError as e:
        return jsonify({'error': str(e)}), 500
    # Verify that the payment method is valid
    try:
        payment_method = stripe.PaymentMethod.retrieve(payment_method_id)
    except stripe.error.StripeError as e:
        return jsonify({'error': str(e)}), 500

@app.route('/deliverables', methods=['POST'])
@jwt_required()
def create_deliverable():
    user_id = get_jwt_identity()['id']

    project_id = request.json['project_id']

    description = request.json['description']

    project = Project.query.get(project_id)
    current_project_employer = project.user_id
    current_project_freelancer = project.provider_id

    if user_id != current_project_freelancer or user_id != current_project_freelancer:
        return jsonify({'error': 'Not member of this project'}), 400

    # Check if user and provider are valid
    user = Users.query.filter_by(id=user_id).first()

    if user is None:
        return jsonify({'error': 'Invalid user'}), 400

    # Create the deliverable
    deliverable = Deliverable(
        project_id=project_id,
        user_id=current_project_employer,
        provider_id=current_project_freelancer,
        description=description
    )

    try:
        db.session.add(deliverable)
        db.session.commit()
    except:
        db.session.rollback()
        return jsonify({'error': 'Error creating deliverable'}), 500

    return jsonify({'deliverables':deliverable.serialize()}), 201

@app.route('/timelines', methods=['POST'])
@jwt_required()
def create_timeline():
    user_id = get_jwt_identity()['id']
    project_id = request.json['project_id']

    timeline_description = request.json['timeline_description']
    timeline_time = request.json.get('timeline_time', datetime.utcnow())

    project = Project.query.get(project_id)
    current_project_employer = project.user_id
    current_project_freelancer = project.provider_id

    if user_id != current_project_freelancer or user_id != current_project_freelancer:
        return jsonify({'error': 'Not member of this project'}), 400

    # Check if user and provider are valid
    user = Users.query.filter_by(id=user_id).first()

    if user is None:
        return jsonify({'error': 'Invalid user or provider'}), 400

    # Create the timeline
    timeline = Timeline(
        project_id=project_id,
        user_id=current_project_employer,
        provider_id=current_project_freelancer,
        timeline_description=timeline_description,
        timeline_time=timeline_time
    )

    try:
        db.session.add(timeline)
        db.session.commit()
    except:
        db.session.rollback()
        return jsonify({'error': 'Error creating timeline'}), 500

    return jsonify({'timelines':timeline.serialize()}), 201

@app.route('/projects/<int:project_id>/deliverables', methods=['GET'])
@jwt_required()
def get_deliverables(project_id):
    user_id = get_jwt_identity()['id']

    project = Project.query.get(project_id)
    current_project_employer = project.user_id
    current_project_freelancer = project.provider_id

    if user_id != current_project_freelancer or user_id != current_project_freelancer:
        return jsonify({'error': 'Not member of this project'}), 400
    deliverables = Deliverable.query.filter_by(project_id=project_id).all()

    return jsonify({'deliverables':[deliverable.serialize() for deliverable in deliverables]}), 200

@app.route('/projects/<int:project_id>/timelines', methods=['GET'])
@jwt_required()
def get_timelines(project_id):
    user_id = get_jwt_identity()['id']

    project = Project.query.get(project_id)
    current_project_employer = project.user_id
    current_project_freelancer = project.provider_id

    if user_id != current_project_freelancer or user_id != current_project_freelancer:
        return jsonify({'error': 'Not member of this project'}), 400

    timelines = Timeline.query.filter_by(project_id=project_id).all()

    return jsonify({'timelines':[timeline.serialize() for timeline in timelines]}), 200

    #
    # if payment_method['status'] == 'consumed':
    #     return jsonify({'error': 'Payment method has already been used'}), 500
    #

    # Create a payment method for the customer
    # try:
    #     payment_method = stripe.PaymentMethod.create(
    #         type='card',
    #         card={
    #             'token': payment_method_id
    #         }
    #     )
    # except stripe.error.StripeError as e:
    #     return jsonify({'error': str(e)}), 500

    # Attach the payment method to the customer
    # try:
    #     customer.payment_methods.attach(payment_method)
    # except stripe.error.StripeError as e:
    #     return jsonify({'error': str(e)}), 500
    #
    # # Set the newly added payment method as the default for future charges
    # try:
    #     customer.invoice_settings.default_payment_method = payment_method.id
    #     customer.save()
    # except stripe.error.StripeError as e:
    #     return jsonify({'error': str(e)}), 500

    # Charge the customer's payment source
    try:
        charge = stripe.Charge.create(
            customer=customer.id,
            amount=amount,
            currency='usd',
            description='My Payment Account Charge'
        )
        print(f'charge by stripe {charge}')
    except stripe.error.StripeError as e:
        return jsonify({'error': str(e)}), 500

    # Update user's balance in your database
    # Return a response to your frontend
    return jsonify({'success': True, 'payment_id': charge.id, 'status': charge.status}), 200
def calculate_order_amount(items):
    # Replace this constant with a calculation of the order's amount
    # Calculate the order total on the server to prevent
    # people from directly manipulating the amount on the client
    #total = items[0].id
    return 1000
@app.route('/create-payment-intent', methods=['POST'])
def create_payment():
    try:
        data = json.loads(request.data)
        # Create a PaymentIntent with the order amount and currency
        intent = stripe.PaymentIntent.create(
            amount=calculate_order_amount(data['items']),
            currency='usd',
            automatic_payment_methods={
                'enabled': True,
            },
        )
        return jsonify({
            'clientSecret': intent['client_secret']
        })
    except Exception as e:
        return jsonify(error=str(e)), 403
@app.route('/api/deposit', methods=['POST'])
@jwt_required()
def create_paymentY():
    data = json.loads(request.data)
    #email = data['email']
    amount = str(int(data['amount'])*100)
    payment_method_id =data['payment_method_id']
    #amount = amount*100

    user_id = get_jwt_identity()['id']
    user = Users.query.filter_by(id = user_id).first()
    email = user.email
    name = f'{user.first_name} {user.last_name}'

    customer = stripe.Customer.create(
        email=email,
        name=name
    )
    try:
        payment_intent = stripe.PaymentIntent.create(
            amount=amount,
            currency='usd',
            payment_method_types=['card'],
            payment_method=payment_method_id,
            receipt_email=email, # Include the email address in the payment request
            customer = customer.id
        )
    except stripe.error.CardError as e:
        return jsonify(error=str(e)), 400

    return jsonify(client_secret=payment_intent.client_secret)

@app.route('/api/depo', methods=['POST'])
@jwt_required()
def stripePaymentElement():
    data = json.loads(request.data)
    #email = data['email']
    amount = str(int(data['amount'])*100)
    #payment_method_id =data['payment_method_id']


    user_id = get_jwt_identity()['id']
    user = Users.query.filter_by(id = user_id).first()
    email = user.email
    name = f'{user.first_name} {user.last_name}'
    customer = stripe.Customer.create(
        email=email,
        name=name
    )
    try:

        intent = stripe.PaymentIntent.create(
            amount=amount,
            currency='usd',
            automatic_payment_methods={
                'enabled': True,
            },
            receipt_email=email,  # Include the email address in the payment request
            customer=customer.id
        )
    except stripe.error.StripeError as e:
        return jsonify(error=str(e)), 400
    return jsonify(client_secret=intent.client_secret)

    # # Get the amount and payment method id from the request body
    # data = request.get_json()
    # amount = data['amount']
    # payment_method_id = data['payment_method_id']
    #
    # # Verify that the payment method is valid
    # if payment_method_id:
    #     try:
    #         payment_intent = stripe.PaymentMethod.retrieve(payment_method_id)
    #         print(f'payment m {payment_intent}')
    #         if payment_intent:
    #
    #             updated_intent = stripe.PaymentMethod.update(
    #             payment_method_id_id,
    #
    #             amount=amount)
    #
    #
    #             return jsonify(updated_intent);
    #
    #     except stripe.error.StripeError as e:
    #
    #         return jsonify({'error': str(e)}), 500
    #
    # return jsonify({"error":"error occured"}),500



    # # Create a payment intent with the verified payment method
    # try:
    #     intent = stripe.PaymentIntent.create(
    #         amount=amount,
    #         currency='usd',
    #         payment_method=payment_method_id,
    #         confirm=True
    #     )
    # except stripe.error.StripeError as e:
    #     return jsonify({'error': str(e)}), 500
    #
    # # Return the payment intent status to the frontend
    # return jsonify({'status': intent.status})


@app.route('/projects/<int:project_id>/escrow_accounts/<int:user_id>')
def get_project_escrow_accounts(project_id, user_id):
    escrow_accounts = MilestoneEscrowAccount.query.filter_by(project_id=project_id, user_id=user_id).all()
    if not escrow_accounts:
        return jsonify({'error': 'No escrow accounts found.'}), 404
    return jsonify({'escrow_accounts': [account.to_dict() for account in escrow_accounts]})

@app.route('/release_milestone', methods=['POST'])
@jwt_required()
def release_milestone():
    data = request.json
    project_id = data.get('project_id')
    provider_id = data.get('provider_id')
    user_id = data.get('user_id')


    # Check if user is a provider
    current_user = get_jwt_identity()["id"]
    employer = Users.query.get(current_user)
    print(f'provider{current_user}')
    if not employer.is_user:
        return jsonify({'error': 'Not authorized to release milestone.'}), 400

    # Debit the escrow account and credit the user's account
    escrow_account = MilestoneEscrowAccount.query.filter_by(project_id=project_id, provider_id=provider_id, user_id=user_id, completed=False).first()
    if not escrow_account:
        return jsonify({'error': 'Not found.'}), 404

    amount = escrow_account.amount
    escrow_account.completed = True

    user_account = MyPaymentAccount.query.filter_by(user_id=user_id).first()
    if not user_account:
        user_account = MyPaymentAccount(user_id=user_id, balance=0)
        db.session.add(user_account)

    user_account.balance += amount

    db.session.commit()

    return jsonify({'success': True, 'amount': amount,'message':'Milestone released'}), 200


@app.route('/admin/users', methods=['GET'])
@jwt_required()
def get_all_users():
    current_user = Users.query.filter_by(public_id=get_jwt_identity()["public_id"]).first()
    if not current_user.is_goopim_admin:
        return jsonify({'message': 'You do not have permission to perform this action'}), 403

    users = Users.query.all()
    return jsonify({'users': [user.to_dict() for user in users]})


# Get all projects
@app.route('/admin/projects', methods=['GET'])
@jwt_required()
def get_admin_projects():
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        projects = Project.query.all()
        serialized_projects = [p.serialize() for p in projects]
        return jsonify({"projects":serialized_projects})
    else:
        return jsonify({'error':"you're not authorized "})




# Get a specific project by ID
@app.route('/admin/projects/<int:project_id>', methods=['GET'])
@jwt_required()
def get_admin_project(project_id):
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        project = Project.query.get_or_404(project_id)
        serialized_project = project.serialize()
        return jsonify({"project":serialized_project})
    else:
        return jsonify({'error':"you're not authorized "})

# Update a project's provider ID and/or contract amount
@app.route('/admin/projects/<int:project_id>', methods=['PUT'])
@jwt_required()
def update_admin_project(project_id):
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        project = Project.query.get_or_404(project_id)

        if 'provider_id' in request.json:
            project.update_is_provider(request.json['provider_id'])
        if 'contract_amount' in request.json:
            project.update_contract_amount(request.json['contract_amount'])

        serialized_project = project.serialize()
        return jsonify({"message":"update succesfull"})
    else:
        return jsonify({'error':"you're not authorized "})


@app.route('/admin/transactions/<int:transaction_id>', methods=['GET'])
@jwt_required()
def get_admin_transaction(transaction_id):
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        transaction = Transaction.query.get_or_404(transaction_id)
        serialized_transaction = transaction.serialize()
        return jsonify({'transaction': serialized_transaction})
    else:
        return jsonify({'error': 'You are not authorized to view this transaction.'})

@app.route('/admin/transactions', methods=['GET'])
@jwt_required()
def get_all_transactions():
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        transactions = Transaction.query.all()
        serialized_transactions = [t.serialize() for t in transactions]
        return jsonify({"transactions": serialized_transactions})
    else:
        return jsonify({'error': "you're not authorized "})


@app.route('/admin/payment-accounts', methods=['GET'])
@jwt_required()
def get_admin_payment_accounts():
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        payment_accounts = MyPaymentAccount.query.all()
        serialized_payment_accounts = [account.serialize() for account in payment_accounts]
        return jsonify({"payment_accounts": serialized_payment_accounts})
    else:
        return jsonify({'error':"you're not authorized "})

@app.route('/admin/payment-accounts/<int:account_id>', methods=['GET'])
@jwt_required()
def get_admin_payment_account(account_id):
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        payment_account = MyPaymentAccount.query.get_or_404(account_id)
        serialized_payment_account = payment_account.serialize()
        return jsonify({"payment_account": serialized_payment_account})
    else:
        return jsonify({'error':"you're not authorized "})
@app.route('/admin/milestone_escrow_accounts', methods=['GET'])
@jwt_required()
def get_all_milestone_escrow_accounts():
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        milestone_escrow_accounts = MilestoneEscrowAccount.query.all()
        serialized_accounts = [account.serialize() for account in milestone_escrow_accounts]
        return jsonify({"milestone_escrow_accounts": serialized_accounts})
    else:
        return jsonify({'error':"you're not authorized "})

@app.route('/admin/milestone_escrow_accounts/<int:account_id>', methods=['GET'])
@jwt_required()
def get_milestone_escrow_account(account_id):
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        account = MilestoneEscrowAccount.query.get_or_404(account_id)
        serialized_account = account.serialize()
        return jsonify({"milestone_escrow_account": serialized_account})
    else:
        return jsonify({'error':"you're not authorized "})
@app.route('/admin/milestones', methods=['GET'])
@jwt_required()
def get_all_milestones():
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        milestones = Milestone.query.all()
        serialized_milestones = [milestone.serialize() for milestone in milestones]
        return jsonify({"milestones": serialized_milestones})
    else:
        return jsonify({'error':"you're not authorized "})

@app.route('/admin/milestones/<int:milestone_id>', methods=['GET'])
@jwt_required()
def get_admin_milestone(milestone_id):
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        milestone = Milestone.query.get_or_404(milestone_id)
        serialized_milestone = milestone.serialize()
        return jsonify({"milestone": serialized_milestone})
    else:
        return jsonify({'error':"you're not authorized "})

@app.route('/escrow_milestones', methods=['POST'])
@jwt_required()
def create_escrow_milestone():
    provider_id = get_jwt_identity()['id']
    project_id = request.json.get('project_id')
    user_id = request.json.get('user_id')
    amount = request.json.get('amount')
    description = request.json.get('description')

    # Check if provider has enough balance in Mypaymentaccount
    provider = User.query.get(provider_id)
    if not provider.is_provider:
        return jsonify({'error': 'Not authorized'})
    if provider.balance < amount:
        return jsonify({'error': 'Provider does not have enough balance to create milestone'})

    # Debit Mypaymentaccount
    provider.balance -= amount
    db.session.commit()

    # Credit EscrowMilestone account
    escrow_account = MilestoneEscrowAccount(project_id=project_id,
                                            user_id=user_id,
                                            provider_id=provider_id,
                                            amount=amount)
    db.session.add(escrow_account)
    db.session.commit()

    # Create initial milestone
    escrow_account.create_more_milestone(milestone_description=description)

    return jsonify({'message': 'Escrow milestone created successfully'})

@app.route('/admin/contracts', methods=['GET'])
@jwt_required()
def get_all_contracts():
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        contracts = Contract.query.all()
        serialized_contracts = [contract.serialize() for contract in contracts]
        return jsonify({"contracts": serialized_contracts})
    else:
        return jsonify({'error': "you're not authorized"})


@app.route('/admin/contracts/<int:id>', methods=['GET'])
@jwt_required()
def get_contract(id):
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        contract = Contract.query.get(id)
        if contract:
            serialized_contract = contract.serialize()
            return jsonify({"contract": serialized_contract})
        else:
            return jsonify({'error': 'contract not found'})
    else:
        return jsonify({'error': "you're not authorized"})


@app.route('/admin/contracts/<int:id>', methods=['DELETE'])
@jwt_required()
def delete_contract(id):
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        contract = Contract.query.get(id)
        if contract:
            db.session.delete(contract)
            db.session.commit()
            return jsonify({'message': 'contract deleted successfully'})
        else:
            return jsonify({'error': 'contract not found'})
    else:
        return jsonify({'error': "you're not authorized"})




@app.route('/webhook', methods=['POST'])
def webhook():
    event = None
    payload = request.data
    sig_header = request.headers['STRIPE_SIGNATURE']

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError as e:
        # Invalid payload
        raise e
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        raise e

    # Handle the event
    if event['type'] == 'payment_intent.succeeded':



        pob = event['data']['object']
        payment_amount = event['data']['object']['amount_received']
        payment_email = event['data']['object']['receipt_email']
        print(f'pobd {pob}')
        print(f'{payment_amount}')
        print(f'{payment_email}')

        user = Users.query.filter_by(email=payment_email).first()

        try:
            mydeposit = MyPaymentAccount.query.filter_by(user_id =user.id).first()
            mydeposit.balance += payment_amount/100

            db.session.commit()
            return jsonify({'message':'payment is successful'})
        except:

            return jsonify({'error':'payment failed'}),400




            print(f'{payment_amount}{payment_email}')
    # ... handle other event types

    else:
      print('Unhandled event type {}'.format(event['type']))

    return jsonify(success=True)



@app.route('/balance', methods=['GET'])
@jwt_required()
def get_balance():
    user_id = get_jwt_identity()['id']
    payment_account = MyPaymentAccount.query.filter_by(user_id=user_id).first()
    if payment_account is None:
        return jsonify({'error': 'Payment account not found'}), 404
    return jsonify(payment_account.serialize()), 200


@app.route('/companies', methods=['POST'])
def create_company():
    data = request.get_json()
    name = data.get('name')
    logo_url = data.get('logoUrl')

    if not name or not logo_url:
        return jsonify({'error': 'Name and logo URL are required'}), 400

    company = Companies(name=name, logo_url=logo_url)
    db.session.add(company)
    db.session.commit()

    return jsonify({'success': True, 'message': 'Company created successfully'}), 201


@app.route('/companies/<int:company_id>', methods=['PUT'])
def edit_company(company_id):
    company = Companies.query.get(company_id)

    if not company:
        return jsonify({'error': 'Company not found'}), 404

    data = request.get_json()
    name = data.get('name')
    logo_url = data.get('logoUrl')

    if not name or not logo_url:
        return jsonify({'error': 'Name and logo URL are required'}), 400

    company.name = name
    company.logo_url = logo_url
    db.session.commit()

    return jsonify({'success': True, 'message': 'Company updated successfully'}), 200


@app.route('/companies/<int:company_id>', methods=['DELETE'])
def delete_company(company_id):
    company = Companies.query.get(company_id)

    if not company:
        return jsonify({'error': 'Company not found'}), 404

    db.session.delete(company)
    db.session.commit()

    return jsonify({'success': True, 'message': 'Company deleted successfully'}), 200
@app.route('/companies/<int:company_id>', methods=['GET'])
def get_a_company(company_id):
    company = Companies.query.get(company_id)
    if not company:
        return jsonify({'error': 'Company not found'}), 404


    return jsonify(company.serialize()), 200

@app.route('/companies', methods=['GET'])
def get_all_companies():
    companies = Companies.query.all()

    company_list = []
    for company in companies:
        company_data = {
            'id': company.id,
            'name': company.name,
            'logo_url': company.logo_url
        }
        company_list.append(company_data)

    return jsonify({'all_companies':company_list}), 200

@app.route('/users/address', methods=['POST'])
@jwt_required()
def create_user_address():
    user_id = get_jwt_identity()["id"]
    data = request.get_json()
    country = data.get('country')
    city = data.get('city')
    street1 = data.get('street1')
    street2 = data.get('street2')

    if not user_id or not country or not city or not street1:
        return jsonify({'error': 'User ID, country, city, and street1 are required'}), 400

    user = Users.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    address = Address(country=country, city=city, street1=street1, street2=street2, user_id=user_id)
    db.session.add(address)
    db.session.commit()

    return jsonify({'success': True, 'message': 'User address created successfully'}), 201

@app.route('/users/address', methods=['PUT'])
def update_user_address():
    current_user_id = get_jwt_identity()["id"]
    user = Users.query.get(current_user_id)

    if not user:
        return jsonify({'error': 'User not found'}), 404

    address = user.address

    if not address:
        return jsonify({'error': 'Address not found'}), 404

    data = request.get_json()
    country = data.get('country')
    city = data.get('city')
    street1 = data.get('street1')
    street2 = data.get('street2')


    if country:
        address.country = country
    if city:
        address.city = city
    if street1:
        address.street1 = street1
    if street2:
        address.street2 = street2

    db.session.commit()

    return jsonify({'success': True, 'message': 'Address updated successfully'}), 200

@app.route('/users/address', methods=['GET'])
@jwt_required()
def get_user_address():
    user_id = get_jwt_identity()["id"]
    user = Users.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    address = user.address
    if not address:
        return jsonify({'error': 'Address not found'}), 404

    address_data = {
        'id': address.id,
        'country': address.country,
        'city': address.city,
        'street1': address.street1,
        'street2': address.street2,
        'user_id': address.user_id
    }

    return jsonify({'address':address_data}), 200


@app.route('/cometchat_email_notification', methods=['POST'])
def chatnotificationwebhook():
    data = request.json  # Get the JSON data from the request

    to = data['to']
    body = data['body']
    text =data['body'][0]['data']['text']

    sender_name = data['body'][0]['data']['entities']['sender']['entity']['name']  # Replace with the actual sender's name
    message_text = text  # Replace with the actual message content

    # Render the HTML template with dynamic values
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        /* Mobile-first responsive styles */
        @media screen and (max-width: 480px) {{
          .container {{
            width: 100%;
            padding: 10px;
          }}

          .message {{
            font-size: 16px;
          }}
        }}
      </style>
    </head>
    <body>
      <div class="container" style="max-width: 600px; margin: 0 auto;">
        <h1>New Message Notification</h1>
        <p>Hello,</p>
        <p>You have received a new message from {sender_name}.</p>
        <div class="message">
          <p>{message_text}</p>
        </div>
        <p>Thank you!</p>
      </div>
    </body>
    </html>
    """

    # Create and send the email
    # msg = Message("New Message Notification", sender="sebastian@goopim.com", recipients=[to])
    # msg.html = html
    #
    # mail.send(msg)

    # Create a SendGrid message
    message = Mail(
        from_email="sebastian@goopim.com",
        to_emails=to,
        subject="New Message Notification",
        html_content=html)

    try:
        # Initialize the SendGrid client with your API key
        sg = SendGridAPIClient(app.config['SENDGRID_API_KEY'])

        # Send the email using the SendGrid client
        response = sg.send(message)

        # Return a success message if the email was sent successfully
        return 'Webhook received successfully'
    except Exception as e:
        # Return an error message if there was an issue sending the email
        return 'Error sending email: ' + str(e)


#fkfkfkk
@app.route("/requestnewverificationlink/<string:email>", methods=['POST'])
def send_verification_request(email=None):
    email_response = ""

    if email is None:
        email = request.form.get('email')

    if not email:
        # Handle the case when email is not provided
        return "Email parameter is missing"
    # Check if the email exists in the users table
    user = Users.query.filter_by(email=email).first()
    if user:
        # Generate a 100-character random code
        public_id= user.public_id
        code = generate_random_code(100)
        VERIFICATION_URL= f'{FRONTEND_URL}/verifyemail?pid={public_id}&token={code}'
        to =user.email
        # Render the HTML template with dynamic values
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <style>
            /* Mobile-first responsive styles */
            @media screen and (max-width: 480px) {{
              .container {{
                width: 100%;
                padding: 10px;
              }}

              .message {{
                font-size: 16px;
              }}
            }}
          </style>
        </head>
        <body>
          <div class="container" style="max-width: 600px; margin: 0 auto;">
            <h1>Verify your email on Goopim</h1>
            <p>Hello,</p>
            <p>Click on the link below to verify your email.</p>
            <div class="message">
              <a href={VERIFICATION_URL}>Verify Email</a>
            </div>
            <p>Thank you!</p>
          </div>
        </body>
        </html>
        """

        # msg = Message("Verify email", sender="sebastian@goopim.com", recipients=[to])
        # msg.html = html
        #
        # mail.send(msg)
        message = Mail(
            from_email="sebastian@goopim.com",
            to_emails=to,
            subject="Verify email",
            html_content=html)

        try:
            # Initialize the SendGrid client with your API key
            sg = SendGridAPIClient(app.config['SENDGRID_API_KEY'])

            # Send the email using the SendGrid client
            response = sg.send(message)

            # Return a success message if the email was sent successfully
            email_response='Webhook received successfully'
        except Exception as e:
            # Return an error message if there was an issue sending the email
            email_response='Error sending email: ' + str(e)

        # Store the email and code in the Verifyaccount model class
        verify_account = Verifyaccount(email=email, code=code)
        db.session.add(verify_account)
        db.session.commit()
        # Create and send the email


        return f"Verification email sent: {email}"
    else:
        return f"There is no Goopim account with this : {email} "

def generate_random_code(length):
    characters = string.ascii_letters + string.digits
    code = ''.join(random.choice(characters) for _ in range(length))
    return code


@app.route('/verifyemail/<string:public_id>/<string:token>', methods=['POST'])
def verify_email(public_id,token):
    user = Users.query.filter_by(public_id=public_id).first()
    if user:
        email = user.email
        code = token

        user_verification = Verifyaccount.query.filter_by(email=email,code=code,valid=True).first()
        if user_verification:
            user.is_verified = True
            user_verification.valid = False

            db.session.commit()

            return jsonify({'message':"Account is verified"})
        else:
            return  jsonify({'message':'Account already verified'})
    else:
        return  jsonify({'message':"Verification link doesn't exist"})

@app.route("/resetpassword", methods=['POST'])
def send_reset_password_request():
    email_response =""
    data = request.get_json()
    email = data.get('email')

    if not email:
        return jsonify(error="Email parameter is missing"), 400

    # Check if the email exists in the users table
    user = Users.query.filter_by(email=email).first()
    if user:
        # Generate a 100-character random code
        public_id= user.public_id
        code = generate_random_code(100)
        VERIFICATION_URL= f'{FRONTEND_URL}/newpassword?pid={public_id}&token={code}'
        to =user.email
        # Render the HTML template with dynamic values
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <style>
            /* Mobile-first responsive styles */
            @media screen and (max-width: 480px) {{
              .container {{
                width: 100%;
                padding: 10px;
              }}

              .message {{
                font-size: 16px;
              }}
            }}
          </style>
        </head>
        <body>
          <div class="container" style="max-width: 600px; margin: 0 auto;">
            <h1>Reset your password</h1>
            <p>Hello,</p>
            <p>Click on the link below to reset your password.</p>
            <div class="message">
              <a href={VERIFICATION_URL}>Click to reset</a>
            </div>
            <p>Thank you!</p>
          </div>
        </body>
        </html>
        """

        message = Mail(
            from_email="sebastian@goopim.com",
            to_emails=email,
            subject="Reset password",
            html_content=html)

        try:
            # Initialize the SendGrid client with your API key
            sg = SendGridAPIClient(app.config['SENDGRID_API_KEY'])

            # Send the email using the SendGrid client
            response = sg.send(message)
            # print(response.status_code)
            # print(response.body)
            # print(response.headers)

            # Return a success message if the email was sent successfully
            email_response = 'Reset password link sent'
        except Exception as e:
            # Return an error message if there was an issue sending the email
            email_response = 'Error sending email: ' + str(e)


        # Store the email and code in the Verifyaccount model class

        password_account = Resetpassword(email=email, code=code)
        db.session.add(password_account)
        db.session.commit()

        # msg = Message('Reset password', 'mrjohnugbor@gmail.com', recipients=['mrjohnugbor@gmail.com'])
        # msg.html = html
        #
        # mail.send(msg)
        # Create and send the email

        return jsonify({'message':'Password reset email sent'})
    else:
        return jsonify({'message': 'There is no Goopim account with this'})


@app.route('/reset/<string:public_id>/<string:token>', methods=['POST'])
def reset_password(public_id,token):
    user = Users.query.filter_by(public_id=public_id).first()
    data = request.get_json()
    if user:
        email = user.email
        code = token

        user_verification = Resetpassword.query.filter_by(email=email,code=code,valid=True).first()
        if user_verification:
            hashed_password = generate_password_hash(data['password'], method='sha256')
            user.password = hashed_password
            user_verification.valid =False


            db.session.commit()

            return jsonify({'message':"Password reseted"})


        else:
            return  jsonify({'message':'Reset link expired'})
    else:
        return  jsonify({'message':"Reset link expired"})

@app.route('/reviews', methods=['POST'])
@jwt_required()
def create_review():
    data = request.get_json()
    rating = data.get('rating')
    review_text = data.get('review')
    reviewer_id = get_jwt_identity()["id"] #data.get('reviewer_id')
    review_owner = 0
    project_id = data.get('project_id')

    if not rating or not review_text  or not project_id:
        return jsonify({'error': 'Missing required fields'}), 400

    project = Project.query.get(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    if project.user_id == reviewer_id:
        review_owner = project.provider_id
    else:
        review_owner =project.user_id
    if project.user_id == reviewer_id or project.provider_id == reviewer_id:
        if reviewer_id in [review.reviewer_id for review in project.reviews]:
            return jsonify({'error': 'Already reviewed this project'}), 400

        review = Review(rating=rating, review=review_text, reviewer_id=reviewer_id, review_owner=review_owner,
                        project_id=project_id)
        db.session.add(review)
        db.session.commit()
        return jsonify({'message': 'Review created successfully'}), 201


    else:
        return jsonify({'error': 'how did you get here'}),

@event.listens_for(Review, 'after_insert')
def update_average_rating(mapper, connection, review):
    review_owner = review.review_owner
    reviews = Review.query.filter_by(review_owner=review_owner).all()
    total_rating = sum(review.rating for review in reviews)
    average_rating = total_rating / len(reviews)

    user = Users.query.filter_by(id=review_owner).first()
    user.rating = average_rating
    user.total_number_of_raters = len(reviews)
    db.session.commit()


@app.route('/admindashboard', methods=['GET'])
@jwt_required()
def goopim_admindashboard():
    current_user = get_jwt_identity()['id']

    if is_admin(current_user):
        pending_withdrawal_count = WithdrawalRequest.query.filter_by(status=WithdrawalStatus.PENDING).count()
        processing_withdrawal_count = WithdrawalRequest.query.filter_by(status=WithdrawalStatus.PROCESSING).count()
        completed_withdrawal_count = WithdrawalRequest.query.filter_by(status=WithdrawalStatus.COMPLETED).count()
        total_balance = db.session.query(db.func.sum(MyPaymentAccount.balance)).scalar()
        total_projects = Project.query.count()
        total_contract_amount = db.session.query(db.func.sum(Project.contract_amount)).scalar()
        total_users = Users.query.count()

        accepted_contract_count = Contract.query.filter_by(contract_status=ContractStatus.ACCEPTED).count()
        rejected_contract_count = Contract.query.filter_by(contract_status=ContractStatus.CANCELLED).count()
        pending_contract_count = Contract.query.filter_by(contract_status=ContractStatus.PENDING).count()

        return jsonify({
            'pending_withdrawal_count': pending_withdrawal_count,
            'processing_withdrawal_count': processing_withdrawal_count,
            'completed_withdrawal_count': completed_withdrawal_count,
            'total_account_balance': total_balance,
            'total_projects': total_projects,
            'total_contract_amount': total_contract_amount,
            'total_users': total_users,
            'accepted_contract_count': accepted_contract_count,
            'rejected_contract_count': rejected_contract_count,
            'pending_contract_count': pending_contract_count
        })

    else:
        return jsonify({"not authorized"})



# app = FastAPI()


class Providers(BaseModel):
    name: str
    description: str
    rating: str
    portfolio: str
    keywords: List[str]
    hourly_rate: str
    username: str
    profile_url: HttpUrl
    provider_id: str
    profile_picture: str
    country: str
    city: str


    @validator("rating")
    def validate_rating(cls, value):
        if float(value) < 0 or float(value) > 5:
            raise ValueError("Rating should be between 0 and 5")
        return value


class ClientQuery(BaseModel):
    query: str


@app.route("/add_or_update_user", methods=["POST"])
def add_or_update_user():
    try:

        provider =request.get_json()
        provider_data = provider["data"]
        print(f"{provider_data}")

        # Call the function from helpers.py ...
        add_or_update_user_function(
            provider_data, embedder, index)

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.route("/find_recommended_providers", methods=["POST"])
def find_recommended_providers():
    provider_data = request.get_json()


    try:
        query_data =  provider_data["project_description"] #provider.dict()

        print(query_data)
        # Call the function from helpers.py
        results = find_recommended_providers_function(
            query_data, embedder, index)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vector_db_status")
def vector_db_status():
    """
    Endpoint to get the status of the vector database.

    Returns:
        dict: A dictionary containing the index statistics.
        Basically how many vectors are there in vector db.

    Raises:
        HTTPException: If there is an error retrieving the index statistics.
    """
    try:
        return index.describe_index_stats().to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.route('/health', methods=["GET"])
def health_check():
    return jsonify({"result": "deployment complete"})


if __name__ == "__main__":
    app.run(port=5000)
    # socketio.run(app, port=5000)
