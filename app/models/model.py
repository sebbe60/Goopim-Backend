# import os
# import re
from datetime import datetime
from flask_security import RoleMixin, UserMixin
from flask_sqlalchemy import SQLAlchemy
# from sqlalchemy import case,event
# from sqlalchemy.dialects.postgresql import UUID
# from sqlalchemy.ext.hybrid import hybrid_property
# #from sqlalchemy.util import text_type
from enum import Enum
from sqlalchemy_serializer import SerializerMixin

import json

# from app.app import db
db = SQLAlchemy()


# db = SQLAlchemy()

class MapUsersToRoles(db.Model):
    __tablename__ = 'map_users_to_roles'
    id = db.Column(db.Integer, db.Identity(), primary_key=True, doc="ID")
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    role_id = db.Column(db.Integer, db.ForeignKey('roles.id'), nullable=False)
    user = db.relationship('Users', info={"view_as": "scalar"})
    role = db.relationship('Role', info={"view_as": "scalar"})


class Role(db.Model, RoleMixin):
    __tablename__ = 'roles'
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)
    description = db.Column(db.String(255))
    user = db.relationship("Users", secondary=MapUsersToRoles.__tablename__, back_populates="roles",
                           info={"view_as": "complex"})

    creation_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    last_updated = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())

    def repr_select_list(self):
        return f"{self.name}"


past_companies = db.Table(
    'pastcompanies',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('company_id', db.Integer, db.ForeignKey('companies.id'), primary_key=True)
)


class Users(db.Model,UserMixin):
   __tablename__ = 'user'
   id = db.Column(db.Integer, db.Identity(), primary_key=True, doc="ID")
   public_id = db.Column(db.String(1000))
   first_name = db.Column(db.String(200))
   last_name = db.Column(db.String(200))
   goopim_username = db.Column(db.String(200), nullable=True, unique=True)

   email = db.Column(db.String(200), unique=True,nullable=False) 
   password = db.Column(db.String(1000), nullable=True)

   roles = db.relationship("Role", secondary=MapUsersToRoles.__tablename__, back_populates="user",
                            info={"view_as": "complex"})
   is_verified = db.Column(db.Boolean, default=False)
   active = db.Column(db.Boolean())
   is_goopim_admin = db.Column(db.Boolean())
   is_google_user = db.Column(db.Boolean())
   is_user =db.Column(db.Boolean, default=False)
   is_provider=db.Column(db.Boolean, default=False)
   profile_img_url = db.Column(db.String(2000),nullable=True)
   profile_cover_url = db.Column(db.String(2000),nullable=True)
   description = db.Column(db.Text,nullable=True)
   keyword =db.Column(db.Text,nullable=True)
   portfolio =db.Column(db.Text,nullable=True)
   rating = db.Column(db.Float, default=0.0)
   hourly_rate= db.Column(db.Float, default=0.0)
   is_profile_complete =db.Column(db.Boolean(), default=False)
   total_number_of_raters = db.Column(db.Integer, default=0)
   total_rated =  db.Column(db.Integer, default=0)
   last_login = db.Column(db.DateTime(timezone=True))
   total_login_count = db.Column(db.Integer, nullable=False, default=0)
   address = db.relationship('Address', backref='user', uselist=False)
   past_companies = db.relationship('Companies', secondary=past_companies, backref='users')
   creation_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
   last_updated = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())
#don't remove id or public id from the dict, used in some api access, used in is_admin method
   def to_dict(self):
       address_country = self.address.country if self.address else None
       address_city =self.address.city if self.address else None
       return {
           'first_name':self.first_name,
           'last_name':self.last_name,
           'id':self.id,

           'email': self.email,
           'public_id':self.public_id,
           'is_freelancer':self.is_provider,
           'is_an_employer':self.is_user,
           'is_controller':self.is_goopim_admin,
           'profile_img_url':self.profile_img_url,
       'description':self.description,
       'keyword':self.keyword,
       'portfolio':self.portfolio,
       'username':self.goopim_username,

       'hourly_rate':self.hourly_rate,
       'is_profile_complete':self.is_profile_complete,
           'address_country': address_country,
           'address_city': address_city,
           'joined':self.creation_date,
           'profile_cover_url':self.profile_cover_url,
           'rating':self.rating,
           'number_of_reviews':self.total_number_of_raters

       }
class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    rating = db.Column(db.Integer, nullable=False)
    review = db.Column(db.String(500), nullable=False)
    reviewer_id = db.Column(db.Integer, nullable=False)
    review_owner = db.Column(db.Integer, nullable=False)
    project_id= db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    #project = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"<Review {self.id}>"


class Address(db.Model):
    __tablename__ = 'address'
    id = db.Column(db.Integer, primary_key=True)
    country = db.Column(db.String(1000))
    city = db.Column(db.String(1000))
    street1 = db.Column(db.String(1000))
    street2 = db.Column(db.String(1000))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Verifyaccount(db.Model):
    __tablename__ = 'verifyaccount'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(1000))
    code = db.Column(db.String(1000))
    valid =db.Column(db.Boolean(), default=True)
    creation_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    last_updated = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())


class Resetpassword(db.Model):
    __tablename__ = 'resetpassword'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(1000))
    code = db.Column(db.String(1000))
    valid =db.Column(db.Boolean(), default=True)
    creation_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    last_updated = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())

class Companies(db.Model):
    __tablename__ = 'companies'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(1000))
    logo_url = db.Column(db.String(1000))

    def serialize(self):
        return {
            "id":self.id,
            "name":self.name,
            "logo_url":self.logo_url
        }

class Project(db.Model,SerializerMixin):
    __tablename__ = 'project'
    id = db.Column(db.Integer, primary_key=True)
    project_public_id =db.Column(db.String(1000))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    provider_id = db.Column(db.Integer, nullable=True)
    contract_amount = db.Column(db.Integer, nullable=True)
    milestone_escrow_accounts = db.relationship('MilestoneEscrowAccount', backref='project_escrow')
    title = db.Column(db.String(1000), nullable=True)
    description = db.Column(db.Text, nullable=True)

    quotes = db.relationship('Quote', backref='project_quotes')
    timelines = db.relationship('Timeline', backref='project_timelines')
    deliverables = db.relationship('Deliverable', backref='project_deliverables')
    reviews = db.relationship('Review', backref='project')


    creation_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    last_updated = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())
   
    def update_is_provider(self, provider_id):
        self.provider_id = provider_id
        db.session.commit()
    def update_contract_amount(self, contract_amount):
        self.contract_amount = contract_amount
        db.session.commit()
    
    def serialize(self):
        return {
            'id': self.id,
            'project_public_id': self.project_public_id,
            'user_id': self.user_id,
            'provider_id': self.provider_id,
           
            'contract_amount': self.contract_amount ,
            'title':self.title,
            'description':self.description,
            'created_at': self.creation_date,
            'updated_at': self.last_updated
      


        }

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user1_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user2_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    user1 = db.relationship('Users', foreign_keys=[user1_id], backref='conversations1')
    user2 = db.relationship('Users', foreign_keys=[user2_id], backref='conversations2')
    creation_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    last_updated = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())

    # define a relationship to the Message model
    messages = db.relationship('Message', backref='conversation', lazy=True)

    # def last_message(self):
    #     messages = self.messages.all()
    #     return messages[-1] if messages and messages.count() > 0 else None
    def last_message(self):
        messages = list(self.messages)
        if messages:
            return messages[-1]
        return None
    def to_dict(self):
        return {
            'id': self.id,
            'user1_id': self.user1_id,
            'user2_id': self.user2_id,
            'creation_date': self.creation_date,
            'last_updated': self.last_updated,
            'last_message': self.last_message().to_dict() if self.last_message() else None
        }
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(1000))
    type = db.Column(db.String(20), default='normal')
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)

    sender = db.relationship('Users', foreign_keys=[sender_id], backref='sent_messages')
    receiver = db.relationship('Users', foreign_keys=[receiver_id], backref='received_messages')
    creation_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    last_updated = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())
    def serialize(self):
        return {
            'id': self.id,
            'text': self.text,
            'type':self.type,
            'sender_id':self.sender_id,
            'receiver_id':self.receiver_id,
            'conversation_id':self.conversation_id
        }
    def to_dict(self):
        return {
            'id': self.id,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'text': self.text,
            'timestamp': self.last_updated,
            'type': self.type
        }
    @classmethod
    def create(cls, text, type, sender, receiver):
        conversation = Conversation.query.filter(
            db.or_(
                db.and_(Conversation.user1 == sender, Conversation.user2 == receiver),
                db.and_(Conversation.user1 == receiver, Conversation.user2 == sender)
            )
        ).first()

        if conversation is None:
            if not sender.is_provider:
                raise Exception('Only providers can start a conversation')
            conversation = Conversation(user1=sender, user2=receiver)
            db.session.add(conversation)
            db.session.commit()

        message = cls(text=text, type=type, sender_id=sender, receiver_id=receiver, conversation_id=conversation.id)
        db.session.add(message)
        db.session.commit()

        return message
class TransactionType(Enum):
    DEPOSIT = 'deposit'
    WITHDRAWAL = 'withdrawal'
    MILESTONE = 'milestone'

class WithdrawalStatus(Enum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'

class ContractStatus(Enum):
    PENDING = 'pending'
    ACCEPTED = 'accepted'
    CANCELLED = 'cancelled'

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    transaction_type = db.Column(db.Enum(TransactionType), nullable=False)

    user = db.relationship('Users', backref='transactions')
    def serialize(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'amount':self.amount,
            'transaction_type':self.transaction_type
        }

class MyPaymentAccount(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    balance = db.Column(db.Float, nullable=False, default=0.0)

    user = db.relationship('Users', backref='payment_account')
    creation_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    last_updated = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())

    def serialize(self):
        return {
            'id': self.id,
            'balance': self.balance
        }

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class WithdrawalRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    status = db.Column(db.Enum(WithdrawalStatus), default='pending', nullable=False)

    user = db.relationship('Users', backref='pending_withdrawals')
    creation_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    last_updated = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())

    @classmethod
    def create_new_withdrawal_request(cls, user_id, amount):
        new_withdrawal_request = cls(user_id=user_id, amount=amount, status=WithdrawalStatus.PENDING)
        db.session.add(new_withdrawal_request)
        db.session.commit()
        return new_withdrawal_request.serialize()
    def serialize(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'amount':self.amount,
            "status":self.status,
            "creation_date":self.creation_date
        }
class MilestoneEscrowAccount(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    provider_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    completed = db.Column(db.Boolean, default=False)
    milestone = db.relationship('Milestone', backref='milestone_escrow_account_as_milestone')
    project_milestones = db.relationship('Project', backref='escrow_accounts_as_project')
    user = db.relationship('Users', foreign_keys=[user_id], backref='milestone_escrow_accounts_as_user')
    provider = db.relationship('Users', foreign_keys=[provider_id], backref='milestone_escrow_accounts_as_provider')
    creation_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    last_updated = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())
    def serialize(self):
        return {
            'id': self.id,
            'project_id': self.project_id,
            'user_id':self.user_id,
            'provider_id':self.provider_id,
            'amount':self.amount,
            'completed':self.completed
        }
    def create_initial_milestone(self):
        milestone = Milestone(milestone_escrow_account=self,
                              user=self.user,
                              provider=self.provider,
                              milestone_amount=self.amount,
                              milestone_description="Initial milestone")
        db.session.add(milestone)
        db.session.commit()
    def create_more_milestone(self,milestone_description):
        milestone = Milestone(milestone_escrow_account=self,
                              user=self.user,
                              provider=self.provider,
                              milestone_amount=self.amount,
                              milestone_description=milestone_description)
        db.session.add(milestone)
        db.session.commit()
class Milestone(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    milestone_escrow_account_id = db.Column(db.Integer, db.ForeignKey('milestone_escrow_account.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    provider_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    milestone_amount = db.Column(db.Float, nullable=False)
    milestone_description = db.Column(db.String(255), nullable=False)

    milestone_escrow_account = db.relationship('MilestoneEscrowAccount', backref='milestones')
    user = db.relationship('Users', foreign_keys=[user_id], backref='milestones_as_user')
    provider = db.relationship('Users', foreign_keys=[provider_id], backref='milestones_as_provider')
    creation_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    last_updated = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())

    def serialize(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'provider_id':self.provider_id,
            'milestone_amount':self.milestone_amount,
            'milestone_description':self.milestone_description
        }
class Quote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    provider_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    quote_details = db.Column(db.String(500), nullable=False)
    quote_amount = db.Column(db.Float, nullable=False)
    message_id = db.Column(db.Integer, db.ForeignKey('message.id'), nullable=False)

    project_quote = db.relationship('Project', backref='quote')
    provider = db.relationship('Users', foreign_keys=[provider_id], backref='provider_quotes')
    user = db.relationship('Users', foreign_keys=[user_id], backref='user_quotes')
    message = db.relationship('Message', backref='quote')
    creation_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    last_updated = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())

    def serialize(self):
        return {
            'id': self.id,
            'project_id': self.project_id,
            'provider_id':self.provider_id,
            'user_id':self.user_id,
            'quote_details':self.quote_details,
            'quote_amount':self.quote_amount,
            'message_id':self.message_id
        }
# class Contract(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     #project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     provider_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     contract_details = db.Column(db.String(200), nullable=False)
#     contract_amount = db.Column(db.Float, nullable=False)
#     contract_status = db.Column(db.Enum(ContractStatus), default='pending')
#     message_id = db.Column(db.Integer, default=0)
#     conversation_id = db.Column(db.Integer, default=0)
#
#     #project = db.relationship('Project', backref='contracts')
#     user = db.relationship('Users', foreign_keys=[user_id], backref='contracts_as_user')
#     provider = db.relationship('Users', foreign_keys=[provider_id], backref='contracts_as_provider')
#     creation_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
#     last_updated = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())
#
#     def serialize(self):
#         return {
#             'id': self.id,
#             'provider_id':self.provider_id,
#             'user_id':self.user_id,
#             'contract_details':self.contract_details,
#             'contract_amount':self.contract_amount,
#             'contract_status':self.contract_status,
#             'message_id':self.message_id
#         }

class Contract(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    provider_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    contract_title = db.Column(db.String(200), nullable=False)
    contract_details = db.Column(db.String(200), nullable=False)
    contract_amount = db.Column(db.Float, nullable=False)
    contract_status = db.Column(db.Enum(ContractStatus), default='pending')



    #project = db.relationship('Project', backref='contracts')
    user = db.relationship('Users', foreign_keys=[user_id], backref='contracts_as_user')
    provider = db.relationship('Users', foreign_keys=[provider_id], backref='contracts_as_provider')
    creation_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    last_updated = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())

    def serialize(self):
        return {
            'id': self.id,
            'provider_id':self.provider_id,
            'user_id':self.user_id,
            'title':self.contract_title,
            'contract_details':self.contract_details,
            'contract_amount':self.contract_amount,
            'contract_status':self.contract_status,
            'employer_profile_image_url': self.user.profile_img_url,
            'freelancer_profile_image_url': self.provider.profile_img_url,
            'employer_name': self.user.first_name +" " + self.user.last_name,
            'freelancer_name': self.provider.first_name +" " + self.provider.last_name,
            'create_at':self.creation_date
        }
class Timeline(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    provider_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timeline_description = db.Column(db.String(200), nullable=False)
    timeline_time = db.Column(db.DateTime, nullable=True, default=datetime.utcnow)

    project_timeline = db.relationship('Project', backref='timeline')
    user = db.relationship('Users', foreign_keys=[user_id], backref='timelines_as_user')
    provider = db.relationship('Users', foreign_keys=[provider_id], backref='timelines_as_provider')
    creation_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    last_updated = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())

    def serialize(self):
        return {
            'id': self.id,
            'provider_id':self.provider_id,
            'user_id':self.user_id,
            'project_id':self.project_id,
            'timeline_description':self.timeline_description,
            'timeline_time':self.timeline_time
        }
class Deliverable(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    provider_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    description = db.Column(db.String(200), nullable=False)

    project_deliverable = db.relationship('Project', backref='deliverable')
    user = db.relationship('Users', foreign_keys=[user_id], backref='deliverables_as_user')
    provider = db.relationship('Users', foreign_keys=[provider_id], backref='deliverables_as_provider')
    def serialize(self):
        return {
            'id': self.id,
            'provider_id':self.provider_id,
            'user_id':self.user_id,
            'project_id':self.project_id,
            'description':self.description

        }

class SearchLog(db.Model):
    __tablename__ = 'searchlog'
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text())
    creation_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())

    def searchlog_to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'creation_date': self.creation_date.isoformat(),

        }
