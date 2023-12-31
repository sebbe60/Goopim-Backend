"""review project update added reviews

Revision ID: d6fdae8c951b
Revises: d3d10629ef0b
Create Date: 2023-06-14 12:53:20.793485

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd6fdae8c951b'
down_revision = 'd3d10629ef0b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_foreign_key(None, 'review', 'project', ['project'], ['id'])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'review', type_='foreignkey')
    # ### end Alembic commands ###
