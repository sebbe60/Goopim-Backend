import db_util
from app.db_util import get_rows, new_row_returning_id, connect_to_database
from app import logd

def main():
    connect_to_database()


if __name__ == '__main__':
    main()
