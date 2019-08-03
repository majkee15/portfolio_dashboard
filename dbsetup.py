import outils
from data_fetcher import Cursor
from base import Base
import os


class DBsetup(Base):

    def __init__(self,database ):
        super().__init__(__class__.__name__)
        self.database = 'stock_prices_eod.sqlite3'
        self.cursor = Cursor(self.database)
        self.commands = outils.commands
        if database in os.listdir():
            self.exists = True
        else:
            self.exists = False

    def setup_database(self):
        if self.exists:
            self.logger.warning(f"Database '{self.database}' already exists in the current directory: '{os.getcwd()}'")
        else:
            with self.cursor as cursor:
                for command in self.commands:
                    cursor.executescript(command)



if __name__ == '__main__':
    database = 'stock_prices_eod.sqlite3'
    db = DBsetup(database)
    db.setup_database()