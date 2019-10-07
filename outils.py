exchange_table = "DROP TABLE IF EXISTS exchanges;" \
                 "CREATE TABLE exchanges(id INTEGER NOT NULL PRIMARY KEY," \
                 "exchange VARCHAR(32) NOT NULL," \
                 " name VARCHAR(255) NOT NULL," \
                 "city VARCHAR(255) NULL," \
                 "country VARCHAR(255) NULL," \
                 "currency VARCHAR(32) NULL," \
                 "timezone_offset time NULL, " \
                 "created_date datetime NULL DEFAULT CURRENT_TIMESTAMP," \
                 "last_updated datetime NULL DEFAULT CURRENT_TIMESTAMP);" \
                 "CREATE UNIQUE INDEX name_exchange on exchanges ( name, exchange ) ;"

security_table = "DROP TABLE IF EXISTS security;" \
                 "CREATE TABLE security (" \
                 "id INTEGER NOT NULL PRIMARY KEY ," \
                 "exchange_id INT(11) NOT NULL," \
                 "ticker VARCHAR(10) NOT NULL," \
                 "name VARCHAR(255) NULL," \
                 "security_class VARCHAR(32) NOT NULL," \
                 "created DATETIME NULL DEFAULT CURRENT_TIMESTAMP," \
                 "last_updated Datetime NULL DEFAULT CURRENT_TIMESTAMP," \
                 "FOREIGN KEY (exchange_id) REFERENCES exchanges(id) ON DELETE " \
                 "NO ACTION  ON UPDATE NO ACTION);" \
                 "CREATE INDEX exchange_id on security(exchange_id ASC);" \
                 "CREATE UNIQUE INDEX ticker on security(ticker ASC)"

data_vendor_table = "DROP TABLE IF EXISTS data_vendor;" \
                    "CREATE TABLE IF NOT EXISTS data_vendor (" \
                    "id INTEGER NOT NULL PRIMARY KEY," \
                    "name VARCHAR(32) NOT NULL," \
                    "website_url VARCHAR(255) NULL DEFAULT NULL," \
                    "created DATETIME NULL DEFAULT CURRENT_TIMESTAMP," \
                    "last_updated Datetime NULL DEFAULT CURRENT_TIMESTAMP);"

daily_price_table = "DROP TABLE IF EXISTS daily_price;" \
                    "CREATE TABLE  IF NOT EXISTS daily_price(" \
                    "id INTEGER NOT NULL PRIMARY KEY ," \
                    "data_vendor_id INT(11) NOT NULL," \
                    "ticker_id INT(11) NOT NULL," \
                    "price_date DATE NOT NULL," \
                    "created_date DATETIME NULL DEFAULT CURRENT_TIMESTAMP," \
                    "last_updated DATETIME NULL DEFAULT CURRENT_TIMESTAMP," \
                    "open_price DECIMAL(11,6) NULL DEFAULT NULL," \
                    "close_price DECIMAL(11,6) NULL DEFAULT NULL," \
                    "high_price DECIMAL(11,6) NULL DEFAULT NULL," \
                    "low_price DECIMAL(11,6) NULL DEFAULT NULL," \
                    "adj_close_price DECIMAL(11,6) NULL DEFAULT NULL," \
                    "volume DECIMAL(11,6) NULL DEFAULT NULL," \
                    "FOREIGN KEY (ticker_id) REFERENCES security(id)," \
                    " FOREIGN KEY (data_vendor_id) REFERENCES data_vendor(id)" \
                    "ON DELETE NO ACTION ON UPDATE NO ACTION);" \
                    "CREATE INDEX price_date on daily_price(price_date ASC);" \
                    "CREATE INDEX ticker_id on daily_price(ticker_id ASC);" \
                    "CREATE UNIQUE INDEX ticer_date on daily_price(ticker_id, price_date)"

setup_vendors = "INSERT INTO data_vendor (name, website_url) VALUES " \
                "('YahooFinance', 'https://finance.yahoo.com')" \
                ""
foreign_keys = "PRAGMA foreign_keys = ON;"

YAHOO_VENDOR_ID = 1

commands = [exchange_table, security_table, data_vendor_table, daily_price_table, setup_vendors, foreign_keys]

# Assets
rp_all_weather = ['SSO', 'QLD', 'VWO', 'IAU', 'TLT', 'IEF', 'MBG']
american_rocket = ['TMF', 'TQQQ', 'UPRO', 'UGLD']
new_balanced = ['MTUM', 'BTAL']
benchmark = ['SPY']